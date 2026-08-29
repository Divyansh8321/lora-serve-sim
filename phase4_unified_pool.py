"""PHASE 4 -- Unify the pools. Now coordination has something to trade.

WORLD: adapters and KV caches share ONE MB-denominated pool. This is what
       ELORA / FastLibra (HPCA 2026) propose, and it is NOT what vLLM does by
       default. The change is architectural, not a policy tweak.

WHY IT MATTERS: in phases 2-3 each pool had fixed, independent capacity, so
       an eviction decision could only choose WHICH item of a given type to
       drop. Total reloads were fixed by the slab size, which is why the
       cross-pool signal moved the staleness metric but not latency.

       Unification creates a decision that did not previously exist: which
       TYPE to evict. And the two types differ by more than an order of
       magnitude -- an adapter is ~20MB, a 2000-token conversation is ~250MB.
       Now the choice is worth making.

QUESTION: once the pools are unified, how much does the eviction decision
       actually buy, and which part of the "smart" logic is doing the work?

FINDING: 27% lower p50 at high pressure, with stale KV falling from ~46% to 0.
       But the mechanism is simpler than expected. An uncoordinated policy
       that merely checks KV BEFORE adapters performs identically to the full
       dependency-aware policy at every pressure level tested -- because
       evicting one 250MB conversation frees what a dozen adapter evictions
       would, so the ordering alone avoids the thrash. The orphan-count and
       stale-priority refinements never get exercised in this workload.

       SO: unification is the enabler; ordering discipline captures nearly all
       of the benefit; elaborate dependency scoring adds little on top.

Run:  python phase4_unified_pool.py
"""


from collections import deque
from core import (ADAPTER_MB, MB_PER_TOKEN, PREFILL_MS_PER_TOKEN,
                        DECODE_MS_PER_TOKEN, SWAP_COLD_MS,
                        make_multi_turn_workload, percentile)


class UnifiedCache:
    """One MB-denominated pool holding both adapters and KV caches.
    Stores each KV entry's owning adapter -- a fact, not a judgement."""

    def __init__(self, capacity_mb):
        self.capacity_mb = capacity_mb
        self.adapters = {}          # adapter_id -> size_mb
        self.conversations = {}     # conversation_id -> {size_mb, adapter_id}

    def is_resident_adapter(self, a): return a in self.adapters
    def is_resident_kv(self, c): return c in self.conversations
    def kv_size(self, c): return self.conversations[c]["size_mb"]
    def kv_owner(self, c): return self.conversations[c]["adapter_id"]

    def used_mb(self):
        return (sum(self.adapters.values())
                + sum(e["size_mb"] for e in self.conversations.values()))

    def free_mb(self): return self.capacity_mb - self.used_mb()
    def can_fit(self, mb): return mb <= self.free_mb()

    def add_adapter(self, a, mb):
        if a in self.adapters: return
        self.adapters[a] = mb

    def delete_adapter(self, a):
        if a not in self.adapters:
            raise RuntimeError("Adapter not resident - cannot evict")
        del self.adapters[a]

    def add_kv(self, c, mb, a):
        self.conversations[c] = {"size_mb": mb, "adapter_id": a}

    def delete_kv(self, c):
        if c not in self.conversations:
            raise RuntimeError("Conversation not resident - cannot evict")
        del self.conversations[c]

    def stale_percent(self):
        """Fraction of resident KV memory that is not currently usable because
        its adapter has been evicted. NOT permanently wasted -- ~60% of it gets
        rescued when the adapter returns -- but it is an opportunity cost."""
        total = sum(e["size_mb"] for e in self.conversations.values())
        if total == 0: return 0.0
        stale = sum(e["size_mb"] for e in self.conversations.values()
                    if e["adapter_id"] not in self.adapters)
        return 100.0 * stale / total


class BasePolicy:
    """Shared scheduling. Subclasses differ ONLY in eviction, so the comparison
    isolates the eviction decision."""

    def __init__(self, wait_threshold=30000):
        self.adapter_memory = {}
        self.kv_memory = {}
        self.wait_threshold = wait_threshold

    def choose_next(self, pending, cache, now):
        waiting = [a for a, q in pending.items() if q]
        most_starved = max(waiting, key=lambda a: now - pending[a][0].arrival_time)
        if now - pending[most_starved][0].arrival_time > self.wait_threshold:
            return most_starved
        resident = [a for a in waiting if cache.is_resident_adapter(a)]
        return max(resident if resident else waiting, key=lambda a: len(pending[a]))

    def record(self, adapter_id, conversation_id, now):
        self.adapter_memory[adapter_id] = now
        self.kv_memory[conversation_id] = now


class SeparatePolicy(BasePolicy):
    """The baseline: two independent LRUs that never consult each other.
    Frees adapters first, then KV. Each rule is locally sensible; the failure
    is that they do not coordinate."""
    name = "separate"

    def choose_evictions(self, cache, needed_mb, prot_a, prot_kv):
        ev, freed = [], 0
        for a in sorted([x for x in cache.adapters if x != prot_a],
                        key=lambda x: self.adapter_memory.get(x, -1)):
            if freed >= needed_mb: break
            ev.append(("adapter", a)); freed += cache.adapters[a]
        for c in sorted([x for x in cache.conversations if x != prot_kv],
                        key=lambda x: self.kv_memory.get(x, -1)):
            if freed >= needed_mb: break
            ev.append(("kv", c)); freed += cache.kv_size(c)
        return ev


class JointPolicy(BasePolicy):
    """Dependency-aware. Three tiers of preference:
      1. KV that is ALREADY stale (its adapter is gone) -- free money
      2. cold KV (boulders: frees a lot per eviction)
      3. adapters last, ranked by how many live KV caches they would orphan"""
    name = "joint"

    def choose_evictions(self, cache, needed_mb, prot_a, prot_kv):
        ev, freed = [], 0
        stale = [c for c in cache.conversations
                 if c != prot_kv and cache.kv_owner(c) not in cache.adapters]
        for c in sorted(stale, key=lambda x: self.kv_memory.get(x, -1)):
            if freed >= needed_mb: return ev
            ev.append(("kv", c)); freed += cache.kv_size(c)

        chosen = set(i for k, i in ev if k == "kv")
        cold = [c for c in cache.conversations if c != prot_kv and c not in chosen]
        for c in sorted(cold, key=lambda x: self.kv_memory.get(x, -1)):
            if freed >= needed_mb: return ev
            ev.append(("kv", c)); freed += cache.kv_size(c); chosen.add(c)

        orphans = {}
        for c in cache.conversations:
            if c in chosen: continue
            o = cache.kv_owner(c)
            orphans[o] = orphans.get(o, 0) + 1
        for a in sorted([x for x in cache.adapters if x != prot_a],
                        key=lambda x: (orphans.get(x, 0), self.adapter_memory.get(x, -1))):
            if freed >= needed_mb: break
            ev.append(("adapter", a)); freed += cache.adapters[a]
        return ev


class Simulator:
    def __init__(self, requests, conversations, cache, policy,
                 max_batch=16, swap_cost=SWAP_COLD_MS):
        self.cache = cache
        self.policy = policy
        self.max_batch = max_batch
        self.swap_cost = swap_cost
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}
        for r in self.incoming:
            if r.conversation_id not in self.convos:
                raise ValueError("Request references a conversation that wasn't supplied")
        self.now = 0.0
        self.pending = {}
        self.finished = []
        self.swaps = 0
        self.recompute_tokens = 0
        self.stale_samples = []

    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            r = self.incoming.popleft()
            a = self.convos[r.conversation_id].adapter_id
            self.pending.setdefault(a, deque()).append(r)

    def pending_is_empty(self): return not any(self.pending.values())

    def apply_evictions(self, evictions):
        for kind, i in evictions:
            if kind == "adapter": self.cache.delete_adapter(i)
            else: self.cache.delete_kv(i)

    def make_room(self, mb, prot_a, prot_kv):
        if self.cache.can_fit(mb): return
        need = mb - self.cache.free_mb()
        self.apply_evictions(self.policy.choose_evictions(self.cache, need, prot_a, prot_kv))

    def ensure_adapter(self, a):
        if self.cache.is_resident_adapter(a): return False
        self.make_room(ADAPTER_MB, a, None)
        if not self.cache.can_fit(ADAPTER_MB): return False
        self.cache.add_adapter(a, ADAPTER_MB)
        self.swaps += 1
        return True

    def ensure_kv(self, cid, prot_a):
        convo = self.convos[cid]
        if convo.kv_cache_size <= 0: return 0
        if self.cache.is_resident_kv(cid): return 0
        mb = convo.kv_cache_size * MB_PER_TOKEN
        self.make_room(mb, prot_a, cid)
        if not self.cache.can_fit(mb): return 0
        self.cache.add_kv(cid, mb, convo.adapter_id)
        return convo.kv_cache_size          # tokens that must be rebuilt

    def grow(self, r, prot_a):
        convo = self.convos[r.conversation_id]
        convo.grow_cache(r.output_tokens)
        if self.cache.is_resident_kv(r.conversation_id):
            delta = r.output_tokens * MB_PER_TOKEN
            self.make_room(delta, prot_a, r.conversation_id)
            if self.cache.can_fit(delta):
                self.cache.conversations[r.conversation_id]["size_mb"] = \
                    convo.kv_cache_size * MB_PER_TOKEN

    def run(self):
        while self.incoming or not self.pending_is_empty():
            self.admit_arrivals()
            if self.pending_is_empty():
                if not self.incoming: break
                self.now = self.incoming[0].arrival_time
                continue

            self.stale_samples.append(self.cache.stale_percent())
            adapter = self.policy.choose_next(self.pending, self.cache, self.now)
            swapped = self.ensure_adapter(adapter)

            q = self.pending[adapter]
            batch = [q.popleft() for _ in range(min(self.max_batch, len(q)))]
            recompute = sum(self.ensure_kv(r.conversation_id, adapter) for r in batch)
            self.recompute_tokens += recompute

            mx = max(r.output_tokens for r in batch)
            duration = (25 + len(batch) * 2
                        + mx * DECODE_MS_PER_TOKEN
                        + recompute * PREFILL_MS_PER_TOKEN
                        + (self.swap_cost if swapped else 0))
            self.now += duration
            for r in batch:
                r.finish(self.now - r.arrival_time)
                self.finished.append(r)
                self.grow(r, adapter)
                self.policy.record(adapter, r.conversation_id, self.now)
        return self.finished


# --- named operating points (see README for justification) ---
OPERATING_POINTS = [
    ("over-provisioned",   1200),
    ("cost-optimised",      800),
    ("aggressively-packed", 400),
]
SEEDS = 5


def trial(capacity_mb, PolicyCls, seed):
    convos, reqs = make_multi_turn_workload(n_conversations=40, n_adapters=12,
                                            turns_per_convo=5, mean_tokens=40,
                                            rate=0.0006, turn_gap=6000, seed=seed)
    sim = Simulator(reqs, convos, UnifiedCache(capacity_mb), PolicyCls())
    done = sim.run()
    lats = [r.latency() for r in done]
    stale = sum(sim.stale_samples) / len(sim.stale_samples) if sim.stale_samples else 0.0
    return percentile(lats, 50), percentile(lats, 95), sim.swaps, stale


class SeparateKVFirst(BasePolicy):
    """Ablation: still uncoordinated (two blind LRUs), but checks KV before
    adapters. Isolates how much of the win is ORDERING vs dependency logic."""
    name = "kv-first"

    def choose_evictions(self, cache, needed_mb, prot_a, prot_kv):
        ev, freed = [], 0
        for c in sorted([x for x in cache.conversations if x != prot_kv],
                        key=lambda x: self.kv_memory.get(x, -1)):
            if freed >= needed_mb: break
            ev.append(("kv", c)); freed += cache.kv_size(c)
        for a in sorted([x for x in cache.adapters if x != prot_a],
                        key=lambda x: self.adapter_memory.get(x, -1)):
            if freed >= needed_mb: break
            ev.append(("adapter", a)); freed += cache.adapters[a]
        return ev


def main():
    print(__doc__)
    print(f"{'operating point':>22} {'pool':>6} {'policy':>10} {'p50':>8} {'p95':>9}"
          f" {'swaps':>7} {'stale KV':>9}")
    print("-" * 78)
    for label, cap in OPERATING_POINTS:
        res = {}
        for P in [SeparatePolicy, SeparateKVFirst, JointPolicy]:
            vals = [trial(cap, P, s) for s in range(SEEDS)]
            p50 = sum(v[0] for v in vals) / SEEDS
            p95 = sum(v[1] for v in vals) / SEEDS
            sw = sum(v[2] for v in vals) / SEEDS
            st = sum(v[3] for v in vals) / SEEDS
            res[P.name] = p50
            print(f"{label:>22} {cap:>6} {P.name:>10} {p50:>8.0f} {p95:>9.0f}"
                  f" {sw:>7.0f} {st:>8.1f}%")
        win = 100 * (res['separate'] - res['joint']) / res['separate']
        print(f"{'':>22} {'':>6} {'-> vs adapter-first':>19} {win:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
