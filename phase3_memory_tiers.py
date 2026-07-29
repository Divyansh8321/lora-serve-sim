"""PHASE 3 -- Add a CPU RAM tier. Eviction becomes demotion, not destruction.

WORLD: same as phase 2, but the GPU pool now sits above a CPU RAM tier.
       Evicting no longer destroys: adapters demote to CPU (~30ms to promote
       back instead of ~300ms from disk), KV caches demote to CPU (~0.105
       ms/MB over PCIe instead of a full recompute). When CPU fills, items
       cascade down -- adapters to disk, KV to nothing.

WHY THIS CONFIGURATION: this is what a well-configured deployment looks like.
       Predibase's LoRAX calls it Tiered Weight Caching (GPU -> CPU -> disk);
       LMCache does the same for KV, with Redis/NVMe tiers below.

QUESTION: how much is a dependency-aware policy worth once a CPU tier exists?

FINDING: far less. The gap collapses from ~27% (phase 2, no tier) to ~2-3%
       with a generous CPU tier -- but returns to ~20% when the CPU tier is
       small. THE MEMORY HIERARCHY SUBSTITUTES FOR POLICY INTELLIGENCE.
       Mechanism: the smart policy never demotes adapters (it demotes cheap-
       to-restore KV instead), so its adapters never cascade to disk. The
       naive policy demotes adapters freely and pays 300ms repeatedly.

       PRACTICAL READING: CPU RAM is single-digit dollars per GB; GPU HBM is
       ~$100+/GB amortised. If you can afford the RAM, buy RAM. If you can't,
       the policy is what saves you.

NOTE ON PCIe COST: modelled at 0.105 ms/MB, not the naive 40GB/s bandwidth
       figure of 0.025. vLLM swaps KV in ~128KB chunks and each cudaMemcpyAsync
       carries ~10us of dispatch overhead, which dominates bandwidth at that
       granularity (FastSwitch). Ignoring it underestimates transfer cost 4.2x.

Run:  python phase3_memory_tiers.py
"""

from collections import deque
from mlora.core import (ADAPTER_MB, MB_PER_TOKEN, PREFILL_MS_PER_TOKEN,
                        DECODE_MS_PER_TOKEN, SWAP_COLD_MS, SWAP_WARM_MS,
                        PCIE_MS_PER_MB, make_multi_turn_workload, percentile)


class TieredCache:
    """GPU / CPU / gone. Adapters can always be re-fetched from disk;
    KV caches that fall off CPU must be recomputed."""

    def __init__(self, gpu_mb, cpu_mb):
        self.gpu_mb = gpu_mb
        self.cpu_mb = cpu_mb
        self.adapters = {}      # aid -> 'gpu' | 'cpu'   (absent = disk)
        self.kv = {}            # cid -> {tier, size_mb, adapter_id}

    def gpu_used(self):
        return (sum(ADAPTER_MB for t in self.adapters.values() if t == 'gpu')
                + sum(e['size_mb'] for e in self.kv.values() if e['tier'] == 'gpu'))

    def cpu_used(self):
        return (sum(ADAPTER_MB for t in self.adapters.values() if t == 'cpu')
                + sum(e['size_mb'] for e in self.kv.values() if e['tier'] == 'cpu'))

    def gpu_free(self): return self.gpu_mb - self.gpu_used()
    def cpu_free(self): return self.cpu_mb - self.cpu_used()

    def adapter_tier(self, a): return self.adapters.get(a)
    def kv_tier(self, c): return self.kv[c]['tier'] if c in self.kv else None
    def kv_size(self, c): return self.kv[c]['size_mb'] if c in self.kv else 0
    def kv_owner(self, c): return self.kv[c]['adapter_id'] if c in self.kv else None
    def gpu_adapters(self): return [a for a, t in self.adapters.items() if t == 'gpu']
    def gpu_kv(self): return [c for c, e in self.kv.items() if e['tier'] == 'gpu']

    def place_adapter(self, a, tier): self.adapters[a] = tier
    def drop_adapter(self, a): self.adapters.pop(a, None)
    def place_kv(self, c, mb, aid, tier):
        self.kv[c] = {'tier': tier, 'size_mb': mb, 'adapter_id': aid}
    def drop_kv(self, c): self.kv.pop(c, None)


class BasePolicy:
    def __init__(self, wait_threshold=30000):
        self.am = {}; self.km = {}
        self.wait_threshold = wait_threshold

    def choose_next(self, pending, cache, now):
        waiting = [a for a, q in pending.items() if q]
        ms = max(waiting, key=lambda a: now - pending[a][0].arrival_time)
        if now - pending[ms][0].arrival_time > self.wait_threshold: return ms
        res = [a for a in waiting if cache.adapter_tier(a) == 'gpu']
        return max(res if res else waiting, key=lambda a: len(pending[a]))

    def record(self, a, c, now): self.am[a] = now; self.km[c] = now


class SeparateTiered(BasePolicy):
    """Two independent LRUs. Demotes adapters first, then KV. No coordination."""
    name = "separate"

    def choose_evictions(self, cache, needed, prot_a, prot_kv):
        ev, freed = [], 0
        for a in sorted([x for x in cache.gpu_adapters() if x != prot_a],
                        key=lambda x: self.am.get(x, -1)):
            if freed >= needed: break
            ev.append(('demote_adapter', a)); freed += ADAPTER_MB
        for c in sorted([x for x in cache.gpu_kv() if x != prot_kv],
                        key=lambda x: self.km.get(x, -1)):
            if freed >= needed: break
            ev.append(('demote_kv', c)); freed += cache.kv_size(c)
        return ev


class JointTiered(BasePolicy):
    """Dependency- and tier-aware: demote KV (cheap to restore over PCIe),
    keep adapters on GPU (expensive to restore if they cascade to disk)."""
    name = "joint"

    def choose_evictions(self, cache, needed, prot_a, prot_kv):
        ev, freed = [], 0
        stale = [c for c in cache.gpu_kv()
                 if c != prot_kv and cache.adapter_tier(cache.kv_owner(c)) != 'gpu']
        for c in sorted(stale, key=lambda x: self.km.get(x, -1)):
            if freed >= needed: return ev
            ev.append(('demote_kv', c)); freed += cache.kv_size(c)
        chosen = set(i for _, i in ev)
        for c in sorted([x for x in cache.gpu_kv() if x != prot_kv and x not in chosen],
                        key=lambda x: self.km.get(x, -1)):
            if freed >= needed: return ev
            ev.append(('demote_kv', c)); freed += cache.kv_size(c); chosen.add(c)
        orphans = {}
        for c in cache.gpu_kv():
            if c in chosen: continue
            o = cache.kv_owner(c); orphans[o] = orphans.get(o, 0) + 1
        for a in sorted([x for x in cache.gpu_adapters() if x != prot_a],
                        key=lambda x: (orphans.get(x, 0), self.am.get(x, -1))):
            if freed >= needed: break
            ev.append(('demote_adapter', a)); freed += ADAPTER_MB
        return ev


class TieredSimulator:
    def __init__(self, requests, conversations, cache, policy, max_batch=16):
        self.cache = cache; self.policy = policy; self.max_batch = max_batch
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}
        self.now = 0.0; self.pending = {}; self.finished = []
        self.from_disk = 0; self.from_cpu = 0
        self.kv_recomputes = 0; self.kv_from_cpu = 0; self.recompute_tokens = 0

    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            r = self.incoming.popleft()
            a = self.convos[r.conversation_id].adapter_id
            self.pending.setdefault(a, deque()).append(r)

    def pending_is_empty(self): return not any(self.pending.values())

    def cpu_make_room(self, mb):
        """CPU full -> cascade: drop the coldest CPU resident entirely."""
        while self.cache.cpu_free() < mb:
            cands = [(self.policy.km.get(c, -1), 'kv', c)
                     for c, e in self.cache.kv.items() if e['tier'] == 'cpu']
            cands += [(self.policy.am.get(a, -1), 'ad', a)
                      for a, t in self.cache.adapters.items() if t == 'cpu']
            if not cands: return
            cands.sort()
            _, kind, i = cands[0]
            self.cache.drop_kv(i) if kind == 'kv' else self.cache.drop_adapter(i)

    def apply_evictions(self, evictions):
        for kind, i in evictions:
            if kind == 'demote_adapter':
                self.cpu_make_room(ADAPTER_MB); self.cache.place_adapter(i, 'cpu')
            elif kind == 'demote_kv':
                self.cpu_make_room(self.cache.kv_size(i)); self.cache.kv[i]['tier'] = 'cpu'

    def make_room(self, mb, pa, pk):
        if self.cache.gpu_free() >= mb: return
        need = mb - self.cache.gpu_free()
        self.apply_evictions(self.policy.choose_evictions(self.cache, need, pa, pk))

    def ensure_adapter(self, a):
        t = self.cache.adapter_tier(a)
        if t == 'gpu': return 0.0
        self.make_room(ADAPTER_MB, a, None)
        if self.cache.gpu_free() < ADAPTER_MB: return 0.0
        self.cache.place_adapter(a, 'gpu')
        if t == 'cpu':
            self.from_cpu += 1; return SWAP_WARM_MS
        self.from_disk += 1; return SWAP_COLD_MS

    def ensure_kv(self, cid, pa):
        convo = self.convos[cid]
        if convo.kv_cache_size <= 0: return 0.0
        t = self.cache.kv_tier(cid)
        if t == 'gpu': return 0.0
        mb = convo.kv_cache_size * MB_PER_TOKEN
        self.make_room(mb, pa, cid)
        if self.cache.gpu_free() < mb: return 0.0
        self.cache.place_kv(cid, mb, convo.adapter_id, 'gpu')
        if t == 'cpu':
            self.kv_from_cpu += 1
            return mb * PCIE_MS_PER_MB
        self.kv_recomputes += 1
        self.recompute_tokens += convo.kv_cache_size
        return convo.kv_cache_size * PREFILL_MS_PER_TOKEN

    def grow(self, r, pa):
        convo = self.convos[r.conversation_id]
        convo.grow_cache(r.output_tokens)
        if self.cache.kv_tier(r.conversation_id) == 'gpu':
            delta = r.output_tokens * MB_PER_TOKEN
            self.make_room(delta, pa, r.conversation_id)
            if self.cache.gpu_free() >= delta:
                self.cache.kv[r.conversation_id]['size_mb'] = \
                    convo.kv_cache_size * MB_PER_TOKEN

    def run(self):
        while self.incoming or not self.pending_is_empty():
            self.admit_arrivals()
            if self.pending_is_empty():
                if not self.incoming: break
                self.now = self.incoming[0].arrival_time; continue
            a = self.policy.choose_next(self.pending, self.cache, self.now)
            cost = self.ensure_adapter(a)
            q = self.pending[a]
            batch = [q.popleft() for _ in range(min(self.max_batch, len(q)))]
            for r in batch: cost += self.ensure_kv(r.conversation_id, a)
            mx = max(r.output_tokens for r in batch)
            cost += 25 + len(batch) * 2 + mx * DECODE_MS_PER_TOKEN
            self.now += cost
            for r in batch:
                r.finish(self.now - r.arrival_time); self.finished.append(r)
                self.grow(r, a); self.policy.record(a, r.conversation_id, self.now)
        return self.finished


CPU_TIERS = [("generous", 4000), ("moderate", 1000), ("tight", 400), ("minimal", 100)]
SEEDS = 5


def trial(gpu_mb, cpu_mb, PolicyCls, seed):
    convos, reqs = make_multi_turn_workload(n_conversations=30, n_adapters=12,
                                            turns_per_convo=10, mean_tokens=100,
                                            rate=0.0002, turn_gap=20000, seed=seed)
    sim = TieredSimulator(reqs, convos, TieredCache(gpu_mb, cpu_mb), PolicyCls())
    done = sim.run()
    return percentile([r.latency() for r in done], 50), sim.from_disk


def main():
    print(__doc__)
    GPU = 900
    print(f"GPU pool fixed at {GPU}MB; varying the CPU tier below it "
          f"(mean of {SEEDS} seeds)\n")
    print(f"{'CPU tier':>12} {'MB':>6} {'separate p50':>13} {'joint p50':>11}"
          f" {'gap':>7} {'adapters from disk':>20}")
    print("-" * 74)
    for label, cpu in CPU_TIERS:
        vals = {}
        for P in [SeparateTiered, JointTiered]:
            runs = [trial(GPU, cpu, P, s) for s in range(SEEDS)]
            vals[P.name] = (sum(v[0] for v in runs) / SEEDS,
                            sum(v[1] for v in runs) / SEEDS)
        gap = 100 * (vals['separate'][0] - vals['joint'][0]) / vals['separate'][0]
        print(f"{label:>12} {cpu:>6} {vals['separate'][0]:>13.0f}"
              f" {vals['joint'][0]:>11.0f} {gap:>6.1f}%"
              f" {vals['separate'][1]:>9.0f} -> {vals['joint'][1]:<7.0f}")
    print("\nPhase 2 (no CPU tier at all) measured a ~27% gap. A generous CPU tier")
    print("shrinks it to a few percent: the hierarchy does the policy's job.")


if __name__ == "__main__":
    main()
