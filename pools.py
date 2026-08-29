"""Separate-pool machinery: adapters and KV caches in GENUINELY separate
memory regions, each with its own capacity and its own LRU.

This mirrors real vLLM: LoRA adapter weights live in a pre-allocated slab
(--max-loras / --max-cpu-loras, managed by LoRALRUCache) while the KV cache
lives in its own BlockPool region. The two never share bytes and, by default,
never consult each other. That blindness -- not competition for shared memory
-- is what produces stale KV in production.

Policies differ ONLY in eviction. Scheduling is identical everywhere so the
comparison isolates the eviction decision.
"""

from collections import deque

from core import (ADAPTER_MB, MB_PER_TOKEN, PREFILL_MS_PER_TOKEN,
                        DECODE_MS_PER_TOKEN, SWAP_COLD_MS, SWAP_WARM_MS,
                        PCIE_MS_PER_MB, percentile)

FIXED_BATCH_MS = 25
PER_REQUEST_MS = 2
MAX_BATCH = 16
WAIT_THRESHOLD_MS = 30000


class AdapterPool:
    """GPU slab -> optional CPU slab -> disk. Capacity in MB."""

    def __init__(self, gpu_mb, cpu_mb=0):
        self.gpu_mb = gpu_mb
        self.cpu_mb = cpu_mb
        self.tier = {}          # adapter_id -> 'gpu' | 'cpu'  (absent = disk)

    def gpu_used(self):
        return sum(ADAPTER_MB for t in self.tier.values() if t == 'gpu')

    def cpu_used(self):
        return sum(ADAPTER_MB for t in self.tier.values() if t == 'cpu')

    def gpu_free(self):
        return self.gpu_mb - self.gpu_used()

    def cpu_free(self):
        return self.cpu_mb - self.cpu_used()

    def where(self, a):
        return self.tier.get(a)

    def is_gpu(self, a):
        return self.tier.get(a) == 'gpu'

    def gpu_residents(self):
        return [a for a, t in self.tier.items() if t == 'gpu']

    def cpu_residents(self):
        return [a for a, t in self.tier.items() if t == 'cpu']

    def place(self, a, tier):
        self.tier[a] = tier

    def drop(self, a):
        self.tier.pop(a, None)


class KVPool:
    """GPU region -> optional CPU region -> gone (must recompute)."""

    def __init__(self, gpu_mb, cpu_mb=0):
        self.gpu_mb = gpu_mb
        self.cpu_mb = cpu_mb
        self.entry = {}         # cid -> {'tier','size_mb','adapter_id'}

    def gpu_used(self):
        return sum(e['size_mb'] for e in self.entry.values() if e['tier'] == 'gpu')

    def cpu_used(self):
        return sum(e['size_mb'] for e in self.entry.values() if e['tier'] == 'cpu')

    def gpu_free(self):
        return self.gpu_mb - self.gpu_used()

    def cpu_free(self):
        return self.cpu_mb - self.cpu_used()

    def where(self, c):
        return self.entry[c]['tier'] if c in self.entry else None

    def is_gpu(self, c):
        return c in self.entry and self.entry[c]['tier'] == 'gpu'

    def size(self, c):
        return self.entry[c]['size_mb'] if c in self.entry else 0

    def owner(self, c):
        return self.entry[c]['adapter_id'] if c in self.entry else None

    def gpu_residents(self):
        return [c for c, e in self.entry.items() if e['tier'] == 'gpu']

    def cpu_residents(self):
        return [c for c, e in self.entry.items() if e['tier'] == 'cpu']

    def place(self, c, size_mb, adapter_id, tier):
        self.entry[c] = {'tier': tier, 'size_mb': size_mb, 'adapter_id': adapter_id}

    def drop(self, c):
        self.entry.pop(c, None)

    # --- the one cross-pool signal used from phase 3 onward ---
    def live_dependents(self, adapter_id):
        """How many GPU-resident KV caches belong to this adapter."""
        return sum(1 for e in self.entry.values()
                   if e['tier'] == 'gpu' and e['adapter_id'] == adapter_id)

    def dependent_mb(self, adapter_id):
        """How many MB of GPU-resident KV would be stranded by evicting it."""
        return sum(e['size_mb'] for e in self.entry.values()
                   if e['tier'] == 'gpu' and e['adapter_id'] == adapter_id)


# --------------------------------------------------------------------------
# Policies. All share choose_next; they differ only in eviction ordering.
# --------------------------------------------------------------------------

class BasePolicy:
    name = "base"
    communicates = False

    def __init__(self):
        self.am = {}        # adapter_id -> last used
        self.km = {}        # conversation_id -> last used

    def choose_next(self, pending, apool, now):
        waiting = [a for a, q in pending.items() if q]
        starved = max(waiting, key=lambda a: now - pending[a][0].arrival_time)
        if now - pending[starved][0].arrival_time > WAIT_THRESHOLD_MS:
            return starved
        resident = [a for a in waiting if apool.is_gpu(a)]
        return max(resident if resident else waiting, key=lambda a: len(pending[a]))

    def record(self, adapter_id, conversation_id, now):
        self.am[adapter_id] = now
        self.km[conversation_id] = now

    # ---- eviction: adapter side ----
    def rank_adapters(self, apool, kpool, protect):
        """Return GPU-resident adapters in eviction order (first = evict first)."""
        cands = [a for a in apool.gpu_residents() if a != protect]
        cands.sort(key=lambda a: self.am.get(a, -1))     # plain LRU
        return cands

    # ---- eviction: KV side ----
    def rank_kv(self, kpool, apool, protect):
        cands = [c for c in kpool.gpu_residents() if c != protect]
        cands.sort(key=lambda c: self.km.get(c, -1))     # plain LRU
        return cands


class NoComm(BasePolicy):
    """Phase 2a / 2b. Two blind LRUs. Neither pool can see the other."""
    name = "blind"
    communicates = False


class OneWayComm(BasePolicy):
    """Phase 3. ONE signal crosses the boundary: before evicting an adapter,
    the adapter LRU asks the KV pool how many live dependents it has.
    Prefers adapters with zero dependents; falls back to LRU within that.
    KV side is unchanged -- still blind."""
    name = "signal"
    communicates = True

    def rank_adapters(self, apool, kpool, protect):
        cands = [a for a in apool.gpu_residents() if a != protect]
        cands.sort(key=lambda a: (kpool.live_dependents(a), self.am.get(a, -1)))
        return cands


class CostAware(BasePolicy):
    """Phase 4. Two-way and size-weighted.
      adapter side: rank by MB of KV that would be stranded (not just count),
                    so one adapter holding a 400MB conversation outranks one
                    holding three 20MB ones.
      KV side:      evict already-stale KV first -- its adapter is gone, so it
                    is currently unusable and costs nothing to drop."""
    name = "cost-aware"
    communicates = True

    def rank_adapters(self, apool, kpool, protect):
        cands = [a for a in apool.gpu_residents() if a != protect]
        cands.sort(key=lambda a: (kpool.dependent_mb(a), self.am.get(a, -1)))
        return cands

    def rank_kv(self, kpool, apool, protect):
        cands = [c for c in kpool.gpu_residents() if c != protect]
        # stale first (owner not on GPU), then LRU
        cands.sort(key=lambda c: (apool.is_gpu(kpool.owner(c)), self.km.get(c, -1)))
        return cands


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class SeparatePoolSim:
    def __init__(self, requests, conversations, apool, kpool, policy):
        self.apool = apool
        self.kpool = kpool
        self.policy = policy
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}
        for r in self.incoming:
            if r.conversation_id not in self.convos:
                raise ValueError("request references unknown conversation")
        self.now = 0.0
        self.pending = {}
        self.finished = []
        # counters
        self.adapter_from_disk = 0
        self.adapter_from_cpu = 0
        self.kv_recomputes = 0
        self.kv_from_cpu = 0
        self.recompute_tokens = 0
        self.stale_samples = []

    # ---------- bookkeeping ----------
    def admit(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            r = self.incoming.popleft()
            a = self.convos[r.conversation_id].adapter_id
            self.pending.setdefault(a, deque()).append(r)

    def empty(self):
        return not any(self.pending.values())

    def stale_pct(self):
        gpu_kv = [c for c in self.kpool.gpu_residents()]
        total = sum(self.kpool.size(c) for c in gpu_kv)
        if total == 0:
            return 0.0
        stale = sum(self.kpool.size(c) for c in gpu_kv
                    if not self.apool.is_gpu(self.kpool.owner(c)))
        return 100.0 * stale / total

    # ---------- adapter pool ----------
    def adapter_cpu_make_room(self, mb):
        while self.apool.cpu_free() < mb:
            cpu = self.apool.cpu_residents()
            if not cpu:
                return
            victim = min(cpu, key=lambda a: self.policy.am.get(a, -1))
            self.apool.drop(victim)                      # cascades to disk

    def adapter_make_room(self, mb, protect):
        if self.apool.gpu_free() >= mb:
            return
        for a in self.policy.rank_adapters(self.apool, self.kpool, protect):
            if self.apool.gpu_free() >= mb:
                return
            if self.apool.cpu_mb > 0:
                self.adapter_cpu_make_room(ADAPTER_MB)
                if self.apool.cpu_free() >= ADAPTER_MB:
                    self.apool.place(a, 'cpu')           # demote
                else:
                    self.apool.drop(a)                   # straight to disk
            else:
                self.apool.drop(a)                       # no CPU tier at all

    def ensure_adapter(self, a):
        """Returns ms of cost."""
        where = self.apool.where(a)
        if where == 'gpu':
            return 0.0
        self.adapter_make_room(ADAPTER_MB, a)
        if self.apool.gpu_free() < ADAPTER_MB:
            return 0.0                                   # could not fit; serve anyway
        self.apool.place(a, 'gpu')
        if where == 'cpu':
            self.adapter_from_cpu += 1
            return SWAP_WARM_MS
        self.adapter_from_disk += 1
        return SWAP_COLD_MS

    # ---------- kv pool ----------
    def kv_cpu_make_room(self, mb):
        while self.kpool.cpu_free() < mb:
            cpu = self.kpool.cpu_residents()
            if not cpu:
                return
            victim = min(cpu, key=lambda c: self.policy.km.get(c, -1))
            self.kpool.drop(victim)                      # gone; must recompute later

    def kv_make_room(self, mb, protect):
        if self.kpool.gpu_free() >= mb:
            return
        for c in self.policy.rank_kv(self.kpool, self.apool, protect):
            if self.kpool.gpu_free() >= mb:
                return
            size = self.kpool.size(c)
            if self.kpool.cpu_mb > 0:
                self.kv_cpu_make_room(size)
                if self.kpool.cpu_free() >= size:
                    self.kpool.entry[c]['tier'] = 'cpu'  # demote
                else:
                    self.kpool.drop(c)
            else:
                self.kpool.drop(c)

    def ensure_kv(self, cid, adapter_id):
        """Returns ms of cost."""
        convo = self.convos[cid]
        if convo.kv_cache_size <= 0:
            return 0.0
        where = self.kpool.where(cid)
        if where == 'gpu':
            return 0.0
        mb = convo.kv_cache_size * MB_PER_TOKEN
        self.kv_make_room(mb, cid)
        if self.kpool.gpu_free() < mb:
            # cannot hold it at all: recompute and discard
            self.kv_recomputes += 1
            self.recompute_tokens += convo.kv_cache_size
            return convo.kv_cache_size * PREFILL_MS_PER_TOKEN
        self.kpool.place(cid, mb, adapter_id, 'gpu')
        if where == 'cpu':
            self.kv_from_cpu += 1
            return mb * PCIE_MS_PER_MB
        self.kv_recomputes += 1
        self.recompute_tokens += convo.kv_cache_size
        return convo.kv_cache_size * PREFILL_MS_PER_TOKEN

    def grow(self, r, adapter_id):
        convo = self.convos[r.conversation_id]
        convo.grow_cache(r.output_tokens)
        if self.kpool.is_gpu(r.conversation_id):
            delta = r.output_tokens * MB_PER_TOKEN
            self.kv_make_room(delta, r.conversation_id)
            if self.kpool.gpu_free() >= delta:
                self.kpool.entry[r.conversation_id]['size_mb'] = \
                    convo.kv_cache_size * MB_PER_TOKEN

    # ---------- main loop ----------
    def run(self):
        while self.incoming or not self.empty():
            self.admit()
            if self.empty():
                if not self.incoming:
                    break
                self.now = self.incoming[0].arrival_time
                continue

            self.stale_samples.append(self.stale_pct())

            adapter = self.policy.choose_next(self.pending, self.apool, self.now)
            cost = self.ensure_adapter(adapter)

            q = self.pending[adapter]
            batch = [q.popleft() for _ in range(min(MAX_BATCH, len(q)))]
            for r in batch:
                cost += self.ensure_kv(r.conversation_id, adapter)

            mx = max(r.output_tokens for r in batch)
            cost += FIXED_BATCH_MS + len(batch) * PER_REQUEST_MS + mx * DECODE_MS_PER_TOKEN
            self.now += cost

            for r in batch:
                r.finish(self.now - r.arrival_time)
                self.finished.append(r)
                self.grow(r, adapter)
                self.policy.record(adapter, r.conversation_id, self.now)
        return self.finished


def summarize(sim):
    lats = [r.latency() for r in sim.finished]
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)) if sim.stale_samples else 0.0
    return {
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "stale": stale,
        "disk": sim.adapter_from_disk,
        "cpu_adapter": sim.adapter_from_cpu,
        "recompute_tokens": sim.recompute_tokens,
        "kv_from_cpu": sim.kv_from_cpu,
    }
