"""PHASE 5 -- Continuous batching. The real execution model.

WORLD: phases 2-4 ran BATCH-AT-A-TIME: pick one adapter, drain its whole queue,
       move on. Real systems -- and ELORA/FastLibra, which we are comparing
       against -- run CONTINUOUS batching: one long-lived batch that the GPU
       steps forward ONE TOKEN at a time. Between steps the scheduler removes
       finished sequences (freeing their KV instantly) and splices in waiting
       requests (prefilling their prompt, chunked, in the same step).

       Two architectures, selectable with --pool:
         separate  -- adapter slab + KV pool, fixed independent sizes (vLLM)
         unified   -- one MB-denominated pool for both (S-LoRA / ELORA-style)

       KV preemption when GPU memory is exhausted mid-decode:
         recompute (default)  -- drop the blocks; rebuild by re-prefilling the
                                 whole sequence-so-far when it is rescheduled
         + optional CPU swap tier (--kv-cpu MB) -- demote to CPU over PCIe,
           copy back later. Mirrors phase 2 Part B.

WHY THIS PHASE EXISTS (see ROADMAP.md):
  1. Comparability: ELORA's numbers are all continuous-batched.
  2. The pathology this project is about -- adapter evicted while its KV is
     still resident -- is a TIMING phenomenon. It needs a state that
     batch-at-a-time never produces: an adapter that is idle RIGHT NOW but was
     recently resident and has a conversation about to send its next turn.
  3. TPOT. ELORA's win is 45.7% TTFT *and* 37.8% TPOT. The TPOT half comes
     entirely from recompute-prefill stealing step time from unrelated
     decoders in a shared batch. No shared batch -> no TPOT channel.

NEW CONSTRAINT -- adapter pinning:
  An adapter cannot be evicted while ANY of its sequences is in the running
  batch. --max-loras is now what it is in vLLM: the max number of DISTINCT
  adapters with live sequences in the batch at once. A waiting request whose
  adapter would be the (max_loras+1)-th distinct adapter is not admitted until
  one drains.

FINDING (mean over 5 seeds, KV pool 800MB, adapter 20MB):

  SEPARATE pool, shrinking the adapter slab (max_loras 12 -> 3):
    max_loras   p50    p95   TTFT p50   TPOT   stale KV   disk loads
       12       689   1076      22       17.0     0.0%        11
        7       747   1531      24       19.6    13.6%        28
        5       871   1957      29       23.1    27.5%        54
        3      1170   2676     296       26.6    49.9%        88

  The pathology reproduces on the realistic substrate -- 50% stale KV at
  max_loras=3, bracketing ELORA's 42.4% (vLLM) / 48.6% (ELORA-WOM). But the
  MECHANISM is now visible and it is a TIMING one: the sharp TTFT knee at
  max_loras=3 (22ms -> 296ms) is ADMISSION STALL -- a waiting request whose
  adapter would be the 4th distinct adapter in the batch cannot be spliced in
  until one of the 3 pinned adapters drains. Phases 2-4, being batch-at-a-time,
  produced the staleness NUMBER without this mechanism.

  TPOT rises 17 -> 27 as the slab shrinks: recompute-prefill of rebuilt stale
  KV competes for the per-step token budget with every decoder in the batch.
  This is the TPOT channel that did not exist before -- and it is exactly
  ELORA's stated reason for their 37.8% TPOT win.

  The cross-pool signal (phase 3) still fails on latency here. cost-aware
  moves stale KV 27.5% -> 23.4% (or, under KV pressure, 17.3% -> 8.4%) but p50
  moves by <1% -- noise. Slab capacity is still fixed and independent; the
  signal changes WHICH adapter reloads, not how many.

  UNIFIED pool (one 1000MB region), adapter 20MB: p50 stays 689 -> 706 across
  the whole sweep, stale KV stays 0%. A big idle conversation's KV is traded
  for an adapter slot mid-stream. Phase 4's headline, holding under continuous
  batching. At 20MB adapters blind / kv-first / cost-aware are identical --
  the phase-4 "ordering is enough" result.

  NEW: that equivalence is partly an artifact of the 20MB adapter size. At
  --adapter-mb 80 the unified pool shows kv-first clearly beating both blind
  and cost-aware (p50 ~700 vs 760-840, stale 0% vs 11-21%): once an adapter is
  no longer a pebble, "evict KV before adapters" stops being free and the
  ordering rule earns its keep. So the phase-4 claim should read: ordering
  discipline is what matters, and it matters MORE as the adapter/KV size gap
  narrows. Whether dependency scoring beats plain ordering still needs prefix
  sharing to test -- that is phase 6.

  With a CPU KV tier (--kv-cpu): recompute events drop ~3x (blocks demoted,
  not dropped) while staleness is unchanged -- phase 2b's "cheaper not rarer",
  reproduced.

CALIBRATION NOTE: core.ADAPTER_MB = 20 is on the low side for a rank-16 LoRA on
  an 8B model targeting all 7 projections (~60-90MB); 20MB is closer to
  rank-8 attention-only. Adapter-slab pressure is this phase's x-axis, so
  --adapter-mb lets you sweep 20/40/80 and check the pathology shape is
  size-robust. core.py is left untouched so phases 1-4 stay on the record.

Run:  python phase5_continuous_batching.py
      python phase5_continuous_batching.py --pool unified --adapter-mb 40
      python phase5_continuous_batching.py --pool separate --kv-gpu 250 --kv-cpu 400
"""

import argparse
import math
from collections import deque, OrderedDict

from core import (MB_PER_TOKEN, PREFILL_MS_PER_TOKEN, DECODE_MS_PER_TOKEN,
                  SWAP_COLD_MS, SWAP_WARM_MS, PCIE_MS_PER_MB,
                  make_multi_turn_workload, percentile)

# --- step-loop constants ---
STEP_TOKEN_BUDGET = 512      # vLLM-style: decode tokens + a prefill chunk per step
FIXED_STEP_MS = 4.0          # kernel launch / sampling overhead per step
PER_SEQ_STEP_MS = 0.05       # marginal cost of one more row in the batch
MAX_BATCH_SEQS = 192         # hard cap on concurrent sequences (KV permitting)
WAIT_THRESHOLD_MS = 30000    # starvation guard, same value as phases 2-4


# ==========================================================================
# Sequence state
# ==========================================================================

class SeqState:
    """One turn, in flight. Lives in the batch from admission to completion."""

    __slots__ = ("req", "cid", "adapter_id", "prompt_tokens", "output_tokens",
                 "prefill_left", "decode_left", "phase", "admitted_at")

    def __init__(self, req, cid, adapter_id, prompt_tokens, now):
        self.req = req
        self.cid = cid
        self.adapter_id = adapter_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = req.output_tokens
        self.prefill_left = prompt_tokens       # tokens still to prefill (chunked)
        self.decode_left = req.output_tokens    # tokens still to generate
        self.phase = "prefill" if prompt_tokens > 0 else "decode"
        self.admitted_at = now

    def done(self):
        return self.prefill_left <= 0 and self.decode_left <= 0


# ==========================================================================
# Caches. Separate = two fixed regions (phase 2). Unified = one pool (phase 4).
# Both expose the same interface the sim below uses.
# ==========================================================================

class SeparateCache:
    """vLLM: adapter slab (MB) + KV pool (MB), independent. Optional CPU tiers."""

    def __init__(self, adapter_mb, kv_gpu_mb, adapter_cpu_mb=0, kv_cpu_mb=0,
                 adapter_size=20.0):
        self.adapter_mb = adapter_mb
        self.kv_gpu_mb = kv_gpu_mb
        self.adapter_cpu_mb = adapter_cpu_mb
        self.kv_cpu_mb = kv_cpu_mb
        self.adapter_size = adapter_size
        self.a_tier = {}          # adapter_id -> 'gpu' | 'cpu'
        self.kv = {}              # cid -> {'tier','size_mb','adapter_id'}

    unified = False

    # ---- adapters ----
    def a_gpu_used(self):
        return sum(self.adapter_size for t in self.a_tier.values() if t == "gpu")

    def a_gpu_free(self):
        return self.adapter_mb - self.a_gpu_used()

    def a_cpu_used(self):
        return sum(self.adapter_size for t in self.a_tier.values() if t == "cpu")

    def a_cpu_free(self):
        return self.adapter_cpu_mb - self.a_cpu_used()

    def adapter_on_gpu(self, a):
        return self.a_tier.get(a) == "gpu"

    def adapter_where(self, a):
        return self.a_tier.get(a)

    def gpu_adapters(self):
        return [a for a, t in self.a_tier.items() if t == "gpu"]

    def place_adapter(self, a, tier):
        self.a_tier[a] = tier

    def drop_adapter(self, a):
        self.a_tier.pop(a, None)

    # ---- kv ----
    def kv_gpu_used(self):
        return sum(e["size_mb"] for e in self.kv.values() if e["tier"] == "gpu")

    def kv_gpu_free(self):
        return self.kv_gpu_mb - self.kv_gpu_used()

    def kv_cpu_used(self):
        return sum(e["size_mb"] for e in self.kv.values() if e["tier"] == "cpu")

    def kv_cpu_free(self):
        return self.kv_cpu_mb - self.kv_cpu_used()

    def kv_on_gpu(self, c):
        return c in self.kv and self.kv[c]["tier"] == "gpu"

    def kv_where(self, c):
        return self.kv[c]["tier"] if c in self.kv else None

    def kv_size(self, c):
        return self.kv[c]["size_mb"] if c in self.kv else 0.0

    def kv_owner(self, c):
        return self.kv[c]["adapter_id"] if c in self.kv else None

    def gpu_kv(self):
        return [c for c, e in self.kv.items() if e["tier"] == "gpu"]

    def place_kv(self, c, size_mb, adapter_id, tier):
        self.kv[c] = {"tier": tier, "size_mb": size_mb, "adapter_id": adapter_id}

    def set_kv_tier(self, c, tier):
        self.kv[c]["tier"] = tier

    def set_kv_size(self, c, size_mb):
        self.kv[c]["size_mb"] = size_mb

    def drop_kv(self, c):
        self.kv.pop(c, None)

    # room helpers used by the sim
    def adapter_fits_gpu(self, need):
        return self.a_gpu_free() >= need - 1e-9

    def kv_fits_gpu(self, need):
        return self.kv_gpu_free() >= need - 1e-9


class UnifiedCache:
    """One MB pool for adapters + KV. Optional single CPU tier for both."""

    unified = True

    def __init__(self, capacity_mb, cpu_mb=0, adapter_size=20.0):
        self.capacity_mb = capacity_mb
        self.cpu_mb = cpu_mb
        self.adapter_size = adapter_size
        self.a_tier = {}
        self.kv = {}

    def used_mb(self):
        return (sum(self.adapter_size for t in self.a_tier.values() if t == "gpu")
                + sum(e["size_mb"] for e in self.kv.values() if e["tier"] == "gpu"))

    def free_mb(self):
        return self.capacity_mb - self.used_mb()

    def cpu_used(self):
        return (sum(self.adapter_size for t in self.a_tier.values() if t == "cpu")
                + sum(e["size_mb"] for e in self.kv.values() if e["tier"] == "cpu"))

    def cpu_free(self):
        return self.cpu_mb - self.cpu_used()

    # adapters
    def adapter_on_gpu(self, a):
        return self.a_tier.get(a) == "gpu"

    def adapter_where(self, a):
        return self.a_tier.get(a)

    def gpu_adapters(self):
        return [a for a, t in self.a_tier.items() if t == "gpu"]

    def place_adapter(self, a, tier):
        self.a_tier[a] = tier

    def drop_adapter(self, a):
        self.a_tier.pop(a, None)

    # kv
    def kv_on_gpu(self, c):
        return c in self.kv and self.kv[c]["tier"] == "gpu"

    def kv_where(self, c):
        return self.kv[c]["tier"] if c in self.kv else None

    def kv_size(self, c):
        return self.kv[c]["size_mb"] if c in self.kv else 0.0

    def kv_owner(self, c):
        return self.kv[c]["adapter_id"] if c in self.kv else None

    def gpu_kv(self):
        return [c for c, e in self.kv.items() if e["tier"] == "gpu"]

    def place_kv(self, c, size_mb, adapter_id, tier):
        self.kv[c] = {"tier": tier, "size_mb": size_mb, "adapter_id": adapter_id}

    def set_kv_tier(self, c, tier):
        self.kv[c]["tier"] = tier

    def set_kv_size(self, c, size_mb):
        self.kv[c]["size_mb"] = size_mb

    def drop_kv(self, c):
        self.kv.pop(c, None)

    # room helpers -- for a unified pool "fits" is the same test for both types
    def adapter_fits_gpu(self, need):
        return self.free_mb() >= need - 1e-9

    def kv_fits_gpu(self, need):
        return self.free_mb() >= need - 1e-9


# ==========================================================================
# Eviction policies. Same philosophy / names as phases 2-4:
#   blind      : two independent LRUs, no cross-pool signal        (phase 2)
#   signal     : adapter side avoids evicting adapters with live KV (phase 3)
#   cost-aware : size-weighted + evict already-stale KV first       (phase 4)
#   kv-first   : unified only -- crude ordering, KV before adapters (phase 4 ablation)
#
# A policy exposes rank_adapters() and rank_kv(): GPU-resident victims in
# eviction order (first = evict first). `pinned` adapters are filtered out by
# the sim before the policy is asked, so policies never see them.
# ==========================================================================

class BlindLRU:
    name = "blind"

    def __init__(self):
        self.a_last = {}     # adapter_id -> last used ms
        self.k_last = {}     # cid -> last used ms

    def touch_adapter(self, a, now):
        self.a_last[a] = now

    def touch_kv(self, c, now):
        self.k_last[c] = now

    def rank_adapters(self, cache, candidates):
        return sorted(candidates, key=lambda a: self.a_last.get(a, -1))

    def rank_kv(self, cache, candidates):
        return sorted(candidates, key=lambda c: self.k_last.get(c, -1))


class SignalLRU(BlindLRU):
    """Phase 3: adapter side prefers to evict adapters with no GPU-resident KV."""
    name = "signal"

    def rank_adapters(self, cache, candidates):
        def live_dependents(a):
            return sum(1 for c in cache.gpu_kv() if cache.kv_owner(c) == a)
        return sorted(candidates, key=lambda a: (live_dependents(a),
                                                 self.a_last.get(a, -1)))


class CostAware(BlindLRU):
    """Phase 4: size-weighted adapter ranking + stale-KV-first on the KV side."""
    name = "cost-aware"

    def rank_adapters(self, cache, candidates):
        def stranded_mb(a):
            return sum(cache.kv_size(c) for c in cache.gpu_kv()
                       if cache.kv_owner(c) == a)
        return sorted(candidates, key=lambda a: (stranded_mb(a),
                                                 self.a_last.get(a, -1)))

    def rank_kv(self, cache, candidates):
        # already-stale KV (owner not on GPU) first -- free to drop -- then LRU
        return sorted(candidates,
                      key=lambda c: (cache.adapter_on_gpu(cache.kv_owner(c)),
                                     self.k_last.get(c, -1)))


class KVFirst(BlindLRU):
    """Phase 4 ablation (unified only): crude ordering, evict KV before adapters.
    Not a distinct rank_* -- the sim's make_room tries KV before adapters when
    the policy sets prefer_kv. Kept as a named policy so the sweep shows it."""
    name = "kv-first"
    prefer_kv = True


POLICIES = {p.name: p for p in [BlindLRU, SignalLRU, CostAware, KVFirst]}


# ==========================================================================
# The continuous-batching simulator
# ==========================================================================

class ContinuousBatchSim:
    def __init__(self, requests, conversations, cache, policy, max_loras,
                 preempt="recompute"):
        self.cache = cache
        self.policy = policy
        self.max_loras = max_loras
        self.preempt = preempt                     # 'recompute' only affects KV path
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}
        for r in self.incoming:
            if r.conversation_id not in self.convos:
                raise ValueError("request references unknown conversation")

        self.now = 0.0
        self.waiting = deque()                     # arrived, not yet admitted
        self.batch = []                            # list[SeqState]
        self.finished = []
        self.adapter_size = cache.adapter_size

        # counters
        self.adapter_from_disk = 0
        self.adapter_from_cpu = 0
        self.kv_recomputes = 0
        self.kv_from_cpu = 0
        self.recompute_tokens = 0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.steps = 0
        self.stale_samples = []
        self.ttft = {}                             # rid -> ms to first output token
        self.tpot_num = 0.0                        # sum of per-token decode latencies
        self.tpot_den = 0

    # ---------------- arrivals ----------------
    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            self.waiting.append(self.incoming.popleft())

    # ---------------- staleness metric (same definition as phases 2-4) ----------
    def stale_pct(self):
        gpu = self.cache.gpu_kv()
        total = sum(self.cache.kv_size(c) for c in gpu)
        if total == 0:
            return 0.0
        stale = sum(self.cache.kv_size(c) for c in gpu
                    if not self.cache.adapter_on_gpu(self.cache.kv_owner(c)))
        return 100.0 * stale / total

    # ---------------- pinning ----------------
    def pinned_adapters(self):
        return {s.adapter_id for s in self.batch}

    def distinct_batch_adapters(self):
        return {s.adapter_id for s in self.batch}

    # ---------------- eviction: make room on the GPU ----------------
    def _evict_adapters_for(self, need_mb, protect_adapter):
        pinned = self.pinned_adapters()
        cands = [a for a in self.cache.gpu_adapters()
                 if a != protect_adapter and a not in pinned]
        for a in self.policy.rank_adapters(self.cache, cands):
            if self.cache.adapter_fits_gpu(need_mb):
                return
            self._demote_or_drop_adapter(a)

    def _demote_or_drop_adapter(self, a):
        c = self.cache
        if c.unified:
            if c.cpu_mb > 0 and c.cpu_free() >= self.adapter_size:
                c.place_adapter(a, "cpu")
            else:
                self._make_cpu_room_unified(self.adapter_size)
                if c.cpu_mb > 0 and c.cpu_free() >= self.adapter_size:
                    c.place_adapter(a, "cpu")
                else:
                    c.drop_adapter(a)
        else:
            if c.adapter_cpu_mb > 0:
                self._make_adapter_cpu_room(self.adapter_size)
                if c.a_cpu_free() >= self.adapter_size:
                    c.place_adapter(a, "cpu")
                else:
                    c.drop_adapter(a)
            else:
                c.drop_adapter(a)

    def _make_adapter_cpu_room(self, need):
        c = self.cache
        while c.a_cpu_free() < need:
            cpu = [a for a, t in c.a_tier.items() if t == "cpu"]
            if not cpu:
                return
            victim = min(cpu, key=lambda a: self.policy.a_last.get(a, -1))
            c.drop_adapter(victim)

    def _evict_kv_for(self, need_mb, protect_cid):
        c = self.cache
        pinned_cids = {s.cid for s in self.batch}
        cands = [x for x in c.gpu_kv() if x != protect_cid and x not in pinned_cids]
        for cid in self.policy.rank_kv(c, cands):
            if c.kv_fits_gpu(need_mb):
                return
            size = c.kv_size(cid)
            if c.unified:
                if c.cpu_mb > 0:
                    self._make_cpu_room_unified(size)
                    if c.cpu_free() >= size:
                        c.set_kv_tier(cid, "cpu")
                        continue
                c.drop_kv(cid)
            else:
                if c.kv_cpu_mb > 0:
                    self._make_kv_cpu_room(size)
                    if c.kv_cpu_free() >= size:
                        c.set_kv_tier(cid, "cpu")
                        continue
                c.drop_kv(cid)

    def _make_kv_cpu_room(self, need):
        c = self.cache
        while c.kv_cpu_free() < need:
            cpu = [x for x, e in c.kv.items() if e["tier"] == "cpu"]
            if not cpu:
                return
            victim = min(cpu, key=lambda x: self.policy.k_last.get(x, -1))
            c.drop_kv(victim)

    def _make_cpu_room_unified(self, need):
        c = self.cache
        while c.cpu_free() < need:
            a_cpu = [a for a, t in c.a_tier.items() if t == "cpu"]
            k_cpu = [x for x, e in c.kv.items() if e["tier"] == "cpu"]
            if not a_cpu and not k_cpu:
                return
            # drop whichever CPU item is least-recently used
            best = None
            best_t = math.inf
            for a in a_cpu:
                t = self.policy.a_last.get(a, -1)
                if t < best_t:
                    best, best_t, best_kind = a, t, "a"
            for x in k_cpu:
                t = self.policy.k_last.get(x, -1)
                if t < best_t:
                    best, best_t, best_kind = x, t, "k"
            if best_kind == "a":
                c.drop_adapter(best)
            else:
                c.drop_kv(best)

    def make_room_for_adapter(self, protect_adapter):
        need = self.adapter_size
        if self.cache.adapter_fits_gpu(need):
            return
        prefer_kv = getattr(self.policy, "prefer_kv", False) and self.cache.unified
        if prefer_kv:
            self._evict_kv_for(need, protect_cid=None)
            if self.cache.adapter_fits_gpu(need):
                return
        self._evict_adapters_for(need, protect_adapter)
        if not self.cache.adapter_fits_gpu(need) and self.cache.unified:
            # last resort in a unified pool: take it out of KV
            self._evict_kv_for(need, protect_cid=None)

    def make_room_for_kv(self, need_mb, protect_cid):
        if self.cache.kv_fits_gpu(need_mb):
            return
        self._evict_kv_for(need_mb, protect_cid)
        if not self.cache.kv_fits_gpu(need_mb) and self.cache.unified:
            self._evict_adapters_for(need_mb, protect_adapter=None)

    # ---------------- ensure resident ----------------
    def ensure_adapter_resident(self, a):
        """Returns ms of swap cost added to THIS step."""
        where = self.cache.adapter_where(a)
        if where == "gpu":
            return 0.0
        self.make_room_for_adapter(protect_adapter=a)
        if not self.cache.adapter_fits_gpu(self.adapter_size):
            return 0.0                              # could not fit; caller will retry
        self.cache.place_adapter(a, "gpu")
        if where == "cpu":
            self.adapter_from_cpu += 1
            return SWAP_WARM_MS
        self.adapter_from_disk += 1
        return SWAP_COLD_MS

    def prompt_mb(self, cid):
        return self.convos[cid].kv_cache_size * MB_PER_TOKEN

    def ensure_kv_resident(self, cid, adapter_id):
        """Bring a conversation's existing KV back to GPU. Returns (ms, recompute_tokens).
        recompute path re-prefills the whole history (counted as prefill work in
        the step budget, so it competes with decoders -- the TPOT channel)."""
        convo = self.convos[cid]
        if convo.kv_cache_size <= 0:
            return 0.0, 0
        where = self.cache.kv_where(cid)
        if where == "gpu":
            return 0.0, 0
        need = convo.kv_cache_size * MB_PER_TOKEN
        self.make_room_for_kv(need, protect_cid=cid)
        if not self.cache.kv_fits_gpu(need):
            # cannot hold history at all -> recompute and keep going without it cached
            self.kv_recomputes += 1
            self.recompute_tokens += convo.kv_cache_size
            return 0.0, convo.kv_cache_size
        if where == "cpu":
            self.cache.set_kv_tier(cid, "gpu")
            self.kv_from_cpu += 1
            return need * PCIE_MS_PER_MB, 0
        # not resident anywhere -> recompute, then cache the rebuilt result
        self.cache.place_kv(cid, need, adapter_id, "gpu")
        self.kv_recomputes += 1
        self.recompute_tokens += convo.kv_cache_size
        return 0.0, convo.kv_cache_size

    # ---------------- scheduling: splice waiting requests into the batch ------
    def schedule(self):
        if not self.waiting:
            return
        # process oldest-first; a starving request jumps the queue
        self.waiting = deque(sorted(self.waiting, key=lambda r: r.arrival_time))
        deferred = deque()
        swap_ms_this_step = 0.0
        prefill_budget = STEP_TOKEN_BUDGET - sum(
            1 for s in self.batch if s.phase == "decode")

        while self.waiting:
            if len(self.batch) >= MAX_BATCH_SEQS:
                break
            r = self.waiting[0]
            starving = (self.now - r.arrival_time) > WAIT_THRESHOLD_MS
            a = self.convos[r.conversation_id].adapter_id

            # --max-loras: distinct adapters with live sequences
            batch_adapters = self.distinct_batch_adapters()
            if a not in batch_adapters and len(batch_adapters) >= self.max_loras:
                if not starving:
                    deferred.append(self.waiting.popleft())
                    continue
                # starving: allow, but this is the pathology surfacing

            self.waiting.popleft()

            swap_ms_this_step += self.ensure_adapter_resident(a)
            if not self.cache.adapter_on_gpu(a):
                deferred.append(r)                  # slab full of pinned adapters
                continue

            convo = self.convos[r.conversation_id]
            prompt_tokens = convo.kv_cache_size     # history so far = prompt for this turn
            kv_ms, recompute = self.ensure_kv_resident(r.conversation_id, a)
            swap_ms_this_step += kv_ms

            # tokens this turn must (re)prefill: recompute history if not cached,
            # else 0 for the already-cached history; the NEW output tokens are decode.
            to_prefill = recompute
            seq = SeqState(r, r.conversation_id, a, to_prefill, self.now)
            seq.decode_left = r.output_tokens
            # if history WAS cached, prefill_left is 0 and we go straight to decode
            if to_prefill <= 0:
                seq.phase = "decode"
            self.batch.append(seq)
            self.policy.touch_adapter(a, self.now)
            self.policy.touch_kv(r.conversation_id, self.now)

            if not self.cache.kv_on_gpu(r.conversation_id) and self.cache.kv_fits_gpu(
                    max(prompt_tokens, 1) * MB_PER_TOKEN):
                self.cache.place_kv(r.conversation_id,
                                    max(prompt_tokens, 0) * MB_PER_TOKEN, a, "gpu")

        self.waiting.extendleft(reversed(deferred))
        return swap_ms_this_step

    # ---------------- one GPU step ----------------
    def step(self):
        swap_ms = self.schedule() or 0.0
        if not self.batch:
            # nothing running; jump clock to next arrival
            if self.incoming:
                self.now = max(self.now, self.incoming[0].arrival_time)
            return

        n_decode = sum(1 for s in self.batch if s.phase == "decode")
        # prefill work admitted this step: each prefilling seq contributes a chunk
        prefill_room = STEP_TOKEN_BUDGET - n_decode
        prefill_this_step = 0
        for s in self.batch:
            if s.phase == "prefill" and prefill_room > 0:
                chunk = min(s.prefill_left, prefill_room)
                s.prefill_left -= chunk
                prefill_this_step += chunk
                prefill_room -= chunk
                if s.prefill_left <= 0:
                    s.phase = "decode"             # first output token this step

        # step duration: one shared decode pass + the prefill chunk riding along
        step_ms = (FIXED_STEP_MS
                   + PER_SEQ_STEP_MS * len(self.batch)
                   + DECODE_MS_PER_TOKEN                     # the one weight-read pass
                   + prefill_this_step * PREFILL_MS_PER_TOKEN
                   + swap_ms)
        self.now += step_ms
        self.steps += 1
        self.prefill_tokens += prefill_this_step

        # advance decoders, retire finished sequences
        still = []
        for s in self.batch:
            if s.phase == "decode":
                if s.req.request_id not in self.ttft:
                    self.ttft[s.req.request_id] = self.now - s.req.arrival_time
                s.decode_left -= 1
                self.decode_tokens += 1
                self.tpot_num += step_ms
                self.tpot_den += 1
                # grow this conversation's KV by one token
                convo = self.convos[s.cid]
                convo.grow_cache(1)
                if self.cache.kv_on_gpu(s.cid):
                    self.cache.set_kv_size(s.cid, convo.kv_cache_size * MB_PER_TOKEN)
            if s.done():
                s.req.finish(self.now - s.req.arrival_time)
                self.finished.append(s.req)
            else:
                still.append(s)
        self.batch = still

    # ---------------- run ----------------
    def run(self):
        guard = 0
        while self.incoming or self.waiting or self.batch:
            self.admit_arrivals()
            if not self.waiting and not self.batch:
                if not self.incoming:
                    break
                self.now = self.incoming[0].arrival_time
                continue
            self.stale_samples.append(self.stale_pct())
            self.step()
            guard += 1
            if guard > 5_000_000:
                raise RuntimeError("step loop did not terminate")
        return self.finished


# ==========================================================================
# Harness
# ==========================================================================

def workload(seed):
    return make_multi_turn_workload(n_conversations=40, n_adapters=12,
                                    turns_per_convo=5, mean_tokens=40,
                                    rate=0.0006, turn_gap=6000, seed=seed)


def build_cache(pool, adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu, cap, cpu,
                adapter_size):
    if pool == "separate":
        return SeparateCache(adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu,
                             adapter_size=adapter_size)
    return UnifiedCache(cap, cpu, adapter_size=adapter_size)


def trial(pool, max_loras, policy_name, seed, adapter_size,
          kv_gpu_mb=800, adapter_cpu=0, kv_cpu=0, unified_cap=1000, unified_cpu=0):
    convos, reqs = workload(seed)
    adapter_mb = max_loras * adapter_size
    cache = build_cache(pool, adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu,
                        unified_cap, unified_cpu, adapter_size)
    policy = POLICIES[policy_name]()
    sim = ContinuousBatchSim(reqs, convos, cache, policy, max_loras)
    sim.run()
    lats = [r.latency() for r in sim.finished]
    ttfts = list(sim.ttft.values())
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)
             if sim.stale_samples else 0.0)
    tpot = sim.tpot_num / sim.tpot_den if sim.tpot_den else float("nan")
    return {
        "n": len(sim.finished),
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "ttft_p50": percentile(ttfts, 50),
        "tpot": tpot,
        "stale": stale,
        "disk": sim.adapter_from_disk,
        "cpu_adapter": sim.adapter_from_cpu,
        "kv_recompute": sim.kv_recomputes,
        "recompute_tokens": sim.recompute_tokens,
        "steps": sim.steps,
    }


SEEDS = 5


def avg(**kw):
    rows = [trial(seed=s, **kw) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rows) / SEEDS for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", choices=["separate", "unified", "both"], default="both")
    ap.add_argument("--adapter-mb", type=float, default=20.0,
                    help="MB per adapter. core default 20 is low for rank-16/8B; "
                         "try 40 or 80 to check the pathology is size-robust.")
    ap.add_argument("--policy", default=None,
                    help="blind|signal|cost-aware|kv-first ; default sweeps a set")
    ap.add_argument("--kv-gpu", type=float, default=800.0)
    ap.add_argument("--kv-cpu", type=float, default=0.0)
    ap.add_argument("--adapter-cpu", type=float, default=0.0)
    ap.add_argument("--unified-cap", type=float, default=1000.0)
    ap.add_argument("--unified-cpu", type=float, default=0.0)
    ap.add_argument("--max-loras", type=int, nargs="+", default=[12, 7, 5, 3])
    args = ap.parse_args()

    print(__doc__)
    print(f"continuous batching | adapter={args.adapter_mb:.0f}MB "
          f"| KV GPU {args.kv_gpu:.0f}MB | mean of {SEEDS} seeds\n")

    pools = ["separate", "unified"] if args.pool == "both" else [args.pool]
    for pool in pools:
        if pool == "separate":
            policies = [args.policy] if args.policy else ["blind", "signal", "cost-aware"]
            cap_note = f"adapter slab = max_loras x {args.adapter_mb:.0f}MB, KV pool {args.kv_gpu:.0f}MB"
        else:
            policies = [args.policy] if args.policy else ["blind", "kv-first", "cost-aware"]
            cap_note = f"one pool {args.unified_cap:.0f}MB"
        print(f"== {pool.upper()} POOL == ({cap_note})")
        print(f"{'max_loras':>9} {'policy':>11} {'p50':>8} {'p95':>9} {'TTFT p50':>9}"
              f" {'TPOT':>7} {'stale KV':>9} {'disk':>6} {'kv recmp':>9}")
        print("-" * 88)
        for ml in args.max_loras:
            for pol in policies:
                m = avg(pool=pool, max_loras=ml, policy_name=pol,
                        adapter_size=args.adapter_mb, kv_gpu_mb=args.kv_gpu,
                        adapter_cpu=args.adapter_cpu, kv_cpu=args.kv_cpu,
                        unified_cap=args.unified_cap, unified_cpu=args.unified_cpu)
                print(f"{ml:>9} {pol:>11} {m['p50']:>8.0f} {m['p95']:>9.0f}"
                      f" {m['ttft_p50']:>9.0f} {m['tpot']:>7.2f} {m['stale']:>8.1f}%"
                      f" {m['disk']:>6.0f} {m['kv_recompute']:>9.0f}")
            print()
    print("Expected shape: as max_loras shrinks, admission stalls when a "
          "(max_loras+1)-th adapter wants in; TTFT + TPOT rise; stale KV climbs\n"
          "during think-time gaps. Unified pool should let a big idle "
          "conversation's KV be traded for an adapter slot -> flatter curve.")


if __name__ == "__main__":
    main()
