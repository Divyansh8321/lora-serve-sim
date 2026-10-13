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

WORKLOADS (--workload):
  chat (default, realistic): 20 adapters (ELORA's smallest setup), ~200 new
       prompt tokens and ~200 reply tokens per turn (LMSYS/ShareGPT scale),
       5 turns, 15 s think time. Total GPU memory 6000 MB, so KV is under
       real pressure (keeping every conversation would need ~19 GB of KV).
  toy: the original 12 adapters, 40-token replies, no prompts, 4000 MB.
       KV is tiny, so KV memory is never really contested.

FAIRNESS (both architectures, every run):
  * same total GPU MB (separate = slab + KV region of that total);
  * same host-RAM adapter cache holding every adapter, so after an adapter's
    first disk load, reloads are CPU->GPU copies. (A real operator serving
    N adapters sets vLLM --max-cpu-loras N; ELORA has 256 GB host RAM.
    vLLM's out-of-box default is max_loras -- see --adapter-cpu.)
  * slab% = the adapter slab as a share of GPU memory. ELORA's vLLM baseline
    uses 0.2, i.e. max_loras ~6-7 here.

FINDING (2026-10-11, mean of 5 seeds; mean latency / mean TTFT, ms):

  chat, --hw ours, 6000 MB
    max_loras slab%   SEPARATE (vLLM)          UNIFIED same cap     stale KV (sep)
       20     60%     4570 / 188               4280 / 156             0%
       10     30%     4216 / 175               4303 / 190             6%
        6     18%     4923 / 977               5149 / 1098           29%
        4     12%     7815 / 3950              8179 / 4223           46%
    UNIFIED with no batch cap: 4280 / 156.

  chat, --hw elora-aggressive: every cell 1360-1480 / 39-165, separate and
    unified within +-4% at every slab size.
  toy, --hw ours: separate 691 -> 763 as max_loras 12 -> 3; unified 691 -> 720.

  READING:
  * Separate vs unified at the SAME max_loras: within ~5% everywhere, and
    the separate pool is sometimes FASTER. A unified pool's flexibility is
    not worth much once both sides can reload adapters from host RAM.
  * Stale KV reaches 46-52% in the separate pool and costs nothing we can
    measure. It sits in the KV region's otherwise-unused space and is
    reclaimed by LRU when needed (vLLM does the same with cached blocks).
  * The latency cliff at small max_loras hits BOTH architectures equally.
    It is the BATCH CAP: vLLM's max_loras is also the max number of distinct
    adapters in one batch, so with 200-token replies a request for a 5th
    adapter waits seconds for a slot. Lift the cap (possible only when the
    pool is unified, since there is no fixed slab) and TTFT drops
    3950 -> 156 ms at the 12% slab. That is the only large unified-pool
    win here, and it disappears against a tuned vLLM (max_loras 10:
    4216 vs 4280). Caveat: running more distinct adapters per batch costs
    extra kernel time (Punica/SGMV), which this model does not charge, so
    the no-cap row is optimistic.
  * On faster hardware (elora-aggressive) sequences finish ~3x sooner, the
    batch cap barely binds, and the two architectures are indistinguishable.

AUDIT HISTORY:
  * 2026-10-11 (a): earlier numbers (+66% separate penalty, TTFT 296 ms) came
    from giving the separate pool 4000 MB of KV ON TOP of its slab and no CPU
    adapter cache (a config vLLM rejects). Decode growth now makes room, and
    Request.input_tokens adds user prompts.
  * 2026-10-11 (b): the unified pool had no CPU adapter cache while the
    separate pool did, so unified reloaded evicted adapters from disk
    (300 ms) and separate from host RAM (30 ms). Both now get the same
    host-RAM adapter cache. The chat workload is now the default.

Run:  python phase5_continuous_batching.py                    # chat workload
      python phase5_continuous_batching.py --hw elora-aggressive
      python phase5_continuous_batching.py --workload toy
"""

import argparse
import math
from collections import deque, OrderedDict
from dataclasses import replace as _replace

from core import (MB_PER_TOKEN, PREFILL_MS_PER_TOKEN, DECODE_MS_PER_TOKEN,
                  SWAP_COLD_MS, SWAP_WARM_MS, PCIE_MS_PER_MB, ADAPTER_MB,
                  HardwareProfile, OURS, get_profile,
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

    def __init__(self, capacity_mb, cpu_mb=0, adapter_size=20.0,
                 adapter_cpu_mb=0):
        self.capacity_mb = capacity_mb
        self.cpu_mb = cpu_mb
        # adapter-only CPU cache, the same thing vLLM's max_cpu_loras gives the
        # separate pool -- so both architectures reload evicted adapters from
        # host RAM, not disk. Kept apart from cpu_mb (which is shared with KV).
        self.adapter_cpu_mb = adapter_cpu_mb
        self.adapter_size = adapter_size
        self.a_tier = {}
        self.kv = {}

    def a_cpu_free(self):
        used = sum(self.adapter_size for t in self.a_tier.values() if t == "acpu")
        return self.adapter_cpu_mb - used

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
        t = self.a_tier.get(a)
        return "cpu" if t == "acpu" else t

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
                 preempt="recompute", hw=OURS):
        hw = get_profile(hw)
        self.hw = hw
        self.mb_per_token = hw.mb_per_token
        self.prefill_ms_per_token = hw.prefill_ms_per_token
        self.decode_ms_per_token = hw.decode_ms_per_token
        self.swap_cold_ms = hw.swap_cold_ms
        self.swap_warm_ms = hw.swap_warm_ms
        self.pcie_ms_per_mb = hw.pcie_ms_per_mb
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
        # if every resident KV belongs to a running sequence, growth cannot be
        # satisfied without preempting one (not modelled); record how far over.
        self.overcommit_peak_mb = 0.0
        self.kv_peak_mb = 0.0

    def _free_mb(self):
        c = self.cache
        return c.free_mb() if c.unified else c.kv_gpu_free()

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
        if c.unified and c.adapter_cpu_mb > 0:
            while c.a_cpu_free() < self.adapter_size:
                cpu = [x for x, t in c.a_tier.items() if t == "acpu"]
                if not cpu:
                    break
                c.drop_adapter(min(cpu, key=lambda x: self.policy.a_last.get(x, -1)))
            if c.a_cpu_free() >= self.adapter_size:
                c.place_adapter(a, "acpu")
            else:
                c.drop_adapter(a)
        elif c.unified:
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
            return self.swap_warm_ms
        self.adapter_from_disk += 1
        return self.swap_cold_ms

    def prompt_mb(self, cid):
        return self.convos[cid].kv_cache_size * self.mb_per_token

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
        need = convo.kv_cache_size * self.mb_per_token
        self.make_room_for_kv(need, protect_cid=cid)
        if not self.cache.kv_fits_gpu(need):
            # cannot hold history at all -> recompute and keep going without it cached
            self.kv_recomputes += 1
            self.recompute_tokens += convo.kv_cache_size
            return 0.0, convo.kv_cache_size
        if where == "cpu":
            self.cache.set_kv_tier(cid, "gpu")
            self.kv_from_cpu += 1
            return need * self.pcie_ms_per_mb, 0
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
            kv_ms, recompute = self.ensure_kv_resident(r.conversation_id, a)
            swap_ms_this_step += kv_ms

            # this turn's NEW user-prompt tokens: always prefilled, and they
            # extend the conversation's KV (room is made for them first).
            new_in = getattr(r, "input_tokens", 0)
            if new_in > 0:
                convo.grow_cache(new_in)
                if self.cache.kv_on_gpu(r.conversation_id):
                    self.make_room_for_kv(new_in * self.mb_per_token,
                                          protect_cid=r.conversation_id)
                    self.cache.set_kv_size(r.conversation_id,
                                           convo.kv_cache_size * self.mb_per_token)
            prompt_tokens = convo.kv_cache_size     # history + this turn's prompt

            # tokens this turn must (re)prefill: recompute history if not cached
            # (0 if it was), plus the new prompt; the output tokens are decode.
            to_prefill = recompute + new_in
            seq = SeqState(r, r.conversation_id, a, to_prefill, self.now)
            seq.decode_left = r.output_tokens
            # if history WAS cached, prefill_left is 0 and we go straight to decode
            if to_prefill <= 0:
                seq.phase = "decode"
            self.batch.append(seq)
            self.policy.touch_adapter(a, self.now)
            self.policy.touch_kv(r.conversation_id, self.now)

            if not self.cache.kv_on_gpu(r.conversation_id) and self.cache.kv_fits_gpu(
                    max(prompt_tokens, 1) * self.mb_per_token):
                self.cache.place_kv(r.conversation_id,
                                    max(prompt_tokens, 0) * self.mb_per_token, a, "gpu")

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
                   + self.decode_ms_per_token                # the one weight-read pass
                   + prefill_this_step * self.prefill_ms_per_token
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
                    # make room for the new token's KV (earlier versions grew
                    # without checking and silently overran the pool)
                    self.make_room_for_kv(self.mb_per_token, protect_cid=s.cid)
                    self.cache.set_kv_size(s.cid, convo.kv_cache_size * self.mb_per_token)
                    self.overcommit_peak_mb = max(self.overcommit_peak_mb,
                                                  -self._free_mb())
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
            self.kv_peak_mb = max(self.kv_peak_mb, sum(
                self.cache.kv_size(c) for c in self.cache.gpu_kv()))
            guard += 1
            if guard > 5_000_000:
                raise RuntimeError("step loop did not terminate")
        return self.finished


# ==========================================================================
# Harness
# ==========================================================================

# Two workloads. `toy` is the original (12 adapters, 40-token replies, no user
# prompts): every conversation's KV is tiny, so KV memory is never really
# under pressure. `chat` is the realistic default: ELORA's smallest setup
# (20 LoRAs), chat-sized turns (~200 new prompt tokens, ~200 reply tokens;
# LMSYS / ShareGPT scale), and a longer think-time between turns.
WORKLOADS = {
    "toy":  dict(n_conversations=40, n_adapters=12, turns_per_convo=5,
                 mean_tokens=40, rate=0.0006, turn_gap=6000,
                 mean_input_tokens=0),
    "chat": dict(n_conversations=80, n_adapters=20, turns_per_convo=5,
                 mean_tokens=200, rate=0.0006, turn_gap=15000,
                 mean_input_tokens=200),
}


def workload(seed, mean_input_tokens=None, kind="toy"):
    kw = dict(WORKLOADS[kind])
    if mean_input_tokens is not None:
        kw["mean_input_tokens"] = mean_input_tokens
    return make_multi_turn_workload(seed=seed, **kw)


def build_cache(pool, adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu, cap, cpu,
                adapter_size):
    if pool == "separate":
        return SeparateCache(adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu,
                             adapter_size=adapter_size)
    return UnifiedCache(cap, cpu, adapter_size=adapter_size,
                        adapter_cpu_mb=adapter_cpu)


def trial(pool, max_loras, policy_name, seed, adapter_size,
          kv_gpu_mb=None, adapter_cpu=None, kv_cpu=0, unified_cap=4000, unified_cpu=0,
          hw="ours", mean_input_tokens=None, kind="toy"):
    """Defaults make the two architectures comparable:
      kv_gpu_mb=None   -> unified_cap - adapter slab, so BOTH get the same
                          total GPU MB (earlier versions gave the separate
                          pool 4000 MB of KV ON TOP of the slab).
      adapter_cpu=None -> a host-RAM adapter cache holding EVERY adapter, for
                          BOTH pools: after an adapter's first (disk) load, a
                          reload is a CPU->GPU copy. That is what an operator
                          serving N adapters configures (vLLM:
                          --max-cpu-loras N; ELORA: 256 GB host memory).
                          vLLM's out-of-the-box default is max_cpu_loras =
                          max_loras; pass --adapter-cpu to model it."""
    convos, reqs = workload(seed, mean_input_tokens, kind)
    adapter_mb = max_loras * adapter_size
    if kv_gpu_mb is None:
        kv_gpu_mb = max(0.0, unified_cap - adapter_mb)
    if adapter_cpu is None:
        adapter_cpu = WORKLOADS[kind]["n_adapters"] * adapter_size
    cache = build_cache(pool, adapter_mb, kv_gpu_mb, adapter_cpu, kv_cpu,
                        unified_cap, unified_cpu, adapter_size)
    policy = POLICIES[policy_name]()
    sim = ContinuousBatchSim(reqs, convos, cache, policy, max_loras, hw=hw)
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
        "overcommit": sim.overcommit_peak_mb,
        "mean_lat": sum(lats) / len(lats),
        "ttft_mean": sum(ttfts) / len(ttfts),
        "load_s": (sim.adapter_from_disk * sim.swap_cold_ms
                   + sim.adapter_from_cpu * sim.swap_warm_ms) / 1000,
        "rebuild_s": sim.recompute_tokens * sim.prefill_ms_per_token / 1000,
        "kv_peak": sim.kv_peak_mb,
    }


SEEDS = 5


def avg(**kw):
    rows = [trial(seed=s, **kw) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rows) / SEEDS for k in rows[0]}


DEFAULTS = {   # per workload: total GPU MB, max_loras sweep
    "chat": dict(cap=6000.0, max_loras=[20, 10, 6, 4]),
    "toy":  dict(cap=4000.0, max_loras=[12, 7, 5, 3]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", choices=["chat", "toy"], default="chat",
                    help="chat (default): 20 adapters, ~200-token prompts and "
                         "replies. toy: the original 12-adapter, 40-token, "
                         "no-prompt workload")
    ap.add_argument("--pool", choices=["separate", "unified", "both"], default="both")
    ap.add_argument("--adapter-mb", type=float, default=float(ADAPTER_MB),
                    help=f"MB per adapter. core.ADAPTER_MB={ADAPTER_MB} "
                         f"(rank-32 all-linear, ELORA's rank).")
    ap.add_argument("--policy", default=None,
                    help="blind|signal|cost-aware|kv-first ; default sweeps a set")
    ap.add_argument("--kv-gpu", type=float, default=None,
                    help="separate-pool KV region MB. Default: --unified-cap minus "
                         "the adapter slab, so both architectures get equal total MB")
    ap.add_argument("--kv-cpu", type=float, default=0.0)
    ap.add_argument("--adapter-cpu", type=float, default=None,
                    help="CPU adapter cache MB, both pools. Default: every "
                         "adapter. vLLM's out-of-box default is max_loras "
                         "adapters (max_cpu_loras)")
    ap.add_argument("--unified-cap", type=float, default=None,
                    help="total GPU MB for EITHER architecture (unified pool size; "
                         "separate = slab + KV of the same total). Default: "
                         "6000 chat / 4000 toy")
    ap.add_argument("--input-tokens", type=float, default=None,
                    help="override mean NEW user-prompt tokens per turn")
    ap.add_argument("--unified-cpu", type=float, default=0.0)
    ap.add_argument("--max-loras", type=int, nargs="+", default=None)
    ap.add_argument("--hw", default="ours",
                    choices=["ours", "elora", "elora-aggressive", "elora-conservative"],
                    help="hardware profile: OURS (~A10, PCIe 40GB/s) or ELORA's "
                         "H800 regime (PCIe 128GB/s, faster HBM). See core.py.")
    ap.add_argument("--pcie-ms-per-mb", type=float, default=None,
                    help="override the profile's PCIe cost")
    ap.add_argument("--decode-ms-per-token", type=float, default=None)
    ap.add_argument("--prefill-ms-per-token", type=float, default=None)
    ap.add_argument("--swap-cold-ms", type=float, default=None)
    args = ap.parse_args()
    d = DEFAULTS[args.workload]
    cap = args.unified_cap if args.unified_cap is not None else d["cap"]
    mls = args.max_loras or d["max_loras"]
    n_adapters = WORKLOADS[args.workload]["n_adapters"]

    hw = get_profile(args.hw)
    _ov = {}
    if args.pcie_ms_per_mb is not None: _ov["pcie_ms_per_mb"] = args.pcie_ms_per_mb
    if args.decode_ms_per_token is not None: _ov["decode_ms_per_token"] = args.decode_ms_per_token
    if args.prefill_ms_per_token is not None: _ov["prefill_ms_per_token"] = args.prefill_ms_per_token
    if args.swap_cold_ms is not None: _ov["swap_cold_ms"] = args.swap_cold_ms
    if _ov:
        hw = _replace(hw, **_ov)

    print(__doc__)
    print(f"continuous batching | workload={args.workload} ({n_adapters} adapters) "
          f"| hw={hw.name} | adapter={args.adapter_mb:.0f}MB | total GPU {cap:.0f}MB "
          f"| mean of {SEEDS} seeds\n")
    print("columns: mean = mean end-to-end latency (ms); TTFT = mean time to first\n"
          "token; load s / rebuild s = total seconds spent loading adapters /\n"
          "re-prefilling evicted KV, summed over the run\n")

    pools = ["separate", "unified"] if args.pool == "both" else [args.pool]
    rows = []
    for pool in pools:
        if pool == "separate":
            policies = [args.policy] if args.policy else ["blind", "cost-aware"]
            label = "SEPARATE (vLLM: adapter slab of max_loras slots + KV region)"
        else:
            policies = [args.policy] if args.policy else ["blind", "kv-first"]
            label = "UNIFIED (one pool), SAME max_loras batch cap"
        rows.append((label, pool, policies, None))
    if "unified" in pools:
        rows.append((f"UNIFIED, NO batch cap (any of the {n_adapters} adapters "
                     f"may share a batch)", "unified", ["blind"], n_adapters))

    for label, pool, policies, fixed_ml in rows:
        print(f"== {label} ==")
        print(f"{'max_loras':>9} {'slab%':>6} {'policy':>11} {'p50':>7} {'mean':>7}"
              f" {'TTFT':>6} {'TPOT':>6} {'stale KV':>9} {'load s':>7} {'rebuild s':>9}")
        print("-" * 86)
        for ml in mls:
            for pol in policies:
                m = avg(pool=pool, max_loras=fixed_ml or ml, policy_name=pol,
                        adapter_size=args.adapter_mb, kv_gpu_mb=args.kv_gpu,
                        adapter_cpu=args.adapter_cpu, kv_cpu=args.kv_cpu,
                        unified_cap=cap, unified_cpu=args.unified_cpu,
                        hw=hw, mean_input_tokens=args.input_tokens,
                        kind=args.workload)
                slab = (f"{100 * ml * args.adapter_mb / cap:5.0f}%"
                        if pool == "separate" else f"{'-':>6}")
                print(f"{(fixed_ml or ml):>9} {slab} {pol:>11} {m['p50']:>7.0f}"
                      f" {m['mean_lat']:>7.0f} {m['ttft_mean']:>6.0f} {m['tpot']:>6.1f}"
                      f" {m['stale']:>8.1f}% {m['load_s']:>7.1f} {m['rebuild_s']:>9.1f}")
            if fixed_ml:
                break
        print()


if __name__ == "__main__":
    main()
