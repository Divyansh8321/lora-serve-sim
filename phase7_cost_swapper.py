"""PHASE 7 -- ELORA's cost-model swapper: does the timer earn its keep?

WORLD: phase 6's substrate -- unified pool, continuous batching, radix prefix
       tree -- plus ELORA's THIRD component: a swapper that runs on a 100ms
       TIMER instead of only reacting when memory fills up.

       When memory is tight, it re-scores every leaf node:
           Eval_i = w_swap * swap_cost_mb_i          (big -> keep)
                  + w_freq * recent_hits_i            (hot -> keep)
                  + w_lru  * (1 - sigmoid(age_i))     (recently used -> keep)
                  + w_floor * enough_loras_term       (don't evict so many
                                                       LoRAs everything cold-starts)
       and evicts lowest-Eval leaves down to a low-water mark.
       When memory is IDLE, it PREFETCHES the adapter for the soonest upcoming
       (adapter, group), paying the cold-start before the request lands.

       ELORA's ablation: replace the whole thing with plain LRU (ELORA-WOS)
       -> 1.42x worse TTFT. So they claim the timer + cost model is nearly as
       important as the dependency manager.

HARDWARE PROFILE (--hw, see core.py). ELORA's swapper is built for ELORA's
       engine: transfers overlap inference on CUDA streams ("no extra swapping
       overhead", VII), and it swaps OUT only "when full", swaps IN below 70%
       util (VI-C). Our model's defaults are the opposite -- synchronous swap
       cost, proactive swap-out at 92% fill -- so an apples-to-apples test must
       align the ENGINE, not just the constants. Three switches, each also a
       standalone flag for one-at-a-time attribution:
         --swap-mode {sync,async}       async: adapter transfer does not stall
                                        the step; the request that needs it is
                                        deferred until _adapter_ready.
         --prefetch-cost {charged,overlapped}   charged (our default): the
                                        prefetch copy costs wall time (folded
                                        into the next step -- honest for sync
                                        HW). overlapped: runs on a stream, free.
         --swap-out-when {full,highwater}   full: the 100ms timer only
                                        prefetches; make_room() still evicts
                                        reactively on demand. NO proactive churn.
       `--hw elora[-aggressive|-conservative]` sets async + overlapped + full
       together (plus PCIe/HBM constants).

WHY A BURSTY WORKLOAD: the swapper only helps under bursts + drift. Steady
       Poisson + fixed Zipf (phases 2-6) never gives it a lull to prefetch in
       or a moving hot-set to track. core.make_bursty_workload adds both:
       inhomogeneous Poisson arrivals (burst_factor x rate for a duty fraction
       of each period) and a Zipf ranking re-shuffled every drift period.

POLICIES:
  react-lru     : evict on-full only, LRU. phases 4-6 baseline.
  react-dep     : evict on-full only, dependency-aware (phase 6's dep-aware).
  swap-full     : 100ms timer + full Eval cost model + idle prefetch. ELORA.
  swap-noprefetch : timer + cost model, NO prefetch. isolates prefetch value.
  swap-wo-freq  : cost model minus the frequency term. ELORA-WOV analogue.
  swap-wo-swap  : cost model minus the swap-cost term. ELORA-WOC analogue.

QUESTION: on a bursty, drifting workload, does the timer + cost model beat
       react-on-full, and by how much -- and which term carries it? Does it
       reproduce ELORA's ~1.4x, or is it a smaller effect here too?

FINDING (mean over 5 seeds, pool 2400MB, prefix 400 tok, 60s drift).

  AT OUR ENGINE (--hw ours: sync swap, prefetch charged, proactive swap-out):

    workload    react-lru  react-dep  swap-full   swap vs react-lru
    STEADY         753        745        788           -4.7%
    BURST x5       877        845       1029          -17.4%
    BURST x10     1497       1460       1820          -21.6%

  The timer-driven cost model makes things WORSE, progressively so as bursts
  intensify -- the opposite of ELORA's "1.42x without it". WHY: the proactive
  swap-OUT at 92% fill is pure churn -- it evicts entries needed again seconds
  later, so adapter loads climb 14 -> 28 -> 38 (swap-full) vs 14 -> 23
  (react-lru). (Earlier drafts showed -3.7/-11.8/-15.6 here; the current, more
  negative numbers are honest -- prefetch used to be modelled as free; it now
  costs wall time under --prefetch-cost charged.)

  AT ELORA'S ENGINE (--hw elora: async swap, prefetch overlapped, evict-on-full):

    workload    react-lru  swap-full  swap vs react-lru   (aggressive / conservative)
    STEADY       261 / 352   260 / 353    +0.4% / -0.4%
    BURST x5     263 / 364   262 / 360    +0.4% / +1.2%
    BURST x10    277 / 401   271 / 390    +2.0% / +2.9%

  THE SIGN FLIPS. swap-full is neutral-to-positive at ELORA's engine, and the
  adapter-load churn is gone (swap-full 13-16 vs react-lru 19-24). So the
  phase-7 negative result was a HARDWARE+MECHANISM-REGIME artifact, not a
  policy defect.

  WHICH SWITCH CARRIES THE FLIP (--sweep, burst x10, one switch at a time from
  the `ours` baseline):

    switch      swap-full vs react-lru
    none          -21.6%
    pcie          -21.6%   (no-op: phase 7 has no PCIe KV path)
    decode        -90.2%   (WORSE: a fixed proactive-eviction overhead is a
                            bigger fraction of a smaller step)
    swapmode      -2.6%    (async: big improvement, not quite over the line)
    prefetch      -16.5%   (overlapped prefetch: small help)
    swapout       +2.2%    <-- THE FLIP: "evict only when full"
    all           +2.0%

  NO SINGLE HARDWARE CONSTANT flips it: --sweep-const swap-cold (300 -> 10ms)
  and --sweep-const decode (12 -> 2) both stay negative the whole way. The
  cause is the PROACTIVE-EVICTION POLICY CHOICE. ELORA's "evict only when full"
  is what avoids the churn; async swap compounds it.

  CONTINUOUS compute-speed sweep (--sweep-scale, decode+prefill scaled
  TOGETHER as one "how much faster is the GPU" dial, 1.0x=ours down to
  0.1x=10x faster) confirms this is not a two-point artifact:
    burst x10 : -21% (1x) -> -104% (~4.6x, WORST) -> -75-80% (10x). Never
                crosses zero at any speed tried.
    burst x5  : gap shrinks monotonically, -17% -> -6%, approaching but
                never reaching zero.
    steady    : gap shrinks monotonically, -5% -> -2%, same pattern.
  Compute speed alone never rescues the proactive-eviction policy, at any
  burst level or any speed -- confirming --swap-out-when full is genuinely
  the causal switch, not an artifact of only checking the two discrete
  aggressive/conservative endpoints.

  - swap-noprefetch (-17.8% at x5, ours): confirms the proactive swap-out is
    the harm, not prefetch.
  - swap-wo-freq / swap-wo-swap within 0.5% of swap-full: no cost-model term
    carries anything while the whole timer approach is net-negative.
  - react-dep (phase 6 dependency-aware eviction, NO timer) is the consistent
    winner at BOTH engines: +1% to +4% over react-lru.

  RECONCILIATION with ELORA-WOS's 1.42x: ELORA-WOS keeps the proactive timer +
  prefetch and only swaps the SCORING for LRU; our react-lru has no timer.
  ELORA's engine (async streams, evict-on-full) is what makes a 100ms swapper
  net-positive. Our earlier negative combined a synchronous, proactively-
  evicting model with A10-class hardware -- three mismatches, each now measured.

REAL-DATA CHECK: run_on_real_traces.py replays real Mooncake conversations
  (real prefix-sharing) with real BurstGPT arrival timing instead of this
  file's synthetic bursty workload. Direction still matches (negative at
  --hw ours, ~zero-to-positive at --hw elora) but the MAGNITUDE collapses to
  noise (-0.7% -> +0.1%, vs double digits here) -- our synthetic bursts are
  sharper/more favorable-to-testing than BurstGPT's real, messier ones. See
  REAL_DATA_FINDINGS.md.

Run:  python phase7_cost_swapper.py
      python phase7_cost_swapper.py --hw elora-aggressive --burst 1 5 10
      python phase7_cost_swapper.py --sweep --hw-list ours --burst 10
      python phase7_cost_swapper.py --sweep-const swap-cold --sweep-range 10 300 8 --burst 10
      python phase7_cost_swapper.py --sweep-scale --sweep-range 1.0 0.1 10 --burst 10
"""

import argparse
import math
from collections import deque

from dataclasses import replace as _replace
from core import (MB_PER_TOKEN, PREFILL_MS_PER_TOKEN, DECODE_MS_PER_TOKEN,
                  SWAP_COLD_MS, ADAPTER_MB, HardwareProfile, OURS, get_profile,
                  make_bursty_workload, percentile)

STEP_TOKEN_BUDGET = 512
FIXED_STEP_MS = 4.0
PER_SEQ_STEP_MS = 0.05
MAX_BATCH_SEQS = 192
WAIT_THRESHOLD_MS = 30000

SWAPPER_INTERVAL_MS = 100.0
HIGH_WATER = 0.92          # start swapping out above this pool fill
LOW_WATER = 0.75           # swap out down to here
IDLE_PREFETCH_BELOW = 0.60 # only prefetch when pool this empty AND batch small
FREQ_HALFLIFE_MS = 15000.0


def _sigmoid(x):
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ==========================================================================
# Radix tree (same structure as phase 6, trimmed to what phase 7 uses)
# ==========================================================================

class RadixNode:
    __slots__ = ("key", "tokens", "adapter_id", "parent", "children",
                 "last_used", "refs", "is_prefix", "hits", "hit_stamp")

    def __init__(self, key, tokens, adapter_id, parent, is_prefix):
        self.key = key
        self.tokens = tokens
        self.adapter_id = adapter_id
        self.parent = parent
        self.children = {}
        self.last_used = 0.0
        self.refs = 0
        self.is_prefix = is_prefix
        self.hits = 0.0            # decayed hit counter
        self.hit_stamp = 0.0

    def mb(self):
        return self.tokens * MB_PER_TOKEN

    def is_leaf(self):
        return not self.children

    def bump_hits(self, now):
        decay = 0.5 ** ((now - self.hit_stamp) / FREQ_HALFLIFE_MS) if self.hit_stamp else 1.0
        self.hits = self.hits * decay + 1.0
        self.hit_stamp = now


class RadixTree:
    def __init__(self):
        self.roots = {}
        self.convo_leaf = {}
        self.resident = set()
        self._resident_nodes = {}         # id(node) -> node, O(residents) scans
        self._nodes = []
        self._resident_mb = 0.0

    def get_prefix_node(self, adapter_id, group, prefix_tokens):
        k = (adapter_id, group)
        n = self.roots.get(k)
        if n is None:
            n = RadixNode(k, prefix_tokens, adapter_id, None, is_prefix=True)
            self.roots[k] = n
            self._nodes.append(n)
        return n

    def get_convo_leaf(self, cid, adapter_id, group, prefix_tokens):
        n = self.convo_leaf.get(cid)
        if n is None:
            parent = (self.get_prefix_node(adapter_id, group, prefix_tokens)
                      if prefix_tokens > 0 else None)
            n = RadixNode(("c", cid), 0, adapter_id, parent, is_prefix=False)
            if parent is not None:
                parent.children[cid] = n
            self.convo_leaf[cid] = n
            self._nodes.append(n)
        return n

    def is_resident(self, n):
        return id(n) in self.resident

    def make_resident(self, n):
        if id(n) not in self.resident:
            self.resident.add(id(n))
            self._resident_nodes[id(n)] = n
            self._resident_mb += n.mb()

    def evict(self, n):
        if id(n) in self.resident:
            self.resident.discard(id(n))
            self._resident_nodes.pop(id(n), None)
            self._resident_mb -= n.mb()

    def resize(self, n, new_tokens):
        if id(n) in self.resident:
            self._resident_mb += (new_tokens - n.tokens) * MB_PER_TOKEN
        n.tokens = new_tokens

    def resident_mb(self):
        return self._resident_mb

    def all_nodes(self):
        return self._nodes

    def resident_nodes(self):
        """O(residents), not O(all nodes ever created)."""
        return self._resident_nodes.values()

    def shared_by(self, n):
        return max(1, len(n.children)) if n.is_prefix else 1

    def match(self, cid, adapter_id, group, prefix_tokens, history_tokens):
        path = []
        reused = 0.0
        if prefix_tokens > 0:
            p = self.get_prefix_node(adapter_id, group, prefix_tokens)
            path.append(p)
            if self.is_resident(p):
                reused += p.mb()
        leaf = self.get_convo_leaf(cid, adapter_id, group, prefix_tokens)
        self.resize(leaf, history_tokens)
        path.append(leaf)
        if self.is_resident(leaf):
            reused += leaf.mb()
        return reused, path


# ==========================================================================
# Eviction / swapper policies
# ==========================================================================

class Policy:
    name = "base"
    uses_timer = False
    prefetch = False
    # cost-model term switches (swap-* policies)
    w_swap = 1.0
    w_freq = 1.0
    w_lru = 1.0
    w_floor = 1.0

    def __init__(self):
        self.a_last = {}

    def touch_adapter(self, a, now):
        self.a_last[a] = now

    # ---- react-on-full eviction (all policies fall back to this) ----
    def rank_reactive(self, sim, candidates):
        raise NotImplementedError


class ReactLRU(Policy):
    name = "react-lru"

    def rank_reactive(self, sim, candidates):
        return sorted(candidates, key=lambda n: n.last_used)


class ReactDep(Policy):
    name = "react-dep"

    def rank_reactive(self, sim, candidates):
        def key(n):
            adapter_gone = n.adapter_id not in sim.adapters_mb
            return (0 if adapter_gone else 1,      # orphaned KV first
                    sim.tree.shared_by(n),          # low fan-out before high
                    n.mb(),                         # small before big
                    n.last_used)
        return sorted(candidates, key=key)


class SwapCostModel(Policy):
    """ELORA-style. Timer-driven Eval scoring + (optionally) idle prefetch.
    Reactive fallback uses the same Eval ranking."""
    name = "swap-full"
    uses_timer = True
    prefetch = True

    def eval_score(self, sim, n, now):
        # HIGHER = more valuable = evict later
        swap_cost = n.mb()                                     # bytes to refill
        age = (now - n.last_used) / max(1.0, FREQ_HALFLIFE_MS)
        recency = 1.0 - _sigmoid(age - 2.0)                    # ~1 fresh, ~0 stale
        freq = n.hits
        # floor: if evicting this would drop an adapter to zero resident KV,
        # and few adapters are resident, protect it a bit
        floor = 0.0
        if not n.is_prefix:
            same = sum(1 for m in sim.tree.resident_nodes()
                       if m.adapter_id == n.adapter_id)
            if same <= 1 and len(sim.adapters_mb) <= sim.max_loras:
                floor = 1.0
        # orphaned KV (adapter already gone) is worthless -> large negative
        if n.adapter_id not in sim.adapters_mb:
            return -1e6 + n.last_used
        shared_bonus = math.log2(sim.tree.shared_by(n) + 1)
        return (self.w_swap * swap_cost * 0.01
                + self.w_freq * freq
                + self.w_lru * recency * 5.0
                + self.w_floor * floor * 10.0
                + shared_bonus)

    def rank_reactive(self, sim, candidates):
        now = sim.now
        return sorted(candidates, key=lambda n: self.eval_score(sim, n, now))


class SwapNoPrefetch(SwapCostModel):
    name = "swap-noprefetch"
    prefetch = False


class SwapWoFreq(SwapCostModel):
    name = "swap-wo-freq"
    w_freq = 0.0


class SwapWoSwap(SwapCostModel):
    name = "swap-wo-swap"
    w_swap = 0.0


POLICIES = {p.name: p for p in [ReactLRU, ReactDep, SwapCostModel,
                                SwapNoPrefetch, SwapWoFreq, SwapWoSwap]}


# ==========================================================================
# Sequence state
# ==========================================================================

class SeqState:
    __slots__ = ("req", "cid", "adapter_id", "prefill_left", "decode_left",
                 "phase", "leaf", "path")

    def __init__(self, req, cid, adapter_id, to_prefill, path):
        self.req = req
        self.cid = cid
        self.adapter_id = adapter_id
        self.prefill_left = to_prefill
        self.decode_left = req.output_tokens
        self.phase = "prefill" if to_prefill > 0 else "decode"
        self.leaf = path[-1]
        self.path = path

    def done(self):
        return self.prefill_left <= 0 and self.decode_left <= 0


# ==========================================================================
# Simulator
# ==========================================================================

class Phase7Sim:
    def __init__(self, requests, conversations, pool_mb, policy, max_loras,
                 hw=OURS):
        hw = get_profile(hw)
        self.hw = hw
        self.adapter_mb = hw.adapter_mb
        self.prefill_ms_per_token = hw.prefill_ms_per_token
        self.decode_ms_per_token = hw.decode_ms_per_token
        self.swap_cold_ms = hw.swap_cold_ms
        self.swap_mode = hw.swap_mode              # "sync" | "async"
        self.prefetch_cost = hw.prefetch_cost      # "charged" | "overlapped"
        self.swap_out_when = hw.swap_out_when      # "highwater" | "full"
        self.pool_mb = pool_mb
        self.policy = policy
        self.max_loras = max_loras
        self.tree = RadixTree()
        self.adapters_mb = {}
        # async swap: adapter id -> wall time its transfer completes
        self._adapter_ready = {}
        # sync prefetch: cost accrued this tick, folded into the next step
        self._pending_prefetch_ms = 0.0
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}

        self.now = 0.0
        self.waiting = deque()
        self.batch = []
        self.finished = []
        self.last_swapper_run = -1e9

        self.adapter_loads = 0
        self.prefetch_loads = 0
        self.recompute_tokens = 0
        self.reused_tokens = 0
        self.steps = 0
        self.stale_samples = []
        self.ttft = {}
        self.tpot_num = 0.0
        self.tpot_den = 0
        # future demand hint for prefetch: (adapter, group) -> earliest arrival
        self._future = {}
        for r in requests:
            c = self.convos[r.conversation_id]
            k = (c.adapter_id, getattr(c, "prefix_group", 0))
            if k not in self._future or r.arrival_time < self._future[k]:
                self._future[k] = r.arrival_time

    # ---------- helpers ----------
    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            self.waiting.append(self.incoming.popleft())

    def used_mb(self):
        return sum(self.adapters_mb.values()) + self.tree.resident_mb()

    def free_mb(self):
        return self.pool_mb - self.used_mb()

    def fill(self):
        return self.used_mb() / self.pool_mb

    def pinned_nodes(self):
        p = set()
        for s in self.batch:
            for n in s.path:
                p.add(id(n))
        return p

    def pinned_adapters(self):
        return {s.adapter_id for s in self.batch}

    def stale_pct(self):
        total = self.tree.resident_mb()
        if total == 0:
            return 0.0
        stale = sum(n.mb() for n in self.tree.resident_nodes()
                    if n.adapter_id not in self.adapters_mb)
        return 100.0 * stale / total

    # ---------- eviction ----------
    def _evict_to(self, target_free, protect_nodes, protect_adapters):
        cands = [n for n in self.tree.resident_nodes() if id(n) not in protect_nodes]
        for n in self.policy.rank_reactive(self, cands):
            if self.free_mb() >= target_free - 1e-9:
                return
            self.tree.evict(n)
        # still short: drop unpinned adapters, LRU-ish
        for a in sorted([a for a in self.adapters_mb if a not in protect_adapters],
                        key=lambda a: self.policy.a_last.get(a, -1)):
            if self.free_mb() >= target_free - 1e-9:
                return
            self.adapters_mb.pop(a, None)

    def make_room(self, need_mb):
        if self.free_mb() >= need_mb - 1e-9:
            return
        self._evict_to(need_mb, self.pinned_nodes(), self.pinned_adapters())

    # ---------- the 100ms swapper ----------
    def run_swapper(self):
        if not self.policy.uses_timer:
            return
        if self.now - self.last_swapper_run < SWAPPER_INTERVAL_MS:
            return
        self.last_swapper_run = self.now

        # 1. proactive swap-OUT if above high-water.  ELORA (VI-C) evicts only
        #    "when full" -- --swap-out-when full skips this, leaving make_room()
        #    to evict reactively on demand.
        if self.swap_out_when == "highwater" and self.fill() > HIGH_WATER:
            target_free = self.pool_mb * (1.0 - LOW_WATER)
            self._evict_to(target_free, self.pinned_nodes(),
                           self.pinned_adapters())

        # 2. idle prefetch: pool has room AND batch is small -> pull in the
        #    adapter for the soonest upcoming (adapter, group) not resident
        if (self.policy.prefetch and self.fill() < IDLE_PREFETCH_BELOW
                and len(self.batch) < self.max_loras):
            soon = sorted(self._future.items(), key=lambda kv: kv[1])
            for (a, _g), t_arr in soon:
                if t_arr < self.now or t_arr > self.now + 5000:
                    continue
                if a in self.adapters_mb:
                    continue
                if self.free_mb() >= self.adapter_mb:
                    self.adapters_mb[a] = self.adapter_mb
                    self.prefetch_loads += 1
                    if self.prefetch_cost == "charged":
                        # sync HW: the copy costs wall time, folded into the
                        # next step (honest -- makes 'ours' slightly worse).
                        self._pending_prefetch_ms += self.swap_cold_ms
                    else:
                        # overlapped: runs on a CUDA stream during the lull;
                        # ready before the burst, no wall-clock charge, but the
                        # adapter is not USABLE until the transfer finishes.
                        self._adapter_ready[a] = self.now + self.swap_cold_ms
                    break

    # ---------- ensure resident ----------
    def ensure_adapter(self, a):
        if a in self.adapters_mb:
            return 0.0
        self.make_room(self.adapter_mb)
        if self.free_mb() < self.adapter_mb - 1e-9:
            return 0.0
        self.adapters_mb[a] = self.adapter_mb
        self.adapter_loads += 1
        if self.swap_mode == "async":
            # transfer overlaps inference (ELORA's Torch-stream model): no
            # wall-clock charge to this step; the request that needs `a` is
            # deferred in schedule() until _adapter_ready[a].
            self._adapter_ready[a] = self.now + self.swap_cold_ms
            return 0.0
        return self.swap_cold_ms

    # ---------- scheduling ----------
    def schedule(self):
        if not self.waiting:
            return 0.0
        self.waiting = deque(sorted(self.waiting, key=lambda r: r.arrival_time))
        deferred = deque()
        swap_ms = 0.0
        while self.waiting:
            if len(self.batch) >= MAX_BATCH_SEQS:
                break
            r = self.waiting[0]
            starving = (self.now - r.arrival_time) > WAIT_THRESHOLD_MS
            convo = self.convos[r.conversation_id]
            a = convo.adapter_id
            group = getattr(convo, "prefix_group", 0)
            ptoks = getattr(convo, "prefix_tokens", 0)

            batch_adapters = {s.adapter_id for s in self.batch}
            if a not in batch_adapters and len(batch_adapters) >= self.max_loras:
                if not starving:
                    deferred.append(self.waiting.popleft())
                    continue

            self.waiting.popleft()
            swap_ms += self.ensure_adapter(a)
            if a not in self.adapters_mb:
                deferred.append(r)
                continue
            # async swap: adapter memory is reserved but the transfer is still
            # in flight -- the request cannot run until it lands. Other adapters
            # in the batch proceed unaffected (that is the "overlap").
            if self.now < self._adapter_ready.get(a, 0.0):
                deferred.append(r)
                continue

            history = convo.kv_cache_size
            reused, path = self.tree.match(r.conversation_id, a, group, ptoks,
                                           history)
            need = sum(n.mb() for n in path if not self.tree.is_resident(n))
            self.make_room(need)
            to_prefill = 0
            for n in path:
                if not self.tree.is_resident(n):
                    if self.free_mb() >= n.mb() - 1e-9:
                        self.tree.make_resident(n)
                    to_prefill += n.tokens
                n.last_used = self.now
                n.bump_hits(self.now)
            self.reused_tokens += int(round(reused / MB_PER_TOKEN))
            self.recompute_tokens += to_prefill

            seq = SeqState(r, r.conversation_id, a, to_prefill, path)
            for n in path:
                n.refs += 1
            self.batch.append(seq)
            self.policy.touch_adapter(a, self.now)

        self.waiting.extendleft(reversed(deferred))
        return swap_ms

    # ---------- one step ----------
    def step(self):
        self.run_swapper()
        swap_ms = self.schedule()
        if not self.batch:
            if self.incoming:
                self.now = max(self.now, self.incoming[0].arrival_time)
                return
            if self.waiting:
                # pool cannot fit the oldest waiter: force it, advance a step
                r = self.waiting[0]
                self.now += FIXED_STEP_MS + self.decode_ms_per_token
                self.batch and None
                self.waiting.popleft()
                r.finish(self.now - r.arrival_time)
                self.finished.append(r)
            return

        n_decode = sum(1 for s in self.batch if s.phase == "decode")
        prefill_room = STEP_TOKEN_BUDGET - n_decode
        prefill_this = 0
        for s in self.batch:
            if s.phase == "prefill" and prefill_room > 0:
                chunk = min(s.prefill_left, prefill_room)
                s.prefill_left -= chunk
                prefill_this += chunk
                prefill_room -= chunk
                if s.prefill_left <= 0:
                    s.phase = "decode"

        step_ms = (FIXED_STEP_MS + PER_SEQ_STEP_MS * len(self.batch)
                   + self.decode_ms_per_token
                   + prefill_this * self.prefill_ms_per_token
                   + swap_ms
                   + self._pending_prefetch_ms)     # sync prefetch cost, if any
        self._pending_prefetch_ms = 0.0
        self.now += step_ms
        self.steps += 1

        still = []
        for s in self.batch:
            if s.phase == "decode":
                if s.req.request_id not in self.ttft:
                    self.ttft[s.req.request_id] = self.now - s.req.arrival_time
                s.decode_left -= 1
                self.tpot_num += step_ms
                self.tpot_den += 1
                convo = self.convos[s.cid]
                convo.grow_cache(1)
                self.tree.resize(s.leaf, convo.kv_cache_size)
                if self.tree.is_resident(s.leaf):
                    s.leaf.last_used = self.now
            if s.done():
                s.req.finish(self.now - s.req.arrival_time)
                self.finished.append(s.req)
                for n in s.path:
                    n.refs = max(0, n.refs - 1)
            else:
                still.append(s)
        self.batch = still

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

SEEDS = 5


def trial(burst_factor, policy_name, seed, pool_mb, prefix_tokens=400,
          max_loras=5, drift_period=60000.0, hw="ours"):
    convos, reqs = make_bursty_workload(
        n_conversations=60, n_adapters=12, turns_per_convo=5, mean_tokens=40,
        base_rate=0.0006, turn_gap=6000, shared_prefix_tokens=prefix_tokens,
        shared_prefix_groups=3, burst_factor=burst_factor,
        burst_period=40000.0, burst_duty=0.25,
        popularity_drift_period=drift_period, seed=seed)
    sim = Phase7Sim(reqs, convos, pool_mb, POLICIES[policy_name](), max_loras,
                    hw=hw)
    sim.run()
    lats = [r.latency() for r in sim.finished]
    ttfts = list(sim.ttft.values())
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)
             if sim.stale_samples else 0.0)
    tpot = sim.tpot_num / sim.tpot_den if sim.tpot_den else float("nan")
    return {
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "ttft_p50": percentile(ttfts, 50),
        "tpot": tpot,
        "stale": stale,
        "adapter_loads": sim.adapter_loads,
        "prefetch": sim.prefetch_loads,
        "recompute_tokens": sim.recompute_tokens,
    }


def avg(**kw):
    rows = [trial(seed=s, **kw) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rows) / SEEDS for k in rows[0]}


# --- hardware-profile helpers (shared by main + sweep modes) --------------

_SWITCH_FIELDS = {
    "pcie": ["pcie_ms_per_mb"],        # phase 7 has no PCIe KV path -> no-op here, kept for parity
    "decode": ["decode_ms_per_token", "prefill_ms_per_token"],
    "swapmode": ["swap_mode"],
    "prefetch": ["prefetch_cost"],
    "swapout": ["swap_out_when"],
}


def _apply_switch(base, switch, elora):
    """Return `base` with the field(s) named by `switch` set to their `elora`
    profile values -- for one-at-a-time attribution."""
    if switch == "none":
        return base
    if switch == "all":
        fields = [f for fs in _SWITCH_FIELDS.values() for f in fs]
    else:
        fields = _SWITCH_FIELDS[switch]
    return _replace(base, **{f: getattr(elora, f) for f in fields})


def _build_hw(name, overrides):
    hw = get_profile(name)
    if overrides:
        hw = _replace(hw, **overrides)
    return hw


def _cli_overrides(args):
    ov = {}
    if getattr(args, "pcie_ms_per_mb", None) is not None:
        ov["pcie_ms_per_mb"] = args.pcie_ms_per_mb
    if getattr(args, "decode_ms_per_token", None) is not None:
        ov["decode_ms_per_token"] = args.decode_ms_per_token
    if getattr(args, "prefill_ms_per_token", None) is not None:
        ov["prefill_ms_per_token"] = args.prefill_ms_per_token
    if getattr(args, "swap_cold_ms", None) is not None:
        ov["swap_cold_ms"] = args.swap_cold_ms
    if getattr(args, "swap_mode", None):
        ov["swap_mode"] = args.swap_mode
    if getattr(args, "swap_out_when", None):
        ov["swap_out_when"] = args.swap_out_when
    if getattr(args, "prefetch_cost", None):
        ov["prefetch_cost"] = args.prefetch_cost
    return ov


def _delta(base_p50, p50):
    return 100.0 * (base_p50 - p50) / base_p50 if base_p50 else float("nan")


# --- sweep modes ---------------------------------------------------------------

def run_sweep(args):
    """{--hw-list} x {--burst} x {--switch-list} -> the swapper's sign per cell."""
    from core import ELORA_AGGRESSIVE
    print(f"SWEEP  pool {args.pool_mb:.0f}MB | prefix {args.prefix_tokens} tok | "
          f"max_loras {args.max_loras} | mean of {SEEDS} seeds")
    print(f"{'hw':>18} {'burst':>6} {'switch':>10} {'swap-full':>10} "
          f"{'react-lru':>10} {'delta%':>9} {'sign':>5}")
    print("-" * 74)
    for hwname in args.hw_list:
        base_hw = get_profile(hwname)
        for bf in args.burst:
            for sw in args.switch_list:
                hw = _apply_switch(base_hw, sw, ELORA_AGGRESSIVE)
                sf = avg(burst_factor=bf, policy_name="swap-full",
                         pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                         max_loras=args.max_loras, drift_period=args.drift_period,
                         hw=hw)
                rl = avg(burst_factor=bf, policy_name="react-lru",
                         pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                         max_loras=args.max_loras, drift_period=args.drift_period,
                         hw=hw)
                d = _delta(rl["p50"], sf["p50"])
                print(f"{hwname:>18} {bf:>6.0f} {sw:>10} {sf['p50']:>10.0f} "
                      f"{rl['p50']:>10.0f} {d:>+8.1f}% {'+' if d >= 0 else '-':>5}")
        print()
    print("First '+' row = the swapper stops being net-negative. Compare rows to")
    print("see which single switch (pcie / decode / swapmode / prefetch / swapout)")
    print("carries the flip.")


_CONST_FIELD = {
    "pcie": "pcie_ms_per_mb", "decode": "decode_ms_per_token",
    "swap-cold": "swap_cold_ms", "prefill": "prefill_ms_per_token",
}


def _sweep_values(lo, hi, n):
    ratio = (hi / lo) ** (1.0 / (n - 1)) if n > 1 else 1.0
    return [lo * ratio ** i for i in range(n)]


def run_sweep_const(args):
    """Vary ONE hardware constant across a geometric range; find the sign flip."""
    field = _CONST_FIELD[args.sweep_const]
    lo, hi, n = args.sweep_range
    vals = _sweep_values(lo, hi, int(n))
    base_hw = _build_hw(args.hw, _cli_overrides(args))
    print(f"SWEEP-CONST {field}  from {lo:g} to {hi:g} ({int(n)} pts, geometric)")
    print(f"start hw={base_hw.name} | burst {args.burst[0]:.0f} | pool "
          f"{args.pool_mb:.0f}MB | prefix {args.prefix_tokens} tok | "
          f"{args.sweep_policy} vs {args.sweep_baseline} | mean of {SEEDS} seeds\n")
    print(f"{field:>18} {args.sweep_policy:>12} {args.sweep_baseline:>12} "
          f"{'delta%':>9} {'sign':>5}")
    print("-" * 62)
    bf = args.burst[0]
    prev = None
    crossover = None
    for v in vals:
        hw = _replace(base_hw, **{field: v})
        pol = avg(burst_factor=bf, policy_name=args.sweep_policy,
                  pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                  max_loras=args.max_loras, drift_period=args.drift_period, hw=hw)
        bas = avg(burst_factor=bf, policy_name=args.sweep_baseline,
                  pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                  max_loras=args.max_loras, drift_period=args.drift_period, hw=hw)
        d = _delta(bas["p50"], pol["p50"])
        print(f"{v:>18.4f} {pol['p50']:>12.0f} {bas['p50']:>12.0f} "
              f"{d:>+8.1f}% {'+' if d >= 0 else '-':>5}")
        if prev is not None and (prev[1] < 0) != (d < 0):
            # linear-interp the crossover in the swept field
            (v0, d0), (v1, d1) = prev, (v, d)
            crossover = v0 + (v1 - v0) * (0 - d0) / (d1 - d0)
        prev = (v, d)
    print()
    if crossover is not None:
        print(f"VERDICT: {args.sweep_policy} crosses zero at {field} ~= "
              f"{crossover:.4f} (interp, burst x{bf:.0f}).")
    else:
        print(f"VERDICT: no sign flip across the range -- "
              f"{args.sweep_policy} stays {'positive' if prev[1] >= 0 else 'negative'}.")


def run_sweep_scale(args):
    """Vary a single COMPUTE SCALE FACTOR that moves decode_ms_per_token AND
    prefill_ms_per_token together, proportionally -- this is what "how much
    faster is the GPU" actually means (a real GPU doesn't get faster at
    decode without also getting faster at prefill). scale=1.0 is `ours`;
    scale=0.18 matches ELORA_AGGRESSIVE's decode ratio, scale=0.35 matches
    ELORA_CONSERVATIVE's. Finer-grained than the two-point aggressive/
    conservative split -- this answers "exactly how much hardware speedup is
    needed before the sign flips", not just "does either endpoint flip it".
    PCIe and the engine switches (swap-mode/prefetch/swap-out) are held at
    --hw's value throughout, so this isolates compute speed specifically.
    """
    lo, hi, n = args.sweep_range
    vals = _sweep_values(lo, hi, int(n))
    base_hw = _build_hw(args.hw, _cli_overrides(args))
    print(f"SWEEP-SCALE compute factor (decode & prefill scaled together)  "
          f"from {lo:g}x to {hi:g}x ({int(n)} pts, geometric)")
    print(f"start hw={base_hw.name} (decode={base_hw.decode_ms_per_token:.3f} "
          f"prefill={base_hw.prefill_ms_per_token:.4f}) | burst {args.burst[0]:.0f} "
          f"| pool {args.pool_mb:.0f}MB | prefix {args.prefix_tokens} tok | "
          f"{args.sweep_policy} vs {args.sweep_baseline} | mean of {SEEDS} seeds\n")
    print(f"{'scale':>8} {'decode ms':>10} {'prefill ms':>11} "
          f"{args.sweep_policy:>12} {args.sweep_baseline:>12} {'delta%':>9} {'sign':>5}")
    print("-" * 72)
    bf = args.burst[0]
    prev = None
    crossover = None
    for scale in vals:
        hw = _replace(base_hw,
                     decode_ms_per_token=base_hw.decode_ms_per_token * scale,
                     prefill_ms_per_token=base_hw.prefill_ms_per_token * scale)
        pol = avg(burst_factor=bf, policy_name=args.sweep_policy,
                  pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                  max_loras=args.max_loras, drift_period=args.drift_period, hw=hw)
        bas = avg(burst_factor=bf, policy_name=args.sweep_baseline,
                  pool_mb=args.pool_mb, prefix_tokens=args.prefix_tokens,
                  max_loras=args.max_loras, drift_period=args.drift_period, hw=hw)
        d = _delta(bas["p50"], pol["p50"])
        print(f"{scale:>8.3f} {hw.decode_ms_per_token:>10.3f} "
              f"{hw.prefill_ms_per_token:>11.4f} {pol['p50']:>12.0f} "
              f"{bas['p50']:>12.0f} {d:>+8.1f}% {'+' if d >= 0 else '-':>5}")
        if prev is not None and (prev[1] < 0) != (d < 0):
            (s0, d0), (s1, d1) = prev, (scale, d)
            crossover = s0 + (s1 - s0) * (0 - d0) / (d1 - d0)
        prev = (scale, d)
    print()
    print("Reference points: ELORA_AGGRESSIVE ~ 0.18x, ELORA_CONSERVATIVE ~ 0.35x")
    if crossover is not None:
        print(f"VERDICT: {args.sweep_policy} crosses zero at compute scale ~= "
              f"{crossover:.3f}x (interp, burst x{bf:.0f}).")
    else:
        print(f"VERDICT: no sign flip across {lo:g}x-{hi:g}x -- "
              f"{args.sweep_policy} stays {'positive' if prev[1] >= 0 else 'negative'} "
              f"at every compute speed tried.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burst", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    ap.add_argument("--prefix-tokens", type=int, default=400)
    ap.add_argument("--pool-mb", type=float, default=2400.0)
    ap.add_argument("--max-loras", type=int, default=5)
    ap.add_argument("--drift-period", type=float, default=60000.0)
    ap.add_argument("--policies", nargs="+",
                    default=["react-lru", "react-dep", "swap-full",
                             "swap-noprefetch", "swap-wo-freq", "swap-wo-swap"])
    # hardware profile + per-constant / per-switch overrides
    ap.add_argument("--hw", default="ours",
                    choices=["ours", "elora", "elora-aggressive", "elora-conservative"])
    ap.add_argument("--pcie-ms-per-mb", type=float, default=None)
    ap.add_argument("--decode-ms-per-token", type=float, default=None)
    ap.add_argument("--prefill-ms-per-token", type=float, default=None)
    ap.add_argument("--swap-cold-ms", type=float, default=None)
    ap.add_argument("--swap-mode", choices=["sync", "async"], default=None)
    ap.add_argument("--swap-out-when", choices=["full", "highwater"], default=None)
    ap.add_argument("--prefetch-cost", choices=["charged", "overlapped"], default=None)
    # sweep modes
    ap.add_argument("--sweep", action="store_true",
                    help="grid over --hw-list x --burst x --switch-list")
    ap.add_argument("--hw-list", nargs="+", default=["ours", "elora-aggressive"])
    ap.add_argument("--switch-list", nargs="+",
                    default=["none", "pcie", "decode", "swapmode", "prefetch",
                             "swapout", "all"])
    ap.add_argument("--sweep-const", choices=list(_CONST_FIELD),
                    help="vary ONE constant across --sweep-range; find the crossover")
    ap.add_argument("--sweep-scale", action="store_true",
                    help="vary a combined compute-speed factor (decode+prefill "
                         "scaled together) across --sweep-range LO HI N (as "
                         "multipliers, e.g. 1.0 0.1 10); finds the exact "
                         "hardware-speedup crossover instead of just testing "
                         "the aggressive/conservative endpoints")
    ap.add_argument("--sweep-range", type=float, nargs=3, default=[0.008, 0.105, 9],
                    metavar=("LO", "HI", "N"))
    ap.add_argument("--sweep-policy", default="swap-full")
    ap.add_argument("--sweep-baseline", default="react-lru")
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args)
        return
    if args.sweep_const:
        run_sweep_const(args)
        return
    if args.sweep_scale:
        run_sweep_scale(args)
        return

    hw = _build_hw(args.hw, _cli_overrides(args))

    print(__doc__)
    print(f"pool {args.pool_mb:.0f}MB | max_loras {args.max_loras} | prefix "
          f"{args.prefix_tokens} tok | drift {args.drift_period/1000:.0f}s | "
          f"hw={hw.name} (pcie={hw.pcie_ms_per_mb:.4f} decode={hw.decode_ms_per_token:.2f} "
          f"mode={hw.swap_mode} pf={hw.prefetch_cost} out={hw.swap_out_when}) | "
          f"adapter {hw.adapter_mb:.0f}MB | mean of {SEEDS} seeds\n")

    for bf in args.burst:
        tag = "STEADY" if bf <= 1.0 else f"BURST x{bf:.0f}"
        print(f"== {tag} (burst_factor={bf}) ==")
        print(f"{'policy':>16} {'p50':>8} {'p95':>9} {'TTFT p50':>9} {'TPOT':>7}"
              f" {'stale KV':>9} {'ldrs':>6} {'prefetch':>9}   vs react-lru")
        print("-" * 96)
        base = None
        for pol in args.policies:
            m = avg(burst_factor=bf, policy_name=pol, pool_mb=args.pool_mb,
                    prefix_tokens=args.prefix_tokens, max_loras=args.max_loras,
                    drift_period=args.drift_period, hw=hw)
            if pol == "react-lru":
                base = m["p50"]
            delta = f"{100*(base-m['p50'])/base:+.1f}%" if base else ""
            print(f"{pol:>16} {m['p50']:>8.0f} {m['p95']:>9.0f} {m['ttft_p50']:>9.0f}"
                  f" {m['tpot']:>7.2f} {m['stale']:>8.1f}% {m['adapter_loads']:>6.0f}"
                  f" {m['prefetch']:>9.0f}   {delta:>10}")
        print()

    print("Question: does the 100ms timer + cost model + prefetch beat")
    print("react-on-full, and does the gap only open once traffic is bursty?")
    print("ELORA claims replacing it with LRU costs 1.42x TTFT. Run --hw elora")
    print("and --sweep-const pcie to test that apples-to-apples.")


if __name__ == "__main__":
    main()
