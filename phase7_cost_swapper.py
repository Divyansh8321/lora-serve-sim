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
       When memory is IDLE, it PREFETCHES: pulls in adapters / KV it predicts
       will be needed soon, paying the cold-start before the request lands.

       ELORA's ablation: replace the whole thing with plain LRU (ELORA-WOS)
       -> 1.42x worse TTFT. So they claim the timer + cost model is nearly as
       important as the dependency manager.

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

FINDING (mean over 5 seeds, pool 2400MB, prefix 400 tok, 60s drift):

    workload    react-lru  react-dep  swap-full   swap vs react-lru
    STEADY         753        745        781           -3.7%
    BURST x5       877        845        981          -11.8%
    BURST x10     1497       1460       1731          -15.6%

  THE TIMER-DRIVEN COST MODEL MAKES THINGS WORSE HERE -- and progressively
  worse as bursts intensify. This is the OPPOSITE of ELORA's claim that
  removing it costs 1.42x TTFT.

  WHY: the proactive swap-OUT at high-water is pure churn. It evicts entries
  at 92% fill that are needed again seconds later. Adapter loads climb
  14 -> 25 -> 34 (swap-full) vs 14 -> 23 (react-lru). It pays reload cost to
  free memory that reactive eviction would have freed only on demand.

  - swap-noprefetch is WORSE still (-17.8% at x5): the proactive swap-out is
    the harmful part; prefetch is a small (5-6 fires) mitigation, not a
    driver. The lulls between bursts are not long/empty enough for the 5s
    prediction window to land useful prefetches.
  - swap-wo-freq / swap-wo-swap are within 0.3% of swap-full: no single cost
    term carries anything, because the whole timer approach is net-negative
    in this regime.
  - react-dep (phase 6's dependency-aware eviction, NO timer) is the actual
    winner: +1% to +4% over react-lru, consistent across every config
    (tight pool, long prefix, fast drift).

  ROBUSTNESS: swap-full ranges from -16% (default sweep) to +0.6% (fast
  drift, generous lulls) to -1% (tight pool, long prefix). Never a win in
  any configuration tested.

  RECONCILIATION with ELORA-WOS's 1.42x: same pattern as phases 4-6. ELORA-WOS
  replaces the cost model with plain LRU BUT KEEPS proactive timer-driven
  swapping and prefetch scaffolding; our react-lru has no timer at all.
  ELORA's H800 has 128GB/s PCIe (8x ours) and 80GB HBM, so an aggressive
  proactive swapper churns far more cheaply there; and their Azure-trace
  bursts may have the long idle windows prefetch needs. On a single smaller
  GPU with tight memory, reacting-on-demand beats a 100ms timer.

Run:  python phase7_cost_swapper.py
      python phase7_cost_swapper.py --burst 1 5 10 --prefix-tokens 400
"""

import argparse
import math
from collections import deque

from core import (MB_PER_TOKEN, PREFILL_MS_PER_TOKEN, DECODE_MS_PER_TOKEN,
                  SWAP_COLD_MS, ADAPTER_MB, make_bursty_workload, percentile)

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
            self._resident_mb += n.mb()

    def evict(self, n):
        if id(n) in self.resident:
            self.resident.discard(id(n))
            self._resident_mb -= n.mb()

    def resize(self, n, new_tokens):
        if id(n) in self.resident:
            self._resident_mb += (new_tokens - n.tokens) * MB_PER_TOKEN
        n.tokens = new_tokens

    def resident_mb(self):
        return self._resident_mb

    def all_nodes(self):
        return self._nodes

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
            same = sum(1 for m in sim.tree.all_nodes()
                       if m.adapter_id == n.adapter_id and sim.tree.is_resident(m))
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
    def __init__(self, requests, conversations, pool_mb, policy, max_loras):
        self.pool_mb = pool_mb
        self.policy = policy
        self.max_loras = max_loras
        self.tree = RadixTree()
        self.adapters_mb = {}
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
        stale = sum(n.mb() for n in self.tree.all_nodes()
                    if self.tree.is_resident(n)
                    and n.adapter_id not in self.adapters_mb)
        return 100.0 * stale / total

    # ---------- eviction ----------
    def _evict_to(self, target_free, protect_nodes, protect_adapters):
        cands = [n for n in self.tree.all_nodes()
                 if self.tree.is_resident(n) and id(n) not in protect_nodes]
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

        # 1. proactive swap-OUT if above high-water
        if self.fill() > HIGH_WATER:
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
                if self.free_mb() >= ADAPTER_MB:
                    self.adapters_mb[a] = ADAPTER_MB
                    self.prefetch_loads += 1
                    break

    # ---------- ensure resident ----------
    def ensure_adapter(self, a):
        if a in self.adapters_mb:
            return 0.0
        self.make_room(ADAPTER_MB)
        if self.free_mb() < ADAPTER_MB - 1e-9:
            return 0.0
        self.adapters_mb[a] = ADAPTER_MB
        self.adapter_loads += 1
        return SWAP_COLD_MS

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
                self.now += FIXED_STEP_MS + DECODE_MS_PER_TOKEN
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
                   + DECODE_MS_PER_TOKEN
                   + prefill_this * PREFILL_MS_PER_TOKEN
                   + swap_ms)
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
          max_loras=5, drift_period=60000.0):
    convos, reqs = make_bursty_workload(
        n_conversations=60, n_adapters=12, turns_per_convo=5, mean_tokens=40,
        base_rate=0.0006, turn_gap=6000, shared_prefix_tokens=prefix_tokens,
        shared_prefix_groups=3, burst_factor=burst_factor,
        burst_period=40000.0, burst_duty=0.25,
        popularity_drift_period=drift_period, seed=seed)
    sim = Phase7Sim(reqs, convos, pool_mb, POLICIES[policy_name](), max_loras)
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
    args = ap.parse_args()

    print(__doc__)
    print(f"pool {args.pool_mb:.0f}MB | max_loras {args.max_loras} | prefix "
          f"{args.prefix_tokens} tok | drift {args.drift_period/1000:.0f}s | "
          f"adapter {ADAPTER_MB}MB | mean of {SEEDS} seeds\n")

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
                    drift_period=args.drift_period)
            if pol == "react-lru":
                base = m["p50"]
            delta = f"{100*(base-m['p50'])/base:+.1f}%" if base else ""
            print(f"{pol:>16} {m['p50']:>8.0f} {m['p95']:>9.0f} {m['ttft_p50']:>9.0f}"
                  f" {m['tpot']:>7.2f} {m['stale']:>8.1f}% {m['adapter_loads']:>6.0f}"
                  f" {m['prefetch']:>9.0f}   {delta:>10}")
        print()

    print("Question: does the 100ms timer + cost model + prefetch beat")
    print("react-on-full, and does the gap only open once traffic is bursty?")
    print("ELORA claims replacing it with LRU costs 1.42x TTFT.")


if __name__ == "__main__":
    main()
