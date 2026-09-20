"""Run phases 6 and 7 against REAL trace data instead of our synthetic
workload generators, and compare to the synthetic results already recorded
in each phase's docstring.

Fixes two of the "gaps" named in README's Scope & honesty section:
  - phase 6's shared prefix was invented (3 fixed-size groups, one length).
    Here it comes from Mooncake's real hash_ids: real, Zipf-shaped, variable
    per conversation, growing turn-by-turn.
  - phase 7's burstiness was invented (a hand-tuned spike formula). Here
    request arrival timing comes directly from BurstGPT's real gaps.

This does NOT touch phase6_radix_prefix.py / phase7_cost_swapper.py's own
CLI -- it imports their Sim classes and policies directly and drives them
with real_traces.py's output, so the phase files' synthetic-workload
behavior (and all previously-recorded numbers) are unaffected.

HONEST LIMITATIONS (see real_traces.py's docstring for detail):
  - Mooncake's variable-depth prefix chains are collapsed to one
    prefix_tokens number per conversation (the sim's tree model wants one).
  - Neither trace has a LoRA-adapter field; adapter identity is inferred
    from Mooncake's own shared-prompt structure (a defensible proxy, not
    ground truth).
  - BurstGPT arrival timing is applied to Mooncake's conversations (the two
    traces are from different real systems) -- this mixes two real datasets,
    it does not reproduce either one's actual joint arrival+content behavior.

Run:  python run_on_real_traces.py
      python run_on_real_traces.py --pool-mb 3000 --hw elora-aggressive
"""

import argparse
import statistics

from collections import Counter

from core import get_profile, percentile
from real_traces import (load_burstgpt_arrivals, load_mooncake_conversations,
                         load_taskmaster_conversations)
from phase6_radix_prefix import RadixSim, POLICIES as P6_POLICIES
from phase7_cost_swapper import Phase7Sim, POLICIES as P7_POLICIES


def _truncate_to_n_conversations(convos, reqs, n_conversations):
    """Keep the first n_conversations conversations (by original trace order).
    Bounding by CONVERSATION count, not raw request count, matters because
    the BurstGPT arrival overlay spaces out one real gap PER CONVERSATION --
    a dataset with more turns/conversation (Taskmaster, ~11) would otherwise
    get spread across a much longer real BurstGPT window than one with fewer
    turns/conversation (Mooncake, ~1.5) for the "same" request-count cutoff,
    diluting concurrency and making the two real-trace results incomparable."""
    if len(convos) <= n_conversations:
        return convos, reqs
    keep_cids = {c.conversation_id for c in
                sorted(convos, key=lambda c: c.conversation_id)[:n_conversations]}
    reqs = [r for r in reqs if r.conversation_id in keep_cids]
    convos = [c for c in convos if c.conversation_id in keep_cids]
    return convos, reqs


def _overlay_burstgpt_timing(convos, reqs, seed):
    """Overlay real BurstGPT inter-arrival gaps onto a real conversation set,
    preserving each conversation's own turn ORDER and turn COUNT but
    replacing WHEN conversations start with a real bursty schedule. Used for
    any trace that lacks real timestamps (Taskmaster) or where we want to
    control which real arrival process drives it (Mooncake).

    `seed` picks a different WINDOW of BurstGPT's real timeline -- there is
    no synthetic randomness to vary here, both traces are fixed real logs, so
    "seed" means "which slice of real history", not "which random draw".
    """
    first_turns = sorted([r for r in reqs if r.is_first_turn],
                         key=lambda r: r.arrival_time)
    n_needed = len(first_turns)
    window_start = seed * n_needed
    gaps = load_burstgpt_arrivals(limit=n_needed, start_at=window_start)
    if len(gaps) < n_needed:
        gaps = load_burstgpt_arrivals(limit=n_needed, start_at=0)
    new_start = {r.conversation_id: gaps[i] for i, r in enumerate(first_turns)}
    first_by_cid = {r.conversation_id: r.arrival_time for r in first_turns}
    for r in reqs:
        old_first = first_by_cid[r.conversation_id]
        shift = new_start[r.conversation_id] - old_first
        r.arrival_time = r.arrival_time + shift
    reqs.sort(key=lambda r: r.arrival_time)
    return convos, reqs


def build_real_workload(seed, n_conversations=2000, use_burstgpt_timing=True,
                        min_adapter_reuse=1):
    """Mooncake conversations/requests, content + prefix structure REAL.
    Arrival timing: real BurstGPT gaps overlaid onto Mooncake's conversation
    order (if use_burstgpt_timing), else Mooncake's own real timestamps.
    Both are real data; this controls WHICH real arrival process drives it.

    `min_adapter_reuse`: the real trace is heavily long-tailed -- most
    adapters (system-prompt groups) are used by exactly ONE conversation
    (see README: real traffic is far more long-tailed than our synthetic
    Zipf assumption). A cache-management question only exists where an
    adapter is used more than once; min_adapter_reuse=2 restricts to
    conversations whose adapter repeats, which is the traffic our memory
    policies actually have a decision to make about.
    """
    convos, reqs = load_mooncake_conversations()

    if min_adapter_reuse > 1:
        pop = Counter(c.adapter_id for c in convos)
        keep = {c.conversation_id for c in convos if pop[c.adapter_id] >= min_adapter_reuse}
        convos = [c for c in convos if c.conversation_id in keep]
        reqs = [r for r in reqs if r.conversation_id in keep]

    reqs = sorted(reqs, key=lambda r: (r.conversation_id, r.arrival_time))
    convos, reqs = _truncate_to_n_conversations(convos, reqs, n_conversations)

    if use_burstgpt_timing:
        convos, reqs = _overlay_burstgpt_timing(convos, reqs, seed)
    else:
        reqs.sort(key=lambda r: r.arrival_time)
    return convos, reqs


def build_taskmaster_workload(seed, n_conversations=2000):
    """Taskmaster conversations/requests: real task-type adapter identity
    (ELORA's own 'Personal Agents' dataset), real multi-turn structure, real
    reply lengths. Arrival timing: BurstGPT overlay, same as ELORA's own
    documented method for this dataset (it has no timestamps of its own).
    """
    convos, reqs = load_taskmaster_conversations()
    reqs = sorted(reqs, key=lambda r: (r.conversation_id, r.arrival_time))
    convos, reqs = _truncate_to_n_conversations(convos, reqs, n_conversations)
    return _overlay_burstgpt_timing(convos, reqs, seed)


WORKLOADS = {
    "mooncake": build_real_workload,
    "taskmaster": build_taskmaster_workload,
}


def _build(workload, seed, n_conversations, min_adapter_reuse):
    if workload == "mooncake":
        return build_real_workload(seed, n_conversations=n_conversations,
                                   min_adapter_reuse=min_adapter_reuse)
    return build_taskmaster_workload(seed, n_conversations=n_conversations)


def trial_phase6(policy_name, seed, pool_mb, max_loras, hw, n_conversations=2000,
                 min_adapter_reuse=1, workload="mooncake"):
    convos, reqs = _build(workload, seed, n_conversations, min_adapter_reuse)
    sim = RadixSim(reqs, convos, pool_mb, P6_POLICIES[policy_name](),
                   max_loras, prefix_tokens=0, hw=hw)
    sim.run()
    lats = [r.latency() for r in sim.finished]
    ttfts = list(sim.ttft.values())
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)
             if sim.stale_samples else 0.0)
    return {"p50": percentile(lats, 50), "p95": percentile(lats, 95),
            "ttft_p50": percentile(ttfts, 50), "stale": stale,
            "adapter_loads": sim.adapter_loads, "n": len(sim.finished)}


def trial_phase7(policy_name, seed, pool_mb, max_loras, hw, n_conversations=2000,
                 min_adapter_reuse=1, workload="mooncake"):
    convos, reqs = _build(workload, seed, n_conversations, min_adapter_reuse)
    sim = Phase7Sim(reqs, convos, pool_mb, P7_POLICIES[policy_name](),
                    max_loras, hw=hw)
    sim.run()
    lats = [r.latency() for r in sim.finished]
    ttfts = list(sim.ttft.values())
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)
             if sim.stale_samples else 0.0)
    tpot = sim.tpot_num / sim.tpot_den if sim.tpot_den else float("nan")
    return {"p50": percentile(lats, 50), "p95": percentile(lats, 95),
            "ttft_p50": percentile(ttfts, 50), "tpot": tpot, "stale": stale,
            "adapter_loads": sim.adapter_loads,
            "prefetch": sim.prefetch_loads, "n": len(sim.finished)}


WINDOWS = 3  # real-history windows to average over (not synthetic-random seeds)


def avg(trial_fn, **kw):
    rows = [trial_fn(seed=s, **kw) for s in range(WINDOWS)]
    return {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", choices=["mooncake", "taskmaster"], default="mooncake",
                    help="mooncake: real prefix-sharing, adapter identity INFERRED, "
                         "long-tailed (hundreds of mostly-one-off adapters). "
                         "taskmaster: ELORA's own 'Personal Agents' dataset, real "
                         "task-type field used directly as adapter identity, "
                         "moderate real popularity skew (15 types).")
    ap.add_argument("--pool-mb", type=float, default=None,
                    help="unified pool MB. Default: auto-scaled from the real "
                         "adapter count actually in play (see --min-adapter-reuse).")
    ap.add_argument("--max-loras", type=int, default=None,
                    help="default: auto-scaled to a REALISTIC fraction of the "
                         "real adapter count in play (ELORA sweeps slab sizes "
                         "comparable to their real LoRA count, never a huge "
                         "oversubscription ratio -- so neither do we)")
    ap.add_argument("--n-conversations", type=int, default=2000,
                    help="how many real conversations to slice out of the trace "
                         "(bounding by conversation, not raw request count, keeps "
                         "the BurstGPT overlay's real-time density comparable "
                         "across datasets with different turns/conversation)")
    ap.add_argument("--min-adapter-reuse", type=int, default=2,
                    help="(mooncake only) keep only conversations whose adapter "
                         "(system-prompt group) is used by >=N conversations. The "
                         "real trace is heavily long-tailed (most adapters used "
                         "ONCE); reuse=1 includes them but there's no caching "
                         "DECISION to make for a never-reused adapter.")
    ap.add_argument("--windows", type=int, default=3,
                    help="how many real-history windows to average over")
    ap.add_argument("--hw", default="ours",
                    choices=["ours", "elora", "elora-aggressive", "elora-conservative"])
    args = ap.parse_args()
    hw = get_profile(args.hw)
    global WINDOWS
    WINDOWS = args.windows

    print(__doc__)
    convos, reqs = _build(args.workload, seed=0, n_conversations=args.n_conversations,
                          min_adapter_reuse=args.min_adapter_reuse)
    n_adapters = len(set(c.adapter_id for c in convos))

    # --max-loras is a GPU SLAB SIZE, sized as a REALISTIC FRACTION of the real
    # adapter count in play -- ELORA sweeps 20/50/100 against a real, KNOWN
    # LoRA count, never against an unbounded firehose of one-off adapters at a
    # 30:1+ oversubscription ratio. We mirror that: default to ~40% of the
    # real in-play adapter count (capped so it stays a plausible single-GPU
    # slab), so the pressure is genuine competition, not an arbitrary mismatch
    # left over from tuning against our old 12-adapter SYNTHETIC workload.
    max_loras = args.max_loras or max(4, min(60, round(0.4 * n_adapters)))
    pool_mb = args.pool_mb or round((max_loras * hw.adapter_mb + 800) / 100) * 100

    print(f"REAL WORKLOAD ({args.workload}): {len(convos)} conversations, "
          f"{len(reqs)} requests, {n_adapters} adapters actually in play"
          + (f" (min_adapter_reuse={args.min_adapter_reuse})" if args.workload == "mooncake" else ""))
    print(f"hw={hw.name} | pool {pool_mb:.0f}MB (auto) | max_loras {max_loras} (auto, "
          f"~40% of {n_adapters} real adapters) | mean of {WINDOWS} real-history windows\n")
    args.pool_mb, args.max_loras = pool_mb, max_loras

    print("== PHASE 6 policies on REAL prefix-sharing structure ==")
    print(f"{'policy':>12} {'p50':>8} {'p95':>9} {'TTFT p50':>9} {'stale KV':>9}")
    print("-" * 52)
    base = None
    for pol in ["lru-leaf", "ordering", "dep-aware"]:
        m = avg(trial_phase6, policy_name=pol, pool_mb=args.pool_mb,
                max_loras=args.max_loras, hw=hw, n_conversations=args.n_conversations,
                min_adapter_reuse=args.min_adapter_reuse, workload=args.workload)
        if pol == "ordering":
            base = m["p50"]
        tag = f"  dep vs ord {100*(base-m['p50'])/base:+.1f}%" if base and pol == "dep-aware" else ""
        print(f"{pol:>12} {m['p50']:>8.0f} {m['p95']:>9.0f} {m['ttft_p50']:>9.0f}"
              f" {m['stale']:>8.1f}%{tag}")

    print("\n== PHASE 7 policies on REAL arrival timing (BurstGPT) ==")
    print(f"{'policy':>16} {'p50':>8} {'p95':>9} {'TTFT p50':>9} {'TPOT':>7}"
          f" {'stale KV':>9} {'ldrs':>6}   vs react-lru")
    print("-" * 78)
    base = None
    for pol in ["react-lru", "react-dep", "swap-full", "swap-noprefetch"]:
        m = avg(trial_phase7, policy_name=pol, pool_mb=args.pool_mb,
                n_conversations=args.n_conversations, min_adapter_reuse=args.min_adapter_reuse,
                max_loras=args.max_loras, hw=hw, workload=args.workload)
        if pol == "react-lru":
            base = m["p50"]
        delta = f"{100*(base-m['p50'])/base:+.1f}%" if base else ""
        print(f"{pol:>16} {m['p50']:>8.0f} {m['p95']:>9.0f} {m['ttft_p50']:>9.0f}"
              f" {m['tpot']:>7.2f} {m['stale']:>8.1f}% {m['adapter_loads']:>6.0f}"
              f"   {delta:>10}")

    print("\nCompare these numbers to the synthetic-workload numbers in each")
    print("phase's own docstring. Large divergence = our synthetic workload")
    print("shape was doing real work in the earlier conclusions.")


if __name__ == "__main__":
    main()
