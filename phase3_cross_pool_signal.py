"""PHASE 3 -- Let the two pools talk. One signal, no shared memory.

WORLD: same separate pools as phase 2. The ONLY change: before evicting an
       adapter, the adapter LRU asks the KV pool a single question --
       "how many GPU-resident KV caches belong to this adapter?" -- and
       prefers to evict adapters with none.

       This is deliberately the CHEAPEST possible fix. No merged pool, no
       re-architecture: one integer crossing a boundary that currently has no
       channel at all. It maps directly onto vLLM's open RFC #37003
       (context-aware KV retention) and issue #45325 (adapter residency is not
       even visible to the scheduler today).

       Two variants:
         signal      -- rank adapters by dependent COUNT, then LRU
         cost-aware  -- rank by dependent MB (size-weighted), and additionally
                        let the KV side evict already-stale KV first

QUESTION: does the cheap signal recover the loss phase 2 measured?

FINDING -- THE KEY NEGATIVE RESULT OF THIS PROJECT:
       The signal works exactly as designed on the metric it targets: stale KV
       falls from ~44% to ~13% (count-based) or ~5% (size-weighted).
       BUT LATENCY DOES NOT IMPROVE. Across every pressure setting tested the
       difference is within noise (+-3%), and sometimes negative.

       WHY: in a separate-pool architecture the adapter slab's capacity is
       fixed and independent. Declining to evict adapter A means evicting
       adapter B instead -- so the TOTAL number of adapter reloads is
       unchanged (~26-30 either way). The signal changes WHICH adapter pays
       the reload, not how many reloads happen. And stale KV is not destroyed:
       when its adapter returns, it is reused for free, so staleness is an
       opportunity cost on space, not a latency cost -- and space you free in
       the KV pool cannot be lent to the adapter pool.

       IMPLICATION: coordination is not enough. The pools must be able to
       trade capacity for coordination to pay off. That is phase 4.

Run:  python phase3_cross_pool_signal.py
"""

from core import make_multi_turn_workload
from pools import (AdapterPool, KVPool, SeparatePoolSim,
                         NoComm, OneWayComm, CostAware, summarize)

SEEDS = 5


def trial(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls, seed):
    convos, reqs = make_multi_turn_workload(n_conversations=40, n_adapters=12,
                                            turns_per_convo=5, mean_tokens=40,
                                            rate=0.0006, turn_gap=6000, seed=seed)
    sim = SeparatePoolSim(reqs, convos, AdapterPool(a_gpu, a_cpu),
                          KVPool(k_gpu, k_cpu), PolicyCls())
    sim.run()
    return summarize(sim)


def avg(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls):
    rs = [trial(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls, s) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rs) / SEEDS for k in rs[0]}


def main():
    print(__doc__)
    print("Adapter slab squeezed; KV pool 800MB; no CPU tiers.\n")
    print(f"{'slab':>6} {'policy':>11} {'p50':>7} {'latency vs blind':>17}"
          f" {'stale KV':>10} {'disk loads':>11}")
    print("-" * 66)
    for a_gpu in [140, 100, 60]:
        base = None
        for P in [NoComm, OneWayComm, CostAware]:
            m = avg(a_gpu, 0, 800, 0, P)
            if base is None:
                base = m['p50']
            delta = 100 * (base - m['p50']) / base
            shown = "  (baseline)" if P is NoComm else f"{delta:+16.1f}%"
            print(f"{str(a_gpu)+'MB':>6} {P.name:>11} {m['p50']:>7.0f} {shown:>17}"
                  f" {m['stale']:>9.1f}% {m['disk']:>11.0f}")
        print()
    print("Staleness collapses. Latency does not move. Freed KV-pool space is")
    print("not fungible with adapter-slab space, so the win has nowhere to go.")


if __name__ == "__main__":
    main()
