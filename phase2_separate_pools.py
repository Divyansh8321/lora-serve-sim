"""PHASE 2 -- Separate adapter and KV pools. The real vLLM architecture.

WORLD: adapters live in a pre-allocated GPU slab (vLLM: --max-loras, managed by
       LoRALRUCache); KV caches live in a separate BlockPool region. They never
       share bytes, and by default neither knows the other exists.

       Part A: no CPU tier. Adapter evicted -> disk (300ms). KV evicted -> gone
               (full recompute).
       Part B: each pool gets its own CPU tier. Adapter -> CPU (30ms) -> disk.
               KV -> CPU (0.105 ms/MB over PCIe) -> gone.

QUESTION: does the staleness pathology exist in a genuinely separate-pool
       architecture, or was it an artifact of our earlier shared-pool model?

FINDING: it exists, and it is large. With the adapter slab holding only 5 of 12
       adapters, up to ~44% of resident KV becomes STALE -- resident but
       unusable, because the adapter it belongs to has been evicted. No shared
       memory is required to produce this; two independently-timed LRUs are
       sufficient. This is the pathology ELORA/FastLibra report at ~48%.

       A CPU tier reduces the cost of each mistake (30ms instead of 300ms) but
       does NOT reduce staleness, because staleness is about which adapter is
       GPU-resident, not about how expensive it was to move.

Run:  python phase2_separate_pools.py
"""

from core import make_multi_turn_workload
from pools import AdapterPool, KVPool, SeparatePoolSim, NoComm, summarize

SEEDS = 5


def workload(seed):
    return make_multi_turn_workload(n_conversations=40, n_adapters=12,
                                    turns_per_convo=5, mean_tokens=40,
                                    rate=0.0006, turn_gap=6000, seed=seed)


def trial(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls, seed):
    convos, reqs = workload(seed)
    sim = SeparatePoolSim(reqs, convos, AdapterPool(a_gpu, a_cpu),
                          KVPool(k_gpu, k_cpu), PolicyCls())
    sim.run()
    return summarize(sim)


def avg(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls):
    rs = [trial(a_gpu, a_cpu, k_gpu, k_cpu, PolicyCls, s) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rs) / SEEDS for k in rs[0]}


def main():
    print(__doc__)
    print("12 adapters x 20MB = 240MB if all resident. Squeezing the adapter slab:\n")

    print("PART A -- no CPU tier (adapter->disk, KV->recompute). KV pool 800MB.")
    print(f"{'adapter slab':>13} {'fits':>6} {'p50':>7} {'p95':>8} {'stale KV':>10} {'disk loads':>11}")
    print("-" * 62)
    for a_gpu in [240, 140, 100, 60]:
        m = avg(a_gpu, 0, 800, 0, NoComm)
        print(f"{str(a_gpu)+'MB':>13} {a_gpu//20:>6} {m['p50']:>7.0f} {m['p95']:>8.0f}"
              f" {m['stale']:>9.1f}% {m['disk']:>11.0f}")

    print("\nPART B -- each pool gets its own CPU tier (adapter 200MB, KV 800MB CPU).")
    print(f"{'adapter slab':>13} {'fits':>6} {'p50':>7} {'p95':>8} {'stale KV':>10} {'disk loads':>11}")
    print("-" * 62)
    for a_gpu in [240, 140, 100, 60]:
        m = avg(a_gpu, 200, 800, 800, NoComm)
        print(f"{str(a_gpu)+'MB':>13} {a_gpu//20:>6} {m['p50']:>7.0f} {m['p95']:>8.0f}"
              f" {m['stale']:>9.1f}% {m['disk']:>11.0f}")

    print("\nStaleness is driven by adapter-slab pressure, not by transfer cost.")
    print("A CPU tier makes each mistake cheaper; it does not make fewer mistakes.")


if __name__ == "__main__":
    main()
