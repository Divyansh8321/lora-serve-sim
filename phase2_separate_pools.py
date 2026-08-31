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

FINDING: it exists, and it is large. Squeezing the slab from 12 -> 3 of 12
       adapters drives STALE KV (resident but unusable, because its adapter was
       evicted) from 0% to ~45%. No shared memory is required; two
       independently-timed LRUs are sufficient.

         adapters fit   p50    stale KV   disk loads
             12         1052      0.0%        11
              7         1063      9.7%        16
              5         1134     23.1%        26
              3         1303     44.7%        49

       This brackets ELORA's measured numbers: vLLM suffers 42.4% invalid KV
       (their motivation), and ELORA's own no-dependency-manager ablation
       (ELORA-WOM) still suffers 48.6%. NOTE those are two different systems --
       do not report "~48%" as if it were the vLLM baseline.

       The sweep x-axis is "adapters that fit", not raw MB: identical numbers
       result at ADAPTER_MB 20 / 90 / 180 (see phase 5). The pathology is
       driven by the fit RATIO.

       A CPU tier reduces the cost of each mistake (30ms instead of 300ms) but
       does NOT reduce staleness -- staleness is about which adapter is
       GPU-resident, not how expensive it was to move.

Run:  python phase2_separate_pools.py
"""

from core import make_multi_turn_workload, ADAPTER_MB
from pools import AdapterPool, KVPool, SeparatePoolSim, NoComm, summarize

SEEDS = 5

# The sweep x-axis is "how many of the 12 adapters the GPU slab can hold".
# Slab MB is derived from ADAPTER_MB so the sweep survives changes to the
# adapter size (rank-32 all-linear = 180 MB; see core.py).
SLOTS_SWEEP = [12, 7, 5, 3]
# KV pool sized so all-conversations-resident is comfortable but pressure
# appears as the slab shrinks: ~40 convos * ~1.6 KB/token * a few hundred
# tokens. 4 GB is a realistic single-GPU KV region alongside a LoRA slab.
KV_POOL_MB = 4000


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
    print(f"12 adapters x {ADAPTER_MB}MB = {12*ADAPTER_MB}MB if all resident. "
          f"Squeezing the adapter slab:\n")

    print(f"PART A -- no CPU tier (adapter->disk, KV->recompute). KV pool {KV_POOL_MB}MB.")
    print(f"{'adapter slab':>13} {'fits':>6} {'p50':>7} {'p95':>8} {'stale KV':>10} {'disk loads':>11}")
    print("-" * 62)
    for slots in SLOTS_SWEEP:
        a_gpu = slots * ADAPTER_MB
        m = avg(a_gpu, 0, KV_POOL_MB, 0, NoComm)
        print(f"{str(a_gpu)+'MB':>13} {slots:>6} {m['p50']:>7.0f} {m['p95']:>8.0f}"
              f" {m['stale']:>9.1f}% {m['disk']:>11.0f}")

    print(f"\nPART B -- each pool gets its own CPU tier "
          f"(adapter {6*ADAPTER_MB}MB, KV {KV_POOL_MB}MB CPU).")
    print(f"{'adapter slab':>13} {'fits':>6} {'p50':>7} {'p95':>8} {'stale KV':>10} {'disk loads':>11}")
    print("-" * 62)
    for slots in SLOTS_SWEEP:
        a_gpu = slots * ADAPTER_MB
        m = avg(a_gpu, 6 * ADAPTER_MB, KV_POOL_MB, KV_POOL_MB, NoComm)
        print(f"{str(a_gpu)+'MB':>13} {slots:>6} {m['p50']:>7.0f} {m['p95']:>8.0f}"
              f" {m['stale']:>9.1f}% {m['disk']:>11.0f}")

    print("\nStaleness is driven by adapter-slab pressure, not by transfer cost.")
    print("A CPU tier makes each mistake cheaper; it does not make fewer mistakes.")


if __name__ == "__main__":
    main()
