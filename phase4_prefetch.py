"""PHASE 4 -- Overlap I/O with compute: queue-driven prefetch.

WORLD: same three-tier world as phase 3, plus one physical fact the earlier
       phases ignored. A GPU has TWO independent engines: compute cores and a
       copy engine. They run concurrently. Phases 1-3 charged loads and compute
       serially; real hardware can do them at the same time.

THE IDEA: while the GPU computes a batch for D milliseconds, use the copy
       engine to pull in adapters that will be needed next. Any transfer that
       fits inside D costs the user nothing -- the work still happens, it just
       happens while nobody is waiting.

NO PREDICTION IS INVOLVED. We only prefetch adapters whose requests are
       ALREADY sitting in the pending queue. That is pipelining, not
       forecasting. Predibase's LoRAX ships this as "Adapter Exchange
       Scheduling"; LMCache calls it the PrefetchController.

THE TENSION: a speculative load needs GPU space NOW, while the current batch is
       still resident. So prefetch raises peak memory pressure. Two variants:
         conservative -- only prefetch into genuinely free space
         aggressive   -- evict to make room for a speculative load

FINDING: real, ~6.7% median-latency improvement (24 seeds, 95% CI
       [4.4%, 8.9%]). Per-seed variance is high -- 3 of 24 seeds came out
       slightly negative -- so 8 seeds was NOT enough to distinguish the effect
       from noise; 24 was. Worth remembering before believing any single run.
       The mechanism works as designed: adapter loads from disk drop from
       ~48 to ~10. But the CEILING IS LOW, because prefetch can only remove
       I/O from the critical path, and by phase 3's world a decent policy has
       already made I/O rare. Conservative and aggressive perform about the
       same here -- there is enough slack that prefetch rarely needs to evict,
       so the memory tension is real but does not bite at these pool sizes.

Run:  python phase4_prefetch.py
"""

from mlora.core import (ADAPTER_MB, DECODE_MS_PER_TOKEN, SWAP_COLD_MS, SWAP_WARM_MS,
                        make_multi_turn_workload, percentile)
from phase3_memory_tiers import (TieredCache, TieredSimulator, JointTiered)


class PrefetchPolicy(JointTiered):
    """JointTiered + a prefetch hook. Returns adapters that have requests
    waiting but are not GPU-resident, deepest queue first."""
    name = "prefetch"

    def choose_prefetch(self, pending, cache, now, current):
        want = [(len(q), a) for a, q in pending.items()
                if q and a != current and cache.adapter_tier(a) != 'gpu']
        want.sort(reverse=True)
        return [a for _, a in want]


class PrefetchSimulator(TieredSimulator):
    def __init__(self, *args, aggressive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggressive = aggressive
        self.prefetched = 0

    def do_prefetch(self, budget_ms, current_adapter):
        """Spend the compute window on I/O. Transfers that fit are free."""
        if not hasattr(self.policy, 'choose_prefetch'):
            return
        spent = 0.0
        for a in self.policy.choose_prefetch(self.pending, self.cache,
                                             self.now, current_adapter):
            tier = self.cache.adapter_tier(a)
            cost = SWAP_WARM_MS if tier == 'cpu' else SWAP_COLD_MS
            if spent + cost > budget_ms:
                break
            if self.cache.gpu_free() < ADAPTER_MB:
                if not self.aggressive:
                    break
                self.make_room(ADAPTER_MB, current_adapter, None)
                if self.cache.gpu_free() < ADAPTER_MB:
                    break
            self.cache.place_adapter(a, 'gpu')
            spent += cost
            self.prefetched += 1

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
            compute = 25 + len(batch) * 2 + mx * DECODE_MS_PER_TOKEN
            # the overlap: I/O runs concurrently with this compute window
            self.do_prefetch(compute, a)
            self.now += cost + compute
            for r in batch:
                r.finish(self.now - r.arrival_time); self.finished.append(r)
                self.grow(r, a); self.policy.record(a, r.conversation_id, self.now)
        return self.finished


SEEDS = 24
GPU_MB, CPU_MB, N_ADAPTERS = 800, 400, 80


def workload(seed):
    return make_multi_turn_workload(n_conversations=150, n_adapters=N_ADAPTERS,
                                    turns_per_convo=8, mean_tokens=80,
                                    rate=0.0002, turn_gap=15000, seed=seed)


def trial(seed, prefetch, aggressive=False):
    convos, reqs = workload(seed)
    cache = TieredCache(GPU_MB, CPU_MB)
    if prefetch:
        sim = PrefetchSimulator(reqs, convos, cache, PrefetchPolicy(),
                                aggressive=aggressive)
    else:
        sim = TieredSimulator(reqs, convos, cache, JointTiered())
    done = sim.run()
    return (percentile([r.latency() for r in done], 50), sim.from_disk,
            getattr(sim, 'prefetched', 0))


def stats(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    print(__doc__)
    print(f"GPU {GPU_MB}MB / CPU {CPU_MB}MB / {N_ADAPTERS} adapters, "
          f"{SEEDS} seeds\n")
    print(f"{'seed':>5} {'baseline p50':>13} {'prefetch p50':>13} {'win':>8}"
          f" {'disk loads':>12} {'prefetched':>11}")
    print("-" * 68)
    wins = []
    for s in range(SEEDS):
        b, bd, _ = trial(s, prefetch=False)
        p, pd, pf = trial(s, prefetch=True)
        w = 100 * (b - p) / b
        wins.append(w)
        if s < 8:
            print(f"{s:>5} {b:>13.0f} {p:>13.0f} {w:>7.1f}%"
                  f" {bd:>5} -> {pd:<4} {pf:>11}")
    print(f"{'...':>5}  ({SEEDS} seeds total)")
    m, sd = stats(wins)
    sem = sd / (len(wins) ** 0.5)
    lo, hi = m - 1.96 * sem, m + 1.96 * sem
    print("-" * 68)
    print(f"mean {m:.2f}%   stdev {sd:.2f}%   95% CI [{lo:.2f}%, {hi:.2f}%]")
    print(f"negative seeds: {sum(1 for w in wins if w < 0)}/{len(wins)}")
    print(f"effect is {'REAL (CI excludes zero)' if lo > 0 else 'NOT SIGNIFICANT'}")

    print("\nconservative vs aggressive (does speculative eviction pay?)")
    cw, aw = [], []
    for s in range(12):
        b, _, _ = trial(s, prefetch=False)
        c, _, _ = trial(s, prefetch=True, aggressive=False)
        a, _, _ = trial(s, prefetch=True, aggressive=True)
        cw.append(100 * (b - c) / b); aw.append(100 * (b - a) / b)
    print(f"  conservative: {stats(cw)[0]:>5.2f}% +/- {stats(cw)[1]:.2f}")
    print(f"  aggressive:   {stats(aw)[0]:>5.2f}% +/- {stats(aw)[1]:.2f}")


if __name__ == "__main__":
    main()
