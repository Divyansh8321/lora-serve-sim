"""PHASE 1 -- Adapter caching only. No KV cache modelled.

WORLD: one base model, many LoRA adapters, a GPU that holds only a few at once.
       Requests are independent (no conversation state).

QUESTION: does a smart adapter-caching policy beat a naive one?

FINDING (measured over 5 seeds): adapter caching alone is worth surprisingly
  little.
  - Cold adapters (from disk/S3, ~300ms): ~3-6% p50, ~4-8% p95. Real but
    modest, and noisy.
  - Warm adapters (from CPU RAM, ~30ms): ~0%. Nothing.
  WHY: the smart policy does cut swaps (roughly 480 -> 395 on a 1200-request
  run), but swapping is only ~20% of wall-clock time, so a 20% reduction in
  swaps buys a few percent overall. Decode dominates.

  THIS IS THE MOTIVATION FOR PHASE 2. If adapters were the whole story, this
  problem would not be worth solving. The memory that actually hurts is the
  KV cache, which phase 1 does not model at all.

Run:  python phase1_adapter_caching.py
"""

from collections import deque
from core import (ADAPTER_MB, DECODE_MS_PER_TOKEN, SWAP_COLD_MS, SWAP_WARM_MS,
                        make_single_turn_workload, summarize)


class AdapterCache:
    """Holds resident adapters. Mechanism only -- it never decides who to evict."""

    def __init__(self, max_slots):
        self.max_slots = max_slots
        self.adapters = []

    def check_is_full(self):
        return len(self.adapters) == self.max_slots

    def is_resident(self, adapter_id):
        return adapter_id in self.adapters

    def add_adapter(self, adapter_id):
        if adapter_id in self.adapters:
            return
        if self.check_is_full():
            raise RuntimeError("Cache full - evict before adding")
        self.adapters.append(adapter_id)

    def delete_adapter(self, adapter_id):
        if adapter_id not in self.adapters:
            raise RuntimeError("Adapter not resident - cannot evict")
        self.adapters.remove(adapter_id)


class CostModel:
    def __init__(self, swap_cost=SWAP_COLD_MS):
        self.fixed_batch_cost = 25
        self.request_cost = 2
        self.output_cost = DECODE_MS_PER_TOKEN
        self.swap_cost = swap_cost

    def calculate(self, batch_size, max_output_tokens, requires_swap):
        return (self.fixed_batch_cost
                + batch_size * self.request_cost
                + max_output_tokens * self.output_cost
                + requires_swap * self.swap_cost)


class DumbPolicy:
    """Serves the biggest backlog, ignoring the cache. Evicts arbitrarily."""
    name = "naive"

    def choose_next(self, pending, cache, now):
        waiting = [a for a, q in pending.items() if q]
        return max(waiting, key=lambda a: len(pending[a]))

    def choose_evict(self, cache, pending, now):
        return cache.adapters[0]

    def record(self, adapter_id, now):
        pass


class HotSetAffinity:
    """Three cooperating ideas: starvation guard, prefer-resident, LRU evict."""
    name = "hot-set"

    def __init__(self, wait_threshold=30000):
        self.memory = {}
        self.wait_threshold = wait_threshold

    def choose_next(self, pending, cache, now):
        waiting = [a for a, q in pending.items() if q]
        # each queue is arrival-sorted, so [0] is that adapter's oldest waiter
        most_starved = max(waiting, key=lambda a: now - pending[a][0].arrival_time)
        if now - pending[most_starved][0].arrival_time > self.wait_threshold:
            return most_starved
        resident = [a for a in waiting if cache.is_resident(a)]
        if resident:
            return max(resident, key=lambda a: len(pending[a]))
        return max(waiting, key=lambda a: len(pending[a]))

    def choose_evict(self, cache, pending, now):
        return min(cache.adapters, key=lambda a: self.memory.get(a, -1))

    def record(self, adapter_id, now):
        self.memory[adapter_id] = now


class Simulator:
    def __init__(self, requests, cache, cost_model, policy, max_batch=16):
        self.cache = cache
        self.cost_model = cost_model
        self.policy = policy
        self.max_batch = max_batch
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.now = 0.0
        self.pending = {}
        self.finished = []
        self.swaps = 0

    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            r = self.incoming.popleft()
            self.pending.setdefault(r.adapter_id, deque()).append(r)

    def pending_is_empty(self):
        return not any(self.pending.values())

    def run(self):
        while self.incoming or not self.pending_is_empty():
            self.admit_arrivals()
            if self.pending_is_empty():
                if not self.incoming:
                    break
                self.now = self.incoming[0].arrival_time
                continue

            adapter = self.policy.choose_next(self.pending, self.cache, self.now)
            swapped = not self.cache.is_resident(adapter)
            if swapped:
                if self.cache.check_is_full():
                    victim = self.policy.choose_evict(self.cache, self.pending, self.now)
                    self.cache.delete_adapter(victim)
                self.cache.add_adapter(adapter)
                self.swaps += 1

            q = self.pending[adapter]
            batch = [q.popleft() for _ in range(min(self.max_batch, len(q)))]
            max_tokens = max(r.output_tokens for r in batch)
            duration = self.cost_model.calculate(len(batch), max_tokens, swapped)
            self.now += duration
            for r in batch:
                r.finish(self.now - r.arrival_time)
                self.finished.append(r)
            self.policy.record(adapter, self.now)
        return self.finished


def trial(skew, swap_cost, slots=6, n_adapters=30, PolicyCls=HotSetAffinity, seed=0):
    reqs = make_single_turn_workload(n_requests=1200, n_adapters=n_adapters,
                                     skew=skew, rate=0.0012, seed=seed)
    sim = Simulator(reqs, AdapterCache(slots), CostModel(swap_cost), PolicyCls())
    m = summarize(sim.run())
    return m, sim.swaps


SEEDS = 5


def main():
    print(__doc__)
    for label, swap in [("COLD adapters (from disk, 300ms)", SWAP_COLD_MS),
                        ("WARM adapters (from CPU RAM, 30ms)", SWAP_WARM_MS)]:
        print(f"\n{'='*74}\n{label}   (mean +/- stdev over {SEEDS} seeds)\n{'='*74}")
        print(f"{'skew':>6} {'p50 win':>20} {'p95 win':>20} {'swaps naive->hot':>20}")
        print("-" * 74)
        for skew in [0.0, 0.4, 0.8, 1.2, 1.6]:
            w50, w95, swn, swh = [], [], [], []
            for s in range(SEEDS):
                mn, sn = trial(skew, swap, PolicyCls=DumbPolicy, seed=s)
                mh, sh = trial(skew, swap, PolicyCls=HotSetAffinity, seed=s)
                w50.append(100 * (mn['p50'] - mh['p50']) / mn['p50'])
                w95.append(100 * (mn['p95'] - mh['p95']) / mn['p95'])
                swn.append(sn); swh.append(sh)
            m50 = sum(w50)/SEEDS; d50 = (sum((x-m50)**2 for x in w50)/SEEDS)**0.5
            m95 = sum(w95)/SEEDS; d95 = (sum((x-m95)**2 for x in w95)/SEEDS)**0.5
            print(f"{skew:>6.1f} {m50:>13.1f}% +/-{d50:>4.1f} {m95:>13.1f}% +/-{d95:>4.1f}"
                  f" {sum(swn)//SEEDS:>9} -> {sum(swh)//SEEDS:<7}")


if __name__ == "__main__":
    main()
