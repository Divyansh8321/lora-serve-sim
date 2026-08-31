"""Shared primitives for all phases.

Everything downstream is arithmetic on these. No model ever runs; this is a
discrete-event simulation of resource usage (memory + time), not of text.
"""

import random


# --- calibrated constants (see README for sources) ---
MB_PER_TOKEN = 0.125      # 8B model, GQA, fp16: 2*32layers*8heads*128dim*2bytes
# ADAPTER_MB: rank-32 LoRA on Llama-3.1-8B, all 7 linear targets
# (q,k,v,o,gate,up,down). Per-principles: 32 layers * [4*(4096*32+32*4096) +
# 2*(4096*32+32*14336) + (14336*32+32*4096)] = 90.2M params * 2 bytes fp16
# = 180 MB. ELORA (HPCA 2026) states "the ranks of LoRAs in our evaluations
# are either 32 or 64", so this matches the architecture we compare against.
# Earlier drafts used 20 MB (rank-8, q/v-only) -- a narrow-adapter regime;
# phase 5's --adapter-mb sweeps 20/90/180 to show the pathology is size-robust.
ADAPTER_MB = 180
PREFILL_MS_PER_TOKEN = 0.15   # compute-bound, parallel across the prompt
DECODE_MS_PER_TOKEN = 12      # bandwidth-bound, one full pass per token
SWAP_COLD_MS = 300        # adapter load from disk / object storage
SWAP_WARM_MS = 30         # adapter load from CPU RAM
PCIE_MS_PER_MB = 0.105    # 40GB/s bandwidth + ~10us dispatch per 128KB chunk


class Request:
    """One turn. Knows its own facts, guards its own transitions."""

    def __init__(self, request_id, conversation_id, output_tokens, arrival_time):
        self.request_id = request_id
        self.conversation_id = conversation_id
        self.arrival_time = arrival_time
        self.output_tokens = output_tokens
        self.end_time = None

    def latency(self):
        if self.end_time is not None:
            return self.end_time - self.arrival_time
        else:
            raise RuntimeError("Request not finished yet")

    def finish(self, duration):
        if self.end_time is not None:
            raise RuntimeError("Request was already finished")
        else:
            self.end_time = self.arrival_time + duration


class Conversation:
    """One thread across many turns. Owns its size; residency lives in the cache."""

    def __init__(self, adapter_id, conversation_id):
        self.conversation_id = conversation_id
        self.adapter_id = adapter_id
        self.kv_cache_size = 0

    def grow_cache(self, growth_size):
        self.kv_cache_size += growth_size


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def summarize(finished):
    lats = [r.latency() for r in finished]
    return {
        "n": len(finished),
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "p99": percentile(lats, 99),
    }


def make_single_turn_workload(n_requests=1200, n_adapters=30, skew=1.0,
                              rate=0.01, mean_tokens=48, seed=0):
    """Phase 1: independent requests, no conversation state.

    skew is the Zipf exponent over adapter popularity:
      0.0 = uniform (no policy can help)
      1.6 = heavily concentrated (a few adapters dominate)
    """
    rng = random.Random(seed)
    weights = [1.0 / (r ** skew) for r in range(1, n_adapters + 1)]
    adapter_ids = list(range(n_adapters))
    rng.shuffle(adapter_ids)

    requests = []
    t = 0.0
    for i in range(n_requests):
        t += rng.expovariate(rate)
        adapter = rng.choices(adapter_ids, weights=weights, k=1)[0]
        tokens = max(1, int(rng.gauss(mean_tokens, mean_tokens * 0.3)))
        r = Request(i, conversation_id=None, output_tokens=tokens, arrival_time=t)
        r.adapter_id = adapter          # phase 1 only: request carries adapter directly
        requests.append(r)
    return requests


def make_multi_turn_workload(n_conversations=40, n_adapters=12, turns_per_convo=5,
                             skew=1.0, rate=0.0006, mean_tokens=40,
                             turn_gap=6000.0, seed=0):
    """Phases 2-4: conversations that emit several turns over time.

    rate     : conversation starts per ms
    turn_gap : mean think-time between turns of one conversation (ms)
    """
    rng = random.Random(seed)
    weights = [1.0 / (r ** skew) for r in range(1, n_adapters + 1)]
    adapter_ids = list(range(n_adapters))
    rng.shuffle(adapter_ids)

    conversations = []
    requests = []
    rid = 0
    t = 0.0
    for cid in range(n_conversations):
        t += rng.expovariate(rate)
        adapter = rng.choices(adapter_ids, weights=weights, k=1)[0]
        conversations.append(Conversation(adapter, cid))
        turn_time = t
        for _ in range(turns_per_convo):
            tokens = max(10, int(rng.gauss(mean_tokens, mean_tokens * 0.3)))
            requests.append(Request(rid, cid, tokens, turn_time))
            rid += 1
            turn_time += rng.expovariate(1.0 / turn_gap)
    return conversations, requests
