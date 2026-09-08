"""Shared primitives for all phases.

Everything downstream is arithmetic on these. No model ever runs; this is a
discrete-event simulation of resource usage (memory + time), not of text.
"""

import random
from dataclasses import dataclass, replace


# --- calibrated constants (see README for sources) ---
# These are the "OURS" hardware regime: a single ~A10-class GPU (24-40 GB HBM,
# PCIe ~40 GB/s). Phases 5-7 can override them via a HardwareProfile (below) so
# the ELORA comparison can be run at ELORA's H800 regime apples-to-apples.
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


# --------------------------------------------------------------------------
# Hardware profiles -- for apples-to-apples ELORA comparison (phases 5-7)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareProfile:
    """A bundle of the time constants that vary with the serving GPU, plus
    three phase-7 engine-behaviour switches that model what ELORA's runtime
    actually does (CUDA-stream overlapped swaps, evict-on-full not proactively).

    mb_per_token and adapter_mb are MODEL ARCHITECTURE, not hardware -- they are
    held IDENTICAL across every profile. They live here only so a phase that
    already threads adapter size (phase 5's --adapter-mb) has one place to read
    it. The RadixNode.mb() / _resident_mb sites in phases 6-7 stay on the bare
    global since they never change.
    """
    name: str
    mb_per_token: float
    adapter_mb: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    swap_cold_ms: float            # adapter load from disk/object store -- NOT a GPU property, held constant
    swap_warm_ms: float            # adapter load from CPU RAM over PCIe -- scales with PCIe
    pcie_ms_per_mb: float
    # phase-7 engine-behaviour knobs; --hw sets them as a bundle, per-switch
    # flags override individually for one-at-a-time attribution.
    swap_mode: str = "sync"           # "sync" | "async" (transfer overlaps inference)
    prefetch_cost: str = "charged"    # "charged" | "overlapped"
    swap_out_when: str = "highwater"  # "highwater" | "full"


OURS = HardwareProfile(
    name="ours",
    mb_per_token=MB_PER_TOKEN, adapter_mb=ADAPTER_MB,
    prefill_ms_per_token=PREFILL_MS_PER_TOKEN, decode_ms_per_token=DECODE_MS_PER_TOKEN,
    swap_cold_ms=SWAP_COLD_MS, swap_warm_ms=SWAP_WARM_MS, pcie_ms_per_mb=PCIE_MS_PER_MB,
)

# ELORA: NVIDIA H800, 80 GB HBM, PCIe 5.0 @ 128 GB/s (Table II). HBM BW ~3.35
# TB/s vs our implied ~600 GB/s. decode/prefill scaling is a BAND -- we report
# both an aggressive (spec-sheet HBM/FLOP ratio) and conservative estimate,
# since it is the single biggest lever on whether the swapper crosses zero.
_ELORA_COMMON = dict(
    name="elora", mb_per_token=MB_PER_TOKEN, adapter_mb=ADAPTER_MB,
    swap_cold_ms=SWAP_COLD_MS,                       # disk/S3 latency, unchanged
    swap_warm_ms=SWAP_WARM_MS * (0.008 / 0.105),     # PCIe-bound -> scales with PCIe
    pcie_ms_per_mb=0.008,                            # PCIe 5.0 @ 128 GB/s + dispatch
    swap_mode="async", prefetch_cost="overlapped", swap_out_when="full",
)
ELORA_AGGRESSIVE = HardwareProfile(
    **_ELORA_COMMON,
    prefill_ms_per_token=PREFILL_MS_PER_TOKEN * 0.13,
    decode_ms_per_token=DECODE_MS_PER_TOKEN * 0.18,
)
ELORA_CONSERVATIVE = replace(
    ELORA_AGGRESSIVE,
    prefill_ms_per_token=PREFILL_MS_PER_TOKEN * 0.30,
    decode_ms_per_token=DECODE_MS_PER_TOKEN * 0.35,
)

PROFILES = {
    "ours": OURS,
    "elora": ELORA_AGGRESSIVE,
    "elora-aggressive": ELORA_AGGRESSIVE,
    "elora-conservative": ELORA_CONSERVATIVE,
}


def get_profile(x):
    """Accept a HardwareProfile, or a name string from PROFILES."""
    if isinstance(x, HardwareProfile):
        return x
    return PROFILES[x]


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


def make_prefix_sharing_workload(n_conversations=40, n_adapters=12,
                                 turns_per_convo=5, skew=1.0, rate=0.0006,
                                 mean_tokens=40, turn_gap=6000.0,
                                 shared_prefix_tokens=0, shared_prefix_groups=3,
                                 seed=0):
    """Phase 6: like make_multi_turn_workload, but conversations are split into
    `shared_prefix_groups` groups and every conversation in a group shares the
    same first `shared_prefix_tokens` tokens -- a "system prompt" that a radix
    prefix tree can match and reuse across users.

    The prefix belongs to a (group, adapter) pair: two conversations only share
    a prefix if they are in the same group AND use the same adapter, because a
    LoRA rewrites the KV. This mirrors ELORA's tree, whose top layer is LoRAs
    and whose prefix nodes hang inside a LoRA's subtree.

    Adds to each Conversation:
      .prefix_group   : int
      .prefix_tokens  : int   (== shared_prefix_tokens, or 0)
      .prefix_key     : (adapter_id, prefix_group) or None
    Adds to each Request:
      .is_first_turn  : bool  (only the first turn pays to build the prefix)

    shared_prefix_tokens=0 reduces EXACTLY to make_multi_turn_workload's
    request/timing structure (same seed -> same turns), so phase 6 at 0 is a
    controlled baseline against phases 2-5.
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
        convo = Conversation(adapter, cid)
        convo.prefix_group = cid % shared_prefix_groups
        convo.prefix_tokens = shared_prefix_tokens
        convo.prefix_key = ((adapter, convo.prefix_group)
                            if shared_prefix_tokens > 0 else None)
        conversations.append(convo)
        turn_time = t
        for turn_idx in range(turns_per_convo):
            tokens = max(10, int(rng.gauss(mean_tokens, mean_tokens * 0.3)))
            r = Request(rid, cid, tokens, turn_time)
            r.is_first_turn = (turn_idx == 0)
            requests.append(r)
            rid += 1
            turn_time += rng.expovariate(1.0 / turn_gap)
    return conversations, requests


def make_bursty_workload(n_conversations=60, n_adapters=12, turns_per_convo=5,
                         skew=1.0, base_rate=0.0006, mean_tokens=40,
                         turn_gap=6000.0, shared_prefix_tokens=0,
                         shared_prefix_groups=3,
                         burst_factor=5.0, burst_period=40000.0,
                         burst_duty=0.25, popularity_drift_period=60000.0,
                         seed=0):
    """Phase 7: bursty arrivals + drifting adapter popularity.

    ELORA drives arrivals from the Microsoft Azure Function trace -- bursty,
    heavy-tailed, with the hot set of LoRAs shifting over time. A steady
    Poisson + fixed Zipf (phases 2-6) never stresses the two things ELORA's
    cost-model swapper is FOR: prefetching during lulls, and re-scoring as the
    hot set moves. This generator adds both, minimally.

    Arrivals: an inhomogeneous Poisson process. The instantaneous rate is
      base_rate * burst_factor   during a burst  (fraction burst_duty of each
                                                  burst_period window)
      base_rate                  otherwise
    Conversation START times are drawn from this; turns within a conversation
    still use turn_gap think-time.

    Popularity drift: every popularity_drift_period ms the Zipf ranking of
    adapters is re-shuffled, so an adapter that was rank-1 (hottest) can drop
    to rank-8. A conversation's adapter is sampled from whatever ranking is
    current at its start time.

    shared_prefix_tokens works as in make_prefix_sharing_workload.

    With burst_factor=1.0 and popularity_drift_period=inf this reduces to a
    prefix-sharing workload with Poisson arrivals (not byte-identical to
    make_prefix_sharing_workload -- different n_conversations default -- but
    the same structure).
    """
    rng = random.Random(seed)
    adapter_ids = list(range(n_adapters))

    def ranking_at(t):
        """Zipf weights over adapter_ids, re-shuffled each drift period."""
        epoch = int(t // popularity_drift_period) if popularity_drift_period > 0 else 0
        r = random.Random(seed * 100003 + epoch)
        order = adapter_ids[:]
        r.shuffle(order)
        weights = [1.0 / (k ** skew) for k in range(1, n_adapters + 1)]
        return order, weights

    def rate_at(t):
        if burst_period <= 0 or burst_factor <= 1.0:
            return base_rate
        phase = (t % burst_period) / burst_period
        return base_rate * burst_factor if phase < burst_duty else base_rate

    conversations = []
    requests = []
    rid = 0
    t = 0.0
    for cid in range(n_conversations):
        # thin an inhomogeneous Poisson process: step with the max rate,
        # accept a point with prob rate_at(t)/max_rate
        max_rate = base_rate * max(1.0, burst_factor)
        while True:
            t += rng.expovariate(max_rate)
            if rng.random() <= rate_at(t) / max_rate:
                break
        order, weights = ranking_at(t)
        adapter = rng.choices(order, weights=weights, k=1)[0]
        convo = Conversation(adapter, cid)
        convo.prefix_group = cid % shared_prefix_groups
        convo.prefix_tokens = shared_prefix_tokens
        convo.prefix_key = ((adapter, convo.prefix_group)
                            if shared_prefix_tokens > 0 else None)
        conversations.append(convo)
        turn_time = t
        for turn_idx in range(turns_per_convo):
            tokens = max(10, int(rng.gauss(mean_tokens, mean_tokens * 0.3)))
            r = Request(rid, cid, tokens, turn_time)
            r.is_first_turn = (turn_idx == 0)
            requests.append(r)
            rid += 1
            turn_time += rng.expovariate(1.0 / turn_gap)
    return conversations, requests
