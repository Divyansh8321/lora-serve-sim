# Roadmap: closing the fidelity gap to production LoRA serving

Phases 1–4 answer "does coordinating adapter + KV eviction pay?" under a
deliberately simple execution model. The catch: **ELORA/FastLibra — the work we
are comparing against — run entirely under continuous batching, with prefix
caching, and a periodic cost-model swapper.** Our batch-at-a-time model with no
prefix sharing is structurally different enough that "we reproduce their
pathology" needs an asterisk.

This roadmap removes the asterisks, one at a time. Each phase is a self-contained
simulator that reuses `core.py` primitives and the existing workload generator,
so results stay comparable to phases 1–4.

The goal is **not** to reimplement vLLM. It is to model the *mechanism* each
system uses precisely enough that (a) our findings are honest and (b) the code
demonstrates we understand what these systems actually do.

---

## Phase 5 — Continuous batching

### What real systems do (plain words)

There is **one long-lived batch** that the GPU advances one token per step.
Every step:

1. every sequence in the batch emits exactly **one** token (one decode pass over
   the whole batch);
2. any sequence that just hit its stop condition is **removed immediately** — its
   KV blocks free that instant;
3. any waiting request is **spliced in** — its prompt is prefilled and it joins
   the per-token loop — as long as KV blocks are available.

Consequences that matter for *this* project:

- **An adapter is pinned while any of its sequences is mid-decode.** You cannot
  evict LoRA-A's weights until every A-sequence in the batch has finished. So
  "evict adapter" is not a free choice per scheduling tick — it is gated by
  in-flight work.
- **Stale KV becomes a timing phenomenon.** A conversation goes quiet (all its
  turns done, adapter no longer pinned) → adapter gets bumped for someone active
  → turn N+1 arrives after the think-time gap → adapter cold-reloads, and its KV
  may or may not have survived. Our workload already has the gap structure
  (`turn_gap=6000`); the batch-at-a-time model just can't express "idle but
  recently-resident adapter".
- **Prefill and decode share the step.** vLLM's default is chunked prefill mixed
  into the decode batch. A big recompute (stale KV rebuilt) steals step time
  from everyone → TPOT rises for unrelated requests. This is exactly ELORA's
  stated reason for their TPOT win.

### Model design (`phase5_continuous_batching.py`)

New core concept: a **step loop** instead of a scheduling loop.

```
State per step:
  now                     : ms, advances by one step_duration each iteration
  batch                   : list[SeqState]   -- sequences currently decoding
  waiting                 : deque[Request]   -- arrived, not yet admitted
  cache                   : UnifiedCache or SeparatePools (reuse from pools.py)

SeqState:
  request, conversation_id, adapter_id
  remaining_tokens        : decremented 1 per step
  phase                   : 'prefill' | 'decode'
  prefill_tokens_left     : for chunked prefill

Each step:
  1. admit_arrivals()            -- move due Requests into `waiting`
  2. schedule():
       - for each waiting request, in arrival order:
           ensure adapter resident (may evict -- but NOT pinned adapters)
           ensure KV space for its prompt (may evict KV -- policy decides)
           if it fits: create SeqState(phase='prefill'), add to batch
           else: leave in waiting (backpressure)
  3. compute step_duration:
       decode_cost   = DECODE_MS_PER_TOKEN            (one pass, batch-parallel)
       prefill_cost  = sum(chunk_tokens) * PREFILL_MS_PER_TOKEN   (chunked)
       + FIXED_BATCH_MS + len(batch) * PER_REQUEST_MS
       + swap costs incurred this step
       step_duration = decode_cost + prefill_cost
  4. advance:
       now += step_duration
       for seq in batch:
           if seq.phase == 'prefill': consume a prefill chunk; maybe -> 'decode'
           else: seq.remaining_tokens -= 1
           grow that conversation's KV by 1 token
           if seq done: record latency; free its KV-pin; remove from batch
  5. sample stale% (same definition as phases 2-4)
```

**Adapter pinning** — the one genuinely new eviction constraint:

```python
def pinned_adapters(batch):
    return {s.adapter_id for s in batch}          # anyone mid-flight

# in the eviction policy:
candidates = [a for a in cache.gpu_residents() if a not in pinned_adapters(batch)]
```

**`--max-loras` now means what it means in vLLM:** the max number of *distinct
adapters with live sequences in the batch at once*. Enforce it in `schedule()`:
do not admit a request whose adapter would be the (max_loras+1)-th distinct
adapter in `batch`.

### What to measure / expect

| knob | prediction | reproduces |
|---|---|---|
| shrink `--max-loras` | admission stalls when a 5th adapter wants in; p50 + TPOT rise; stale KV climbs during think-time gaps | phase-2 pathology, now timing-driven |
| add CPU adapter tier | reload cost drops, stall frequency unchanged | phase-2b "cheaper not rarer" |
| unified vs separate pool (reuse `pools.py` policies) | unified lets a big idle conversation's KV be traded for an adapter slot mid-stream | phase-4 headline, under continuous batching |

### Validation hook

`validation/workload.py` already replays the real arrival schedule against real
vLLM (which *is* continuous-batched). Phase 5's p50/TPOT curve is what
`validation/compare.py` should be checked against — not phase 2's. Add a
`--model phase5` switch to `compare.py`.

### Done when

- `python phase5_continuous_batching.py` prints a `--max-loras` sweep with
  p50 / p95 / TPOT / stale-KV, mean over 5 seeds
- separate-vs-unified comparison reuses `pools.py` policies unchanged
- a short docstring finding, in the phase-1..4 house style
- README limitations table: continuous batching moves from "not modelled" to
  "phase 5"

---

## Phase 6 — RadixAttention prefix tree

### What real systems do

SGLang's RadixAttention (and ELORA's dependency manager, built on it) store KV
blocks in a **radix tree keyed by token prefix**. A new request walks the tree
from the root matching its prompt tokens; every node on the matched path is KV it
can **reuse for free**. Shared system prompts, few-shot preambles, and tool
definitions — identical across users — become one shared path near the root.

ELORA's twist: the tree's **top layer is LoRAs**, then KV prefixes hang in each
LoRA's subtree. Matching is: find LoRA node → DFS its subtree for the longest
KV-prefix match → reuse. Eviction candidates are **leaf nodes only** (you can't
evict an interior node someone is sharing).

### Model design (`phase6_radix_prefix.py`)

Extend the workload first:

```python
# core.py addition
def make_prefix_sharing_workload(..., shared_prefix_tokens=200,
                                 shared_prefix_groups=3, ...):
    # each conversation is assigned to one of `shared_prefix_groups`;
    # all conversations in a group share the first `shared_prefix_tokens`
    # tokens (a "system prompt"). Rest as before.
```

Then the tree:

```
RadixNode:
  token_span      : the tokens this node covers
  children        : dict[token_key -> RadixNode]
  kv_size_mb
  owner_adapter   : which LoRA subtree this lives in
  last_used
  ref_count       : how many active sequences share this path

match(prompt_tokens, adapter_id):
    node = roots[adapter_id]
    matched_mb = 0
    walk children by longest token-prefix match, summing kv_size_mb
    return matched_mb, leaf_node        # matched_mb is reused free

insert(new_tokens, leaf_node): extend tree below last matched node

evictable(): all leaf nodes with ref_count == 0, ranked by policy
```

Recompute cost for a request becomes `(prompt_tokens - matched_tokens) *
PREFILL_MS_PER_TOKEN` instead of the full prompt.

### Policies to compare (the actual research question)

| policy | eviction rule |
|---|---|
| `LRULeaf` | evict LRU leaf, ignore which LoRA it's under |
| `OrderingOnly` | phase-4 style: evict biggest KV subtree before any adapter |
| `DependencyAware` | ELORA-style: never evict a LoRA with a non-empty, recently-used KV subtree; rank leaves by (is-stale, size, LRU) |

**Hypothesis under test:** phase 4 found `OrderingOnly ≈ DependencyAware` with
*no* prefix sharing. Prediction: as `shared_prefix_tokens` grows, the gap opens
— because now evicting the wrong LoRA strands a *shared* prefix that many
sequences wanted, and "biggest first" no longer captures that. If the gap opens,
ELORA is right and our phase-4 workload was too easy. If it doesn't, ELORA's
tree is over-engineered for the LoRA-dependency case. **Either result is worth
reporting.**

### Done when

- prefix-sharing sweep (`shared_prefix_tokens` = 0 / 100 / 400 / 1000)
- the three policies compared at each point, mean over 5 seeds
- a clear statement: does dependency-awareness beat ordering, and from what
  sharing level
- README: phase 6 result feeds back into the phase-4 "Scope & honesty" caveat

---

## Phase 7 — ELORA-style cost-model swapper

### What real systems do

ELORA doesn't evict reactively on "cache full". Every **100ms** a background
swapper re-scores every tree node:

```
Eval_i  ~  f( swap_cost_i , visit_frequency_i , (1 - sigmoid(t_i)) , enough_loras_term )
```

- **swap_cost_i** — bytes to move × PCIe cost (bigger = keep, evicting is
  expensive)
- **visit_frequency_i** — how often this node was hit recently (hot = keep)
- **`1 - sigmoid(t_i)`** — a soft-LRU recency term
- **enough_loras_term** — a floor so the swapper doesn't evict so many LoRAs
  that every incoming request cold-starts

When memory is **idle**, it *prefetches* (swaps in likely-next LoRAs/KV). When
**busy**, it swaps out lowest-`Eval_i` leaves first.

ELORA's ablations: replace the cost model with plain LRU (ELORA-WOS) → **1.42×
worse TTFT**. Drop individual terms (WOL/WOC/WOV/WOU) → 1.19–1.25× worse each.

### Model design (`phase7_cost_swapper.py`)

Reuse phase 5's step loop + phase 6's tree. Add:

```python
class CostModelSwapper:
    def __init__(self, interval_ms=100, w_cost=..., w_freq=..., w_lru=..., lora_floor=...):
        ...
    def maybe_run(self, now, cache, batch):
        if now - self.last_run < self.interval_ms: return
        self.last_run = now
        for node in cache.all_leaves():
            node.eval = (self.w_cost * node.swap_cost_mb
                         + self.w_freq * node.recent_hits
                         + self.w_lru  * (1 - sigmoid((now - node.last_used)/SCALE))
                         + self.lora_floor_term(cache))
        if cache.pressure() > HIGH:
            evict lowest-eval leaves until under LOW watermark
        elif cache.pressure() < LOW and idle:
            prefetch highest-value non-resident LoRAs/KV
```

### Comparisons

| variant | matches ELORA's |
|---|---|
| full cost model | ELORA |
| cost model → LRU | ELORA-WOS (expect ~1.4× worse) |
| drop `w_cost` term | ELORA-WOC |
| drop `w_freq` term | ELORA-WOV |
| no periodic run, react on full only | phases 4–6 baseline |

### Done when

- the variants compared on the phase-6 prefix-sharing workload
- a statement on whether the cost model reproduces ELORA's ~1.4× advantage over
  LRU *in our setting*, and which term carries it
- if it doesn't reproduce: say so, and name the workload feature we're missing
  (most likely the Azure-trace burstiness — see below)

### Status: DONE — `phase7_cost_swapper.py`

Result: the timer-driven swapper is **net-negative at our engine model**
(sync swap, charged prefetch, proactive evict-at-92%): −5% to −22% p50, worse
under bursts. **At ELORA's engine model** (`--hw elora`: async CUDA-stream
swap, overlapped prefetch, evict-on-full) it **flips to +0.4% to +2.9%**. The
`--sweep` attribution isolates the single responsible change: `--swap-out-when
full` — a *policy* choice, not a hardware constant. `--sweep-const` confirms no
hardware constant alone flips it. `react-dep` (phase-6 eviction, no timer) is
the consistent winner at both engines.

### Phase 7b — hardware-profile sweep (DONE, folded into phase 7)

`core.HardwareProfile` + `OURS` / `ELORA_AGGRESSIVE` / `ELORA_CONSERVATIVE`.
Phases 5–7 take `--hw`; phase 7 adds `--swap-mode`, `--prefetch-cost`,
`--swap-out-when` (each a standalone switch), `--sweep` (switch attribution
grid), `--sweep-const` (single-constant crossover), and `sweeps/*.py`
standalone artifacts. Phase 5: pathology is HW-robust (fit-ratio effect).
Phase 6: dependency-scoring's value *shrinks* at faster HW.

---

## Cross-cutting: workload realism

ELORA drives arrivals from the **Microsoft Azure Function trace** — bursty,
heavy-tailed, with LoRA popularity drifting over time. Our
`make_multi_turn_workload` uses a fixed Zipf and Poisson arrivals. Two of the
phases above ("does the cost model help", "does dependency-awareness help") may
*only* show an effect under burstiness, because that's when static partitioning
and reactive LRU fall behind.

Cheap addition, worth doing before phase 7:

```python
def make_bursty_workload(..., burst_factor=4, drift=True, ...):
    # Poisson base rate with periodic bursts; optionally rotate the Zipf
    # ranking every T ms so the "hot set" of adapters moves.
```

Run phases 3–7 on both the steady and bursty workloads. If a mechanism only pays
under burstiness, that itself is the finding — and it matches ELORA's framing
("production traces show the distributions are dynamic").

---

## Suggested order & checkpoints

1. **Phase 5** (continuous batching) — biggest single fidelity gain; everything
   downstream needs the step loop. Checkpoint: phase-2 sweep reproduced with
   timing-driven staleness, validation curve re-pointed at phase 5.
2. **Bursty workload generator** — small, unlocks the phase 7 comparison.
3. **Phase 6** (prefix tree) — resolves the open question in the phase-4 caveat.
   Checkpoint: a yes/no on "does dependency-awareness beat ordering under
   sharing".
4. **Phase 7** (cost swapper + `--hw` profiles) — DONE. The swapper is
   net-negative at our engine, net-positive at ELORA's; `--swap-out-when full`
   (a policy choice) carries the flip, not any hardware constant.
5. **GPU validation run** — once phase 5 exists, run `validation/run_sweep.sh`
   and compare against phase 5, not phase 2. Still pending real hardware.

Each checkpoint is a commit + a docstring finding in the house style. No phase
merges into a claim in the README until its checkpoint statement is written.
