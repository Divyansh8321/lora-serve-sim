# Running the simulator on real production traces

This closes two of the "Scope & honesty" gaps: our synthetic shared-prefix
structure and our synthetic bursty-arrival formula were both invented. Here
we replace them with **real, public, free-to-download traces** and re-run
phases 6 and 7 unmodified against real data instead of our formulas.

## The two datasets

**[Mooncake FAST'25 trace](https://github.com/kvcache-ai/Mooncake)**
(`traces/conversation_trace.jsonl`) — a real anonymized request log released
specifically to study KV-cache sharing. Each request lists `hash_ids`: the
512-token prefix blocks its prompt matches. Matching hash IDs across requests
= shareable cached KV. 12,031 requests, 7,900 real multi-turn conversations.

**[BurstGPT](https://github.com/HPMLL/BurstGPT)** (CC-BY-4.0) — 10M+ real
ChatGPT/GPT-4 request logs from Azure OpenAI, with real arrival timestamps.
We use the arrival-gap structure only (not the token counts, which come from
Mooncake).

Neither trace has a LoRA-adapter field — both are single-model traces. Adapter
identity is inferred from Mooncake's own structure: the first shared prefix
block after the universal root is treated as "which system prompt / adapter
this conversation's session belongs to." See `real_traces.py`'s docstring for
the full mapping and its honest limitations.

## Finding 1 (unplanned): real traffic is far more long-tailed than our synthetic assumption

Our synthetic workload assumes 12 adapters with a Zipf(1.0) popularity skew —
a few adapters dominate, everyone reuses one of a small set. The real trace
does not look like that:

- 7,373 distinct adapters across 7,900 conversations
- only 433 adapters are used by **more than one** conversation
- the single most popular adapter is used by just **7** conversations

This is a genuinely different regime: mostly one-off, unique customer
configurations, with a much smaller "hot set" than we assumed. It also means
most of the raw trace has **no caching decision to make at all** — a memory
policy is irrelevant to an adapter that's used exactly once. We restrict to
conversations whose adapter is used ≥2 times (`--min-adapter-reuse 2`,
default) — the traffic where a caching policy actually matters — which leaves
433 real adapters over 960 conversations, 1,447 requests.

## Finding 2: the phase-6 pathology reproduces at real-trace scale, and blind LRU is worse than synthetic predicted

433 real adapters competing for a 12-slot slab (`pool 2400MB`, matching a
realistic vLLM `--max-loras 12`):

| policy | p50 (ms) | stale KV |
|---|---|---|
| `lru-leaf` | **11,763,584** (~3.3 hours) | 58.3% |
| `ordering` | 6,035 | 41.9% |
| `dep-aware` | 6,035 | 38.0% |

Stale KV (38–58%) lands squarely in ELORA's own reported range (42.4% vLLM,
48.6% ELORA-WOM) — on **real production-style traffic**, not our synthetic
workload. That is the strongest confirmation of the core pathology in this
whole project.

`lru-leaf`'s collapse is far more severe than the synthetic version (which
topped out around 3–40× worse, not ~2000×). 97% of its requests wait multiple
hours. This is real: with hundreds of genuinely competing adapters, blind LRU
doesn't just degrade, it causes near-total system collapse. `ordering` and
`dep-aware` remain indistinguishable from each other here (dep-aware +0.0%) —
consistent with the synthetic phase-6 finding that dependency *scoring* adds
little once you're not doing something as blind as `lru-leaf`.

## Finding 3: on real arrival timing, the swapper's effect shrinks to noise

Phase 7's synthetic bursty workload showed a clear, large effect (the swapper
net-negative at `--hw ours`, net-positive at `--hw elora`). On the real
BurstGPT arrival pattern, at the same real Mooncake conversations:

| policy | `--hw ours` vs react-lru | `--hw elora-aggressive` vs react-lru |
|---|---|---|
| `swap-full` | −0.7% | +0.1% |
| `swap-noprefetch` | −0.7% | +0.1% |

The *direction* still matches (negative at `ours`, ~zero-to-positive at
`elora`) but the **magnitude collapsed to noise** — a fraction of a percent,
versus double-digit percentages on our synthetic bursty formula. Adapter load
counts are nearly identical across every phase-7 policy (~14,800 either way at
`--hw elora`), meaning the swapper isn't getting enough distinguishing
structure from real arrival timing to matter much either way.

**Honest reading:** our synthetic `make_bursty_workload` (hand-tuned 5–10×
rate spikes) produces a much sharper, more favorable-to-testing burst pattern
than BurstGPT's real, messier bursts. The synthetic bursts may have
overstated how much the swapper's behavior matters. This is exactly the kind
of thing "test on real data" is supposed to catch — and it did.

## Finding 4: continuous, not just two-point, hardware-speed sweep

`phase7_cost_swapper.py --sweep-scale` varies decode+prefill together
(the way an actual GPU speedup would) across a fine geometric range, instead
of only testing the two discrete aggressive/conservative points:

At burst ×10, scale 1.0× (=`ours`) down to 0.1× (10× faster than `ours`):
`swap-full` never crosses zero — it gets **worse** (−21% → −104% around 5×
faster) before drifting back toward −75 to −80% at very high speed. At
steady and burst ×5, the same sweep shows the gap **shrinking monotonically**
toward (but never reaching) zero as speed increases.

This sharpens the earlier two-point finding: raw compute speed, swept
continuously, **never flips the sign** at any burst level tried. It confirms
`--swap-out-when full` (a policy choice, not a hardware constant) really is
the causal variable — this isn't an artifact of only checking two discrete
speed points.

## How to reproduce

```bash
# download the traces (one-time, ~55MB total)
mkdir -p traces && cd traces
curl -sL -o mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
curl -sL -o BurstGPT_without_fails_1.csv \
  https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv
cd ..

python real_traces.py                          # sanity-check both loaders
python run_on_real_traces.py                   # phase 6 + 7 on real data, --hw ours
python run_on_real_traces.py --hw elora-aggressive
python phase7_cost_swapper.py --sweep-scale --sweep-range 1.0 0.1 10 --burst 10
```

## Honest limitations of this pass

1. **Two different real systems, stitched together.** BurstGPT's arrivals are
   real Azure OpenAI ChatGPT/GPT-4 traffic; Mooncake's content/prefix
   structure is a real (different) Kimi/Moonshot deployment. Overlaying one
   trace's timing onto another's content is not a reproduction of either
   system's actual joint behavior — it's the best available combination of
   two real signals, not one ground-truth trace.
2. **Adapter identity is inferred, not measured.** Neither trace has a LoRA
   field. The depth-2 shared-prompt-block proxy is defensible but not ground
   truth.
3. **Mooncake's variable-depth prefix chains are collapsed to one number per
   conversation** (the simulator's tree model wants a single `prefix_tokens`
   length; real chains grow turn-by-turn). See `real_traces.py`.
4. **`lru-leaf`'s multi-hour p50 is a genuine simulator output, not a bug** —
   verified by direct inspection (97% of requests affected, TTFT stays normal
   throughout) — but it is also a magnitude no real deployment would tolerate;
   a real system would shed load or trigger alerts long before this point. It
   should be read as "this policy is catastrophically bad at this scale," not
   as a literal production forecast.
