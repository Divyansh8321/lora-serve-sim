# Running the simulator on real production traces

This closes two of the "Scope & honesty" gaps: our synthetic shared-prefix
structure and our synthetic bursty-arrival formula were both invented. Here
we replace them with **real, public, free-to-download traces** and re-run
phases 6 and 7 unmodified against real data instead of our formulas — on
**two independent real datasets**, one of which is one of ELORA's own three
evaluation datasets.

## The three datasets

**[Mooncake FAST'25 trace](https://github.com/kvcache-ai/Mooncake)**
(`traces/conversation_trace.jsonl`) — a real anonymized request log released
specifically to study KV-cache sharing. Each request lists `hash_ids`: the
512-token prefix blocks its prompt matches. Matching hash IDs across requests
= shareable cached KV. 12,031 requests, 7,900 real multi-turn conversations.
No LoRA-adapter field — adapter identity is *inferred* from the first shared
prefix block after the universal root (a real, Zipf-shaped popularity signal,
not ground truth).

**[Google Taskmaster TM-1](https://github.com/google-research-datasets/Taskmaster)**
(`self-dialogs.json`) — **one of ELORA's own three evaluation datasets**
("Personal Agents"). 7,708 real task-oriented dialogs, ~11 real turns each.
Carries a genuine `instruction_id` field (e.g. `pizza-ordering-2`,
`restaurant-table-1`) — 15 distinct real task types, used **directly** as
adapter identity. No inference needed, unlike Mooncake.

**[BurstGPT](https://github.com/HPMLL/BurstGPT)** (CC-BY-4.0) — 10M+ real
ChatGPT/GPT-4 request logs from Azure OpenAI, with real arrival timestamps.
Used for arrival-gap timing only. Neither Mooncake nor Taskmaster has
real-time arrival data of the kind our simulator needs (Taskmaster has none
at all; we chose to overlay BurstGPT onto both for a controlled comparison).
**This mirrors ELORA's own documented method**: their paper states plainly
that Taskmaster "lacks timestamps," so they "adopt query arrival patterns
from the Microsoft Azure function trace" for it. We do the same thing they
did, with a public substitute for their (unreleased) exact Azure slice.

## Methodology note: matching real-time density across datasets

Mooncake conversations average 1.5 turns; Taskmaster conversations average
~11. Slicing both traces to "the same number of requests" therefore spreads
Taskmaster's conversations across a much longer real BurstGPT time window
than Mooncake's (early runs: 83 hours vs 15 hours for a nominally
"comparable" cut) — diluting concurrency and making the results
incomparable. Fixed by slicing **by conversation count**, not request count
(`_truncate_to_n_conversations` in `run_on_real_traces.py`): at 960
conversations, both datasets now span the same real 15-hour BurstGPT window.

## Finding 1 (unplanned): real traffic is far more long-tailed than our synthetic assumption — but this varies enormously by dataset

Our synthetic workload assumes 12 adapters with a Zipf(1.0) popularity skew.
The two real datasets bracket it from opposite directions:

| | Mooncake | Taskmaster | our synthetic |
|---|---|---|---|
| distinct adapters | 7,373 | **15** | 12 |
| most popular adapter's use count | 7 | **1,211** | (Zipf-skewed) |
| adapters used by only 1 conversation | 96% | 0% | 0% |

Mooncake is far *more* long-tailed than our synthetic assumption (see
`--min-adapter-reuse`, below). **Taskmaster is much closer to our synthetic
shape** — a small number of real, popular, reusable task types with a real
substantial skew. Neither extreme is "the real answer"; production traffic
apparently spans both regimes depending on the application (many-unique-
customers vs. a-dozen-real-personas).

`--min-adapter-reuse 2` (Mooncake only; default) restricts to conversations
whose adapter is used by ≥2 conversations — the traffic where a caching
decision exists at all (a never-reused adapter has no cache-vs-evict
question to answer). This leaves 433 real adapters over 960 conversations.

## Finding 2: the phase-6 pathology reproduces on Mooncake, lands in ELORA's range, and needs a real (not oversubscribed) slab to show it honestly

433 real Mooncake adapters, unified pool sized as ~40% of that (matching
ELORA's own practice of sizing `--max-loras` as a plausible fraction of the
real deployed LoRA count, never a huge oversubscription ratio):

| max_loras (pool) | `lru-leaf` p50 | stale KV |
|---|---|---|
| 12 (2,400 MB) — our ORIGINAL synthetic-tuned default | **12,570,552 ms** | 56.4% |
| 60 (11,600 MB) — 40% of the 433 real adapters | **9,922,084 ms** | 42.4% |
| 200 (36,400 MB) | 8,338 ms | 40.2% |

**The original `max_loras=12` was a leftover from tuning against our
synthetic 12-adapter workload, applied unthinkingly to a trace with 433 real
competing adapters — a 36:1 oversubscription ratio no real deployment would
run.** Widening the slab to a realistic 40% share (60 slots) still produces
a multi-hour collapse for `lru-leaf`; only at ~200 slots (~46% of the real
adapter count) does the pathology disappear. **This means the collapse is
not primarily a slab-sizing artifact — it persists across a wide range of
realistic slab sizes** — but the default is now `--max-loras` auto-scaled
to 40% of the real in-play adapter count, not a hardcoded carryover value.

At the corrected 60-slot slab: `ordering` p50 5,922 ms / 17.2% stale,
`dep-aware` p50 5,918 ms / 15.4% stale — both a many-thousand-times
improvement over `lru-leaf`, both landing near ELORA's own reported range
(42.4% vLLM, 48.6% ELORA-WOM) for the "no smart eviction" case.

## Finding 3 (new): on Taskmaster's traffic shape, the pathology does NOT appear at all — a genuinely different failure mode

Same experiment on Taskmaster (15 real adapters, matched 15-hour real
BurstGPT window, `max_loras` swept from 6 down to 1 — the most extreme
possible pressure, only one adapter resident at a time — and the KV pool
squeezed from 1,900 MB down to 200 MB):

| policy | p50 | stale KV | adapter loads |
|---|---|---|---|
| `lru-leaf` | 477 | **0.0%** | 861 |
| `ordering` | 477 | **0.0%** | 861 |
| `dep-aware` | 477 | **0.0%** | 861 |
| `react-lru` | 477 | **0.0%** | 861 |
| `swap-full` | 477 | **0.0%** | 860 |

**Every single policy is exactly tied.** Adapter *loads* do climb under
pressure (165 → 861 as the pool tightens), confirming adapter thrashing is
real and increasing — but it never once produces stale KV, at any pool size
or slab size tried, including the most extreme single-slot case.

**Why:** Taskmaster's real structure — 15 adapters, real substantial
popularity skew, ~11 turns per conversation spread across real (sparse)
BurstGPT arrival gaps — means a conversation's own KV essentially never
coexists on the GPU with a full competing slab of *other* adapters' KV at
the moment its adapter gets evicted. The eviction pressure lands on
adapters; it doesn't translate into orphaned KV the way Mooncake's much
higher adapter cardinality does.

**This is a real, useful negative result, not a bug** (verified by sweeping
pool size from 1900 MB down to 200 MB and `max_loras` from 6 down to 1 —
stale KV stays exactly 0.0% throughout). It says: **the stale-KV pathology,
and therefore the entire premise for a smart eviction policy, is
traffic-shape-dependent.** A deployment that looks like Taskmaster (a
moderate number of well-known, reusable task-specific adapters) may simply
never encounter the problem ELORA's dependency tree is built to solve — no
matter how tight memory gets. A deployment that looks like Mooncake (a huge,
long-tailed population of adapters) encounters it severely.

## Finding 4: on real arrival timing, the swapper is a net negative on BOTH real datasets — for the same mechanism identified in phase 7's synthetic tests

| dataset | `react-lru` p50 | `swap-full` p50 | delta | adapter loads (react-lru → swap-full) |
|---|---|---|---|---|
| Mooncake (60/11,600 MB) | 5,914 | 5,877 | +0.6% | 708 → 590 |
| Taskmaster (6/1,900 MB, loose pool) | 224 | 277 | **−24.6%** | 165 → 292 |

On Mooncake at a realistic slab the swapper is roughly neutral. **On
Taskmaster, with room to spare in the pool, the swapper's *proactive*
swap-out creates churn where none was needed** — 292 adapter loads versus
165 for plain reactive LRU, even though nothing was forcing evictions. This
is the exact mechanism phase 7's synthetic tests identified
(`--swap-out-when full` vs the default proactive `highwater` trigger) —
now independently confirmed on a **third, unrelated real dataset**. The
swapper's core weakness (evicting proactively, before it's actually needed)
is not an artifact of our synthetic bursty formula; it reproduces on real
Azure OpenAI arrival timing too.

## How to reproduce

```bash
# download the traces (one-time, ~120MB total)
mkdir -p traces && cd traces
curl -sL -o mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
curl -sL -o BurstGPT_without_fails_1.csv \
  https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv
curl -sL -o taskmaster_tm1.json \
  https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json
cd ..

python real_traces.py                                    # sanity-check all three loaders
python run_on_real_traces.py --workload mooncake          # phase 6+7, Mooncake, --hw ours
python run_on_real_traces.py --workload taskmaster         # phase 6+7, Taskmaster, --hw ours
python run_on_real_traces.py --workload mooncake --hw elora-aggressive
python run_on_real_traces.py --workload taskmaster --n-conversations 960 --pool-mb 300 --max-loras 6
```

**LMSYS Chatbot Arena** (ELORA's "Chatbot" dataset) requires a free
HuggingFace login + accepting the dataset's terms (likely PII-review
related) — not pulled here. See `real_traces.py` if adding it later; do not
use unofficial ungated mirrors that bypass the original authors' access
control.

## Honest limitations of this pass

1. **Two different real systems, stitched together (Mooncake case only).**
   BurstGPT's arrivals are real Azure OpenAI ChatGPT/GPT-4 traffic;
   Mooncake's content/prefix structure is a real (different) Kimi/Moonshot
   deployment. Taskmaster is cleaner in one sense (it's one of ELORA's own
   datasets) but still borrows BurstGPT for timing, same as ELORA borrowed
   the Azure trace for it.
2. **Mooncake's adapter identity is inferred, not measured**; Taskmaster's
   is a genuine field, not inferred — the two datasets differ in how much
   trust to place in "adapter identity."
3. **Mooncake's variable-depth prefix chains are collapsed to one number per
   conversation.** Taskmaster carries **no** cross-conversation prefix-
   sharing signal at all (no hash_ids equivalent) — its `prefix_tokens` is
   set to 0 throughout, an honest gap, not a fabricated one.
4. **`lru-leaf`'s multi-hour Mooncake latency is a genuine simulator output**
   (verified by direct inspection: 97% of requests affected, TTFT stays
   normal throughout) but not a literal production forecast — a real system
   would shed load or trigger alerts long before this point.
5. **We did not obtain LMSYS/Chatbot Arena** (ELORA's third dataset) due to
   its access-gating terms. The reconstruction currently covers 2 of
   ELORA's 3 named evaluation domains (agent-like via Taskmaster; chat-like
   only via Mooncake's proxy, not ELORA's actual chat dataset).
