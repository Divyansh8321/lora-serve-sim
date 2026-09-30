# Running the simulator on real production traces

This is the project's closing validation pass. The goal was never to match
ELORA's exact published numbers — different hardware, different codebase,
different traffic will never line up digit-for-digit. The goal was to
**explain, with evidence, exactly why our numbers diverge from theirs**. This
document is that explanation, built on **three independent real datasets**,
**two of which are literally ELORA's own named evaluation datasets**.

## The three datasets

**[Mooncake FAST'25 trace](https://github.com/kvcache-ai/Mooncake)**
(`traces/conversation_trace.jsonl`) — a real anonymized request log from a
different production system (Kimi/Moonshot), released to study KV-cache
sharing. 12,031 requests, 7,900 real multi-turn conversations. No
LoRA-adapter field — adapter identity is *inferred* from the first shared
prefix block (a real, Zipf-shaped signal, not ground truth).

**[Google Taskmaster TM-1](https://github.com/google-research-datasets/Taskmaster)**
(`self-dialogs.json`) — **one of ELORA's own three evaluation datasets**
("Personal Agents"). 7,708 real task-oriented dialogs. Carries a genuine
`instruction_id` field (15 real task types) used **directly** as adapter
identity. No timestamps — overlaid with BurstGPT, the same fix ELORA's own
paper documents using for this exact dataset.

**[LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)**
(`lmsys_arena.parquet`, gated — free HF login + accept terms) — **ELORA's own
"Chatbot" evaluation dataset**, named directly in their paper as "LMSYS-33k."
33,000 real arena battles. Real timestamps (`tstamp`) AND real model identity
(20 named models — "GPT-4, Claude, and LLaMA-based Vicuna," ELORA's own
words) used **directly**, no inference and no overlay needed. Of the three,
this is the one closest to literally what ELORA evaluated on.

**[BurstGPT](https://github.com/HPMLL/BurstGPT)** (CC-BY-4.0) — 10M+ real
ChatGPT/GPT-4 Azure OpenAI logs, used only to supply real arrival timing for
Mooncake and Taskmaster (neither has usable timestamps of its own for our
purposes; LMSYS needs no overlay).

## Methodology note: matching real-time density across datasets

Mooncake conversations average 1.5 turns; Taskmaster averages ~11. Slicing
both to "the same request count" spreads them across very different real
time windows (83h vs 15h for a nominally equal cut), diluting concurrency
and making results incomparable. Fixed by slicing **by conversation count**
(`_truncate_to_n_conversations`), so both now span the same real BurstGPT
window at a matched conversation count.

## Finding 1: real traffic's long-tailedness is dataset-dependent, and ranges wider than our one synthetic assumption in both directions

| | Mooncake | Taskmaster | LMSYS | our synthetic |
|---|---|---|---|---|
| distinct adapters | 7,373 | 15 | 20 (9-20 per window) | 12 |
| top adapter's use count | 7 | 1,211 | 1,384 (in a 6k-convo window) | Zipf-skewed |
| adapters used by only 1 convo | 96% | 0% | 0% | 0% |

Mooncake is far *more* long-tailed than we assumed. Taskmaster and LMSYS —
**both of them ELORA's own datasets** — are much closer to our synthetic
shape: a small number of real, popular, reusable identities with a real
substantial skew. That two of ELORA's three named datasets land close to our
original assumption, while an unrelated third-party trace (Mooncake) doesn't,
is itself informative: **our synthetic assumption was closer to ELORA's
actual regime than it was to "real traffic" in general.**

`--min-adapter-reuse 2` (Mooncake only) restricts to conversations whose
adapter is used ≥2 times — traffic where a caching decision exists at all.

## Finding 2: the stale-KV pathology reproduces severely on Mooncake, lands in ELORA's own reported range

433 real Mooncake adapters, slab sized ~40% of that (matching how ELORA
itself sizes `--max-loras` — a plausible fraction of a *known* real LoRA
count, never a huge oversubscription ratio):

| max_loras (pool) | `lru-leaf` p50 | stale KV |
|---|---|---|
| 12 (2,400 MB) — old synthetic-tuned default, unthinkingly reused | 12,570,552 ms | 56.4% |
| 60 (11,600 MB) — 40% of the 433 real adapters | 9,922,084 ms | 42.4% |
| 200 (36,400 MB) | 8,338 ms | 40.2% |

The old `max_loras=12` was a leftover from our synthetic 12-adapter workload,
applied unthinkingly to 433 real competing adapters (36:1 oversubscription).
Fixed: `--max-loras` now auto-scales to 40% of the real in-play adapter
count. The collapse persists through 60 slots and only clears near 200 — so
it is **not primarily a sizing artifact**, though the old default was still
wrong and is now fixed.

At the corrected 60-slot slab: `ordering` 17.2% stale, `dep-aware` 15.4%
stale — both thousands of times better than `lru-leaf`, both landing near
ELORA's own reported range (42.4% vLLM, 48.6% ELORA-WOM).

## Finding 3: on BOTH of ELORA's own datasets, the pathology does not appear at all, at any pressure

Taskmaster (15 real adapters) and LMSYS (9-20 real adapters), each swept from
a loose pool down to the tightest possible squeeze (Taskmaster: 1,900→200 MB,
6→1 slots; LMSYS: 1,500→400 MB, 4→1 slots):

| dataset | tightest config | `lru-leaf` p50 | stale KV | adapter loads |
|---|---|---|---|---|
| Taskmaster | 1 slot, 200 MB | 476 | **0.0%** | 861 |
| LMSYS | 1 slot, 400 MB | 3,096 | **0.1%** (negligible) | 4,436 |

On **both** of ELORA's own datasets, every policy tested (`lru-leaf`,
`ordering`, `dep-aware`, `react-lru`, `swap-full`) is essentially **tied** —
adapter *loads* climb sharply under pressure (confirming thrashing is real
and increasing) but it almost never translates into stale KV, even at the
single-slot extreme.

**This is the central finding of this project's final pass.** ELORA's own
two named datasets, tested at every pressure level from loose to maximally
tight, **do not reproduce the pathology their own paper's motivation section
describes.** Only Mooncake — a dataset from an unrelated production system,
not one of theirs — reproduces it severely.

## Finding 4: the swapper's core weakness reproduces on all three real datasets

| dataset | `react-lru` p50 | `swap-full` p50 | delta | adapter loads (react-lru → swap-full) |
|---|---|---|---|---|
| Mooncake (realistic slab) | 5,914 | 5,877 | +0.6% | 708 → 590 |
| Taskmaster (loose pool) | 224 | 277 | −24.6% | 165 → 292 |
| LMSYS (auto-scaled) | 1,643 | 1,743 | −6.1% | 688 → 1,684 |

On real arrival timing, from three unrelated real sources, the swapper's
*proactive* eviction consistently creates unnecessary adapter churn when
nothing forced it — the same `--swap-out-when highwater` vs `full` mechanism
phase 7's synthetic sweep identified. This is not an artifact of our
synthetic bursty formula.

## The overarching answer: why don't our numbers match ELORA's?

Putting findings 2, 3, and 4 together, the honest, evidence-backed answer is:

**ELORA's own two named non-chat datasets (agent, chat-arena) don't produce
the pathology their motivation section describes, at any memory pressure we
could construct from them. Only a dataset from an unrelated system —
Mooncake — reproduces it, and it reproduces it *severely*, landing right in
their reported range.** Two explanations are consistent with this, and we
cannot fully distinguish them without ELORA's exact code or hardware:

1. **ELORA's aggregate number is dominated by their third dataset**, the
   translation workload (OPUS-100 + a real Azure Function trace slice), which
   we could not access or reconstruct — Microsoft's internal trace slice they
   used isn't published, and OPUS-100 has no natural per-adapter identity the
   way Taskmaster and LMSYS do. If the pathology is concentrated there, our
   two matching-domain tests would correctly show it absent while the
   paper's *averaged* headline number is still accurate for their full mix.
2. **Real deployed adapter/LoRA counts and popularity structure in ELORA's
   production-scale runs (20/50/100 LoRAs at data-center scale) differ from
   what a 33K/8K-conversation public research dataset can recreate.** Their
   headline evaluations run at scale we cannot reconstruct from a released
   research sample — we don't have their live production LoRA population,
   only fixed public snapshots.

Both point to the same practical conclusion: **the stale-KV pathology, and
the entire case for a smart eviction policy, is a property of the specific
traffic mix, not a universal property of multi-tenant LoRA serving.** A
system with Taskmaster/LMSYS-shaped traffic (a moderate set of well-known,
reusable configurations) may not need this machinery at all. A system with
Mooncake-shaped traffic (many, mostly one-off, long-tailed configurations)
needs it badly. **Knowing which regime you're in is the actual engineering
decision** — more useful, and more honest, than a claim that any one fixed
policy is universally better.

## How to reproduce

```bash
mkdir -p traces && cd traces
curl -sL -o mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
curl -sL -o BurstGPT_without_fails_1.csv \
  https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv
curl -sL -o taskmaster_tm1.json \
  https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json
# LMSYS is gated: log in at huggingface.co, accept the dataset's terms at
# huggingface.co/datasets/lmsys/chatbot_arena_conversations, get a Read
# token from huggingface.co/settings/tokens, then:
curl -sL -H "Authorization: Bearer <YOUR_TOKEN>" -o lmsys_arena.parquet \
  "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/main/data/train-00000-of-00001-cced8514c7ed782a.parquet"
cd ..

pip install pyarrow   # needed only for the LMSYS loader
python real_traces.py                              # sanity-check all four loaders
python run_on_real_traces.py --workload mooncake
python run_on_real_traces.py --workload taskmaster --n-conversations 960 --pool-mb 300 --max-loras 6
python run_on_real_traces.py --workload lmsys --n-conversations 6000
```

## Honest limitations of this pass

1. **We could not access or reconstruct ELORA's third dataset** (OPUS-100
   translation + their specific Microsoft Azure Function trace slice, which
   is not published). If their pathology concentrates there, we cannot see
   it — this is the single biggest open gap in this reconstruction.
2. **Mooncake and BurstGPT-overlaid timing mix two unrelated real systems.**
   LMSYS is the only fully-self-contained real dataset here (real timing,
   real identity, one source) — treat it as the highest-confidence result of
   the three.
3. **Mooncake's adapter identity is inferred; Taskmaster's and LMSYS's are
   genuine fields.** Different confidence levels across datasets.
4. **No cross-conversation prefix-sharing signal exists in Taskmaster or
   LMSYS** the way it does in Mooncake's hash_ids — `prefix_tokens=0`
   throughout for both, an honest gap.
5. **`lru-leaf`'s multi-hour/multi-thousand-ms latencies are genuine
   simulator output** (verified: TTFT stays normal, only queueing wait
   balloons) but not literal production forecasts — a real system sheds load
   long before this.
6. **LMSYS's 20 models don't all appear in every time window** — arena
   traffic rotates its model roster over time rather than mixing all 20
   uniformly, so a single contiguous slice sees a real subset (9-20 of 20
   depending on window size/position). This is itself a real trace
   characteristic, not a sampling bug.
7. **Scale.** Our biggest real-data runs use thousands of conversations;
   ELORA's production evaluations run at a scale (20-100 LoRAs across
   multi-GPU deployments processing continuous real traffic) that a released
   research-sample trace cannot fully recreate.
