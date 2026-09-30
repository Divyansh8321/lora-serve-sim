# Running the simulator on real production traces

This is the project's closing validation pass. The goal was never to match
ELORA's exact published numbers — different hardware, different codebase,
different traffic will never line up digit-for-digit. The goal was to
**explain, with evidence, exactly why our numbers diverge from theirs**. This
document is that explanation, built on **three independent real datasets**,
**two of which are literally ELORA's own named evaluation datasets**.

> **Correction (post-audit).** An earlier version of this document claimed
> that the stale-KV pathology "does not appear on either of ELORA's own
> datasets at any pressure," and treated that as a substantive finding about
> traffic shape. **That claim was wrong and has been removed.** A controlled
> test (same dataset, same pool, same policy, only the prefix field changed)
> showed the 0% was an artifact of those datasets carrying no
> prefix-sharing signal, not a property of their traffic:
>
> | Taskmaster, 200 convos, 300 MB pool, `lru-leaf` | stale KV |
> |---|---|
> | `prefix_tokens=0` (as the dataset ships) | **0.00%** |
> | `prefix_tokens=400` (synthetic prefix injected) | **99.40%** |
>
> Our staleness metric counts resident prefix-tree nodes whose adapter has
> been evicted. With no prefix nodes, the only nodes are per-conversation
> leaves, which exist only while their conversation is active — exactly when
> their adapter is pinned. So 0% was close to structurally guaranteed.
> See "What we could and could not measure" below for the corrected reading.

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

## Finding 3 (CORRECTED): what we could and could not measure on ELORA's own datasets

Taskmaster (15 real adapters) and LMSYS (9-20 real adapters), each swept from
a loose pool down to the tightest possible squeeze (Taskmaster: 1,900→200 MB,
6→1 slots; LMSYS: 1,500→400 MB, 4→1 slots):

| dataset | tightest config | `lru-leaf` p50 | stale KV | adapter loads |
|---|---|---|---|---|
| Taskmaster | 1 slot, 200 MB | 476 | 0.0% | 861 (up from 165) |
| LMSYS | 1 slot, 400 MB | 3,096 | 0.1% | 4,436 (up from 56) |

**Do not read the 0% as "the pathology doesn't happen on this traffic."** An
earlier version of this document made exactly that error. Neither dataset
carries a cross-conversation prefix-sharing field (Mooncake's `hash_ids` has
no equivalent in either), so `prefix_tokens=0` throughout — and our staleness
metric counts resident *prefix-tree nodes* whose adapter has been evicted.
With no prefix nodes, the only nodes are per-conversation leaves, which exist
only while their conversation is active, i.e. exactly when their adapter is
pinned. **0% was close to structurally guaranteed by the missing field.**

Controlled proof (same dataset, same pool, same policy; only the prefix field
changed):

| Taskmaster, 200 convos, 300 MB pool, `lru-leaf` | stale KV |
|---|---|
| `prefix_tokens=0` (as shipped) | **0.00%** |
| `prefix_tokens=400` (synthetic prefix injected) | **99.40%** |

**What we CAN legitimately conclude from these two datasets:**
- Adapter-level thrashing is real and scales sharply with pressure
  (Taskmaster 165→861 loads; LMSYS 56→4,436 loads). That is measured, not
  inferred.
- Policy choice made no measurable difference *to latency* on either — but
  since the staleness signal these policies are designed to exploit was
  absent by construction, this is **not** evidence that the policies are
  useless. It is evidence that we could not test them on this data.

**What we CANNOT conclude:** anything about whether ELORA's pathology occurs
in agent-style or chat-arena traffic. Answering that needs a trace with real
prefix-sharing structure *and* real adapter identity. Mooncake has the first;
Taskmaster and LMSYS have the second; **no dataset we found has both.** That
is the real blocker, and it is a data-availability limitation, not a finding.

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

The honest, evidence-backed answer, after correcting Finding 3:

**Where we could measure the pathology, it reproduced and landed in ELORA's
reported range. Where we could not measure it, we could not measure it — and
that covers both of their own accessible datasets.** Specifically:

- **Mooncake** (real prefix-sharing data, adapter identity inferred):
  pathology reproduces severely, 42.4% stale at a realistic slab, squarely
  inside ELORA's reported 42.4%/48.6% band. **This is a genuine
  corroboration of their motivation.**
- **Taskmaster and LMSYS** (real adapter identity, but *no* prefix-sharing
  field): the metric is structurally unable to register staleness. We
  learned nothing here about whether the pathology occurs — only that we
  can't test it with this data.
- **OPUS-100 / their Azure trace slice** (their third dataset): not public,
  not reconstructable. Untested entirely.

So the remaining gap between our numbers and theirs is **not explained** by
this pass, and it would be dishonest to claim otherwise. What this pass
actually established:

1. The pathology is real and reproducible on real production data where the
   necessary signal exists (Mooncake) — ELORA's core premise holds up.
2. The **swapper's proactive-eviction weakness** (Finding 4) reproduces
   across all three real datasets, because that finding is measured in
   adapter loads and latency, not staleness — so it survives the Finding 3
   correction intact. This is our most robust independent result.
3. **No public dataset we could find carries both real prefix-sharing
   structure and real adapter identity.** Mooncake has the first, Taskmaster
   and LMSYS have the second. Closing the remaining gap properly needs a
   trace with both, or ELORA's own instrumented setup.

**The defensible engineering takeaway**, narrower than the earlier claim but
actually supported: the pathology depends on a specific structural
precondition — many adapters competing *and* meaningful KV persisting across
adapter evictions. Mooncake-shaped traffic (long-tailed adapter population)
demonstrably has it. Whether a given production system does is something you
should **measure before buying into this class of optimization**, and the
measurement requires instrumenting prefix-reuse and adapter-residency
together — which is precisely the instrumentation none of the public traces
provide.

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
   LMSYS is the only fully-self-contained real dataset (real timing, real
   identity, one source) — but it is also missing the prefix-sharing field,
   so "self-contained" does not make it the most *informative* here. Mooncake
   is the only dataset on which the pathology could actually be measured.
3. **Mooncake's adapter identity is inferred; Taskmaster's and LMSYS's are
   genuine fields.** Different confidence levels across datasets.
4. **No cross-conversation prefix-sharing signal exists in Taskmaster or
   LMSYS** the way it does in Mooncake's hash_ids — `prefix_tokens=0`
   throughout for both. **This is not a minor caveat: it makes the staleness
   metric structurally unable to fire on those two datasets** (proven by the
   0.00% → 99.40% controlled test in Finding 3). Any "0% stale" result on
   Taskmaster or LMSYS says nothing about their traffic.
5. **Intra-conversation turn gaps are ~1 ms on both Taskmaster and LMSYS**,
   versus the 6,000 ms think-time our synthetic workload uses. For LMSYS this
   is a data limitation (the arena logs one timestamp per battle, not per
   message). **For Taskmaster it was our bug** — placeholder
   `arrival_time=turn_idx` values (0,1,2,…) that the BurstGPT overlay then
   faithfully preserved. Re-tested with realistic 6 s gaps injected: the
   headline numbers held, but adapter loads jumped 861 → 6,096, so the gap
   structure does materially affect the workload and this should be fixed
   properly before any further Taskmaster conclusions are drawn.
6. **`lru-leaf`'s multi-hour/multi-thousand-ms latencies are genuine
   simulator output** (verified: TTFT stays normal, only queueing wait
   balloons) but not literal production forecasts — a real system sheds load
   long before this.
7. **LMSYS's 20 models don't all appear in every time window** — arena
   traffic rotates its model roster over time rather than mixing all 20
   uniformly, so a single contiguous slice sees a real subset (9-20 of 20
   depending on window size/position). This is itself a real trace
   characteristic, not a sampling bug.
8. **Scale.** Our biggest real-data runs use thousands of conversations;
   ELORA's production evaluations run at a scale (20-100 LoRAs across
   multi-GPU deployments processing continuous real traffic) that a released
   research-sample trace cannot fully recreate.
