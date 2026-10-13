# Running the simulator on real production traces

This is the project's closing validation pass. The goal was never to match
ELORA's exact published numbers — different hardware, different codebase,
different traffic will never line up digit-for-digit. The goal was to
**explain, with evidence, exactly why our numbers diverge from theirs**. This
document is that explanation, built on **three independent real datasets**,
**two of which are literally ELORA's own named evaluation datasets**.

> ## Update 4 (2026-10-11): phase 7 rebuilt to ELORA's spec, re-run on real data
>
> Phase 7 is now `phase7_elora.py` (host-memory KV tier, LoRA-parent tree,
> Eq. 3–6; see the README). The heuristic `swap-full` / `react-lru` numbers in
> Correction 3 and Finding 4 come from the archived version
> (`archive/phase7_heuristic_swapper.py`). On the synthetic chat workload the new
> phase 7 shows ELORA −21% to −87% TTFT vs an oracle-tuned vLLM, all from the
> host KV tier. On the real traces (3 windows each, `--hw ours`), change in
> mean TTFT vs oracle vLLM:
>
> | dataset | turn timing | `elora` | `elora-nohost` | `vllm-offload` | `slora` | KV blocks swapped back in (`elora`) |
> |---|---|---|---|---|---|---|
> | LMSYS (2,000 convos, 1,500 MB) | as loaded | −2% | −2% | −1% | 0% | 0 |
> | LMSYS | `--think-ms 15000` | −16% (69 → 58 ms) | −16% | −8% | −11% | 0 |
> | Taskmaster (2,000 convos, 1,900 MB) | as loaded | 0% | 0% | 0% | +7% | 0 |
> | Taskmaster | `--think-ms 15000` | −6% | −3% | +1% | +69% | 1,218 |
> | Mooncake (960 convos, 433 adapters, 11,600 MB, big prompts) | BurstGPT | **+10%** (785 → 864 ms) | +36% | **−30%** (553 ms) | +77% | 6,172 |
>
> Oracle vLLM on Mooncake is `max_loras` 60 (TTFT 785 ms; the 0.2-ratio
> setting, 12, gives 1,067 ms). With 433 adapters, `elora`'s one shared pool
> keeps pushing adapters out to make room for big KV and reloads them
> (1,014 adapter loads vs 82 for vLLM ml=60), so it **loses to tuned vLLM**.
> `vllm-offload` ml=60 — vLLM's fixed adapter slab plus a CPU KV tier — is the
> best design here by a wide margin. Again the host KV tier is what helps;
> the unified pool hurts when adapters are many.
>
> **Caveat:** the Mooncake pool is auto-sized as 60 adapters + 800 MB
> (11,600 MB), so vLLM ml=60 leaves only ~800 MB for KV while median prompts
> are ~6,900 tokens (~860 MB). That memory split is our harness's choice, not
> a realistic deployment, and it favours a large fixed adapter slab. Treat
> this row as unconfirmed until rerun with a realistic GPU size.
>
> **Why LMSYS and Taskmaster barely separate the designs.** ELORA's win comes from
> conversations that come back *after* their KV was pushed out of GPU memory.
> As loaded, LMSYS (one timestamp per conversation) and Taskmaster (no
> timestamps; our loader spaces turns 1 ms apart) deliver all of a
> conversation's turns at once, so evicted KV is never needed again ("KV
> blocks swapped back in" = 0). `--think-ms` adds an *assumed* gap between
> turns. Even then their turns are tiny (median 16 / 8 new prompt tokens), so
> recomputing evicted KV is cheap and the host tier has little to save.

> ## ⚠️ Correction 3 (code audit, 2026-10-11) — supersedes the numbers below
>
> Phase 6's `lru-leaf` treated a node as a leaf only if it had *no children
> at all*. Children are never pruned, so **shared prefix nodes could never be
> evicted**. Every staleness number and every `lru-leaf` latency in this
> document came from that bug:
>
> | measurement | as published | old code, re-run | fixed code |
> |---|---|---|---|
> | Taskmaster control, prefix 400, 300 MB | 99.40% stale | **99.40%**, p50 3,752,225 ms | **0.00%**, p50 551 ms |
> | Mooncake, 960 convos, 11.6 GB, 60 slots | 42.5% stale, `lru-leaf` p50 9.9M ms | — | **0.4%** stale, p50 6,971 ms |
>
> The real-trace loaders also **discarded real prompt sizes** (Mooncake's
> `input_length`, median 6,909 tokens). They now pass the new tokens per turn through
> (`Request.input_tokens`). Re-run on fixed code (3 real-history windows each):
>
> | dataset | phase 6 policies (p50) | `react-lru` → `swap-full` | adapter loads |
> |---|---|---|---|
> | Mooncake (433 adapters, 11.6 GB) | 6,971 / 7,018 / 6,967 — all within 1% | 6,977 → 6,985 (**−0.1%**) | 653 → 678 |
> | Taskmaster, loose (1,900 MB, 6 slots) | — | 241 → 305 (**−26.9%**) | 165 → 310 |
> | Taskmaster, tight (300 MB, 6 slots) | 491 for all three | 491 → 491 (0%) | 861 → 860 |
> | LMSYS (6,000 convos, 1,500 MB) | 1,842 / 1,842 / 1,836 | 1,842 → 1,921 (**−4.3%**) | 1,426 → 2,191 |
>
> **What stands:** the swapper's proactive-eviction churn costs 27% on Taskmaster
> and 4% on LMSYS and is neutral on Mooncake (Finding 4, with updated magnitudes).
> The adapter-population facts (Finding 1) and the `max_loras` sizing fix
> also stand. **What falls:** every staleness figure and the `lru-leaf`
> latency collapse. On all three real datasets, *no eviction policy is
> distinguishable from plain LRU*. The simulator's limits behind that are listed in
> `ELORA_ALIGNMENT.md` (no host-memory tier, drop-instead-of-swap, our
> heuristics instead of ELORA's Eq. 3–6).

> ## ⚠️ Corrections 1–2 (earlier self-audit)
>
> A self-audit found **two** overstated claims, both now corrected. Together
> they mean **this document contains no valid measurement of the stale-KV
> pathology on real data.**
>
> **Correction 1 — "the pathology doesn't appear on ELORA's own datasets" was
> an artifact.** Taskmaster and LMSYS carry no prefix-sharing field, so
> `prefix_tokens=0`. Our staleness metric counts resident prefix-tree nodes
> whose adapter was evicted; with no prefix nodes, the only nodes are
> per-conversation leaves, which exist only while their conversation is
> active — exactly when its adapter is pinned. 0% was near-structurally
> guaranteed. Controlled test (Taskmaster, 200 convos, 300 MB, `lru-leaf`):
>
> | prefix_tokens | stale KV |
> |---|---|
> | 0 (as shipped) | **0.00%** |
> | 400 (injected) | **99.40%** |
>
> **Correction 2 — the Mooncake "42.4% matches ELORA's 42.4%" was a
> coincidence of a constant we chose.** Our derivation of `prefix_tokens`
> from Mooncake's `hash_ids` collapses to a **constant 1024** for every
> conversation (min == median == max), not the "variable, Zipf-shaped,
> growing turn-by-turn" structure earlier drafts described. And staleness is
> highly sensitive to that constant (Mooncake, 960 convos, 11.6 GB, 60 slots):
>
> | prefix_tokens | stale KV |
> |---|---|
> | 1024 (the value we ship) | **42.5%** |
> | 512 | **53.7%** |
> | 0 | **0.0%** |
>
> So the numerical agreement with ELORA's reported 42.4% reflects our chosen
> constant, **not independent corroboration.** It should never have been
> presented as a match.
>
> **What still stands:** Finding 4 (the swapper's proactive-eviction churn) —
> measured in adapter loads and latency, not staleness — though it holds on
> only **two of three** datasets, not all three as originally written. The
> synthetic-workload results (phases 2–7, `--sweep-scale`, `--swap-out-when`
> attribution) are unaffected throughout.

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
| adapters used by only 1 convo | 94% | 0% | 0% | 0% |

Mooncake is far *more* long-tailed than we assumed. Taskmaster and LMSYS —
**both of them ELORA's own datasets** — are much closer to our synthetic
shape: a small number of real, popular, reusable identities with a real
substantial skew. That two of ELORA's three named datasets land close to our
original assumption, while an unrelated third-party trace (Mooncake) doesn't,
is itself informative: **our synthetic assumption was closer to ELORA's
actual regime than it was to "real traffic" in general.**

`--min-adapter-reuse 2` (Mooncake only) restricts to conversations whose
adapter is used ≥2 times — traffic where a caching decision exists at all.

## Finding 2 (SUPERSEDED by Correction 3 — staleness and `lru-leaf` latencies below were the leaf bug): Mooncake's staleness numbers are a function of our own derived constant, not a measurement

433 real Mooncake adapters, slab sized ~40% of that:

| max_loras (pool) | `lru-leaf` p50 | stale KV |
|---|---|---|
| 12 (2,400 MB) — old synthetic-tuned default | 12,570,552 ms | 56.4% |
| 60 (11,600 MB) — 40% of the 433 real adapters | 9,922,084 ms | 42.5% |
| 200 (36,400 MB) | 8,338 ms | 40.2% |

**The `max_loras` fix is real and stands.** The old 12 was a leftover from our
synthetic 12-adapter workload applied to 433 real competing adapters (36:1
oversubscription); `--max-loras` now auto-scales to 40% of the real in-play
count. The latency collapse persists through 60 slots and only clears near
200, so it isn't primarily a sizing artifact.

**The staleness numbers in that table do NOT stand as measurements.** Our
`prefix_tokens` derivation from Mooncake's `hash_ids` yields a constant 1024
for every conversation, and staleness tracks that constant directly:

| prefix_tokens (960 convos, 11.6 GB, 60 slots) | stale KV | adapter loads |
|---|---|---|
| 1024 (shipped) | 42.5% | 140 |
| 512 | 53.7% | 305 |
| 0 | 0.01% | 554 |

Earlier drafts framed "42.4% on Mooncake ≈ ELORA's reported 42.4%" as
independent corroboration of their premise. **It is not.** It is the output
of a constant we chose, which happens to land near their number. Choosing 512
instead would have produced 53.7% and an equally confident-sounding but
different story.

**What this leaves:** we have no valid real-data measurement of the stale-KV
pathology. Mooncake is the only dataset with prefix data at all, and our
reduction of it to one constant destroys the variation that would make the
measurement meaningful. Doing this properly requires modelling Mooncake's
actual variable-depth, per-turn-growing hash chains rather than collapsing
them — which the current phase 6/7 tree model (one `prefix_tokens` per
conversation) cannot represent without modification.

## Finding 3 (CORRECTED; the 99.40% control below was the leaf bug — see Correction 3): what we could and could not measure on ELORA's own datasets

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

## Finding 4 (stands; magnitudes updated in Correction 3: Taskmaster −26.9%, LMSYS −4.3%, Mooncake −0.1%): the swapper's proactive eviction hurts on TWO of three real datasets

| dataset | `react-lru` p50 | `swap-full` p50 | delta | adapter loads (react-lru → swap-full) |
|---|---|---|---|---|
| Mooncake (realistic slab) | 5,911 | 5,854 | **+1.0%** | 703 → **588** (churn *down*) |
| Taskmaster (loose pool) | 225 | 277 | **−23.5%** | 165 → 292 (churn up) |
| LMSYS (auto-scaled) | 1,611 | 1,718 | **−6.6%** | 56 → 1,116 (churn way up) |

This is the **only real-data finding unaffected by the prefix-derivation
problem**, because it is measured in adapter loads and latency, not staleness.

Earlier drafts said "reproduces on all three." **That was wrong** — it
reproduces on two. On Mooncake the swapper actually *reduces* churn (703 →
588) and is marginally faster. The honest statement: on two of three real
datasets the swapper's proactive eviction creates unnecessary churn and costs
6–24% p50; on the third it helps slightly. Directionally consistent with the
synthetic `--swap-out-when` finding in the majority of cases, but **not
unanimous**, and the disagreeing case is the one dataset with the most
adapters in play.

## The overarching answer: why don't our numbers match ELORA's?

The honest answer, after both corrections: **we did not explain it, and this
pass produced no valid real-data measurement of the pathology at all.**

- **Mooncake** — the only dataset with any prefix data. But our reduction of
  its variable-depth hash chains to one number yields a constant, and
  staleness tracks that constant (1024→42.5%, 512→53.7%, 0→0%). The apparent
  agreement with ELORA's 42.4% is an artifact of the constant we picked.
  **Not a measurement.**
- **Taskmaster / LMSYS** — real adapter identity, no prefix field, so the
  staleness metric cannot fire. **Not a measurement.**
- **OPUS-100 / their Azure trace slice** — not public. **Untested.**

What this pass *did* legitimately establish:

1. **Finding 4** — the swapper's proactive eviction costs 6–24% p50 on two of
   three real datasets (helps ~1% on the third). Measured in adapter loads and
   latency, so unaffected by the prefix problem. **The only surviving
   real-data result.**
2. **Finding 1** — real adapter populations vary enormously (Mooncake 7,373
   adapters, 94% used once; Taskmaster 15; LMSYS 20). Structural fact, no
   derivation involved.
3. **The `max_loras` sizing bug** — a genuine methodology fix.
4. **A concrete, named blocker:** no public dataset carries both real
   prefix-sharing structure *and* real adapter identity, and our tree model
   (one `prefix_tokens` per conversation) cannot represent Mooncake's real
   variable-depth chains even where the data exists.

**What would actually be needed** to answer the original question: extend the
phase 6/7 prefix tree to model per-turn-growing, variable-depth chains
directly from Mooncake's `hash_ids` instead of collapsing them to a scalar.
That's a real change to the simulator's data model, not another dataset or
another sweep. Until then, any staleness claim from this project rests on
synthetic prefix structure (phases 2–7, where the generator produces genuinely
varied values) — which is defensible as a *model* but is not real-data
validation.

**The honest headline for this project:** it reproduces the *mechanisms*
(admission stall, proactive-eviction churn, the capacity-fungibility argument
for pool unification) on a from-scratch simulator, and it identifies exactly
which of ELORA's claims can and cannot be checked against public data and why.
It does **not** independently validate ELORA's reported magnitudes on real
traffic, and earlier drafts of this document wrongly implied it did.

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
