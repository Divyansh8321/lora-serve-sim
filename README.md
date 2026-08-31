# Multi-tenant LoRA serving: does coordinating adapter and KV cache eviction pay?

A discrete-event simulator for the memory problem in multi-tenant LoRA serving.
One base model, many LoRA adapters, many concurrent multi-turn conversations,
one GPU. Four phases, each isolating one variable.

**Headline:** the stale-KV pathology that motivates recent work (ELORA/FastLibra,
HPCA 2026) is real and reproducible — but **fixing it with cross-pool
coordination alone does not improve latency.** The two pools have to be able to
trade capacity before coordination can pay off. Unification is the enabler;
once unified, in *this* workload simple ordering discipline captures nearly all
of the benefit — which is a claim about the workload as much as the mechanism
(see [Scope & honesty](#scope--honesty)).

**What ELORA actually claims** (paper, HPCA 2026 — not paraphrased from memory):
a *unified caching pool* + a *dependency-aware cache manager* (a RadixAttention
prefix tree whose nodes are LoRAs and KV blocks) + a *cost-model swapper* that
re-scores every tree node every 100ms. Reported: **−45.7% TTFT**, **−37.8%
TPOT**, **+78.9% peak load** vs vLLM. Stock vLLM suffers **42.4%** invalid KV;
ELORA's own no-dependency-manager ablation (ELORA-WOM) still suffers **48.6%**.
This simulator reproduces the pathology and isolates two configurations ELORA's
ablation table skips — see below.

## Run

```bash
python phase1_adapter_caching.py     # adapters only
python phase2_separate_pools.py      # + KV, separate pools (real vLLM architecture)
python phase3_cross_pool_signal.py   # + one cross-pool signal   <- key negative result
python phase4_unified_pool.py        # + unified pool (S-LoRA / ELORA-style)
python phase5_continuous_batching.py # + continuous batching (real execution model)
```
Pure standard library. No GPU, no model, no network.

`validation/` drives a real `vllm serve` with the *identical* workload
(`make_multi_turn_workload`) to check the curve on hardware. See
`validation/README.md`. Phases 6–7 (see [Roadmap](#roadmap)) add the
RadixAttention prefix tree and ELORA's cost-model swapper.

## The four findings

| phase | what changes | result |
|---|---|---|
| 1 | adapter cache only | **3-6% p50** cold swaps, **~0%** warm — the motivation, not a win |
| 2 | + KV, **separate** pools | stale KV **0% → 44%** as the adapter slab shrinks; p50 +24%, disk loads 11 → 49 |
| 3 | + cross-pool signal | stale KV **44% → 33%** (5% size-weighted), **latency unchanged (±3%)** |
| 4 | + **unified** pool | **27%** lower p50, stale KV **46% → 0%**; ordering-only ≈ dependency-aware *in this workload* |
| 5 | + **continuous batching** | pathology reproduces (50% stale KV) but is now **timing-driven** — a sharp TTFT knee from *admission stall*; new **TPOT channel** (17→27) from recompute-prefill; unified stays flat |

### Phase 2 — the pathology is architectural, not incidental

Real vLLM keeps LoRA weights in a pre-allocated slab (`--max-loras`, managed by
`LoRALRUCache`) and KV in a separate `BlockPool`. They never share bytes and
never consult each other — ELORA §II confirms this: vLLM "allocated different
sizes of memory blocks for LoRAs and KVs, preventing their sharing," and sets
the LoRA memory ratio to 0.2 empirically. **Two independently-timed LRUs are
sufficient to produce stale KV** — no shared-memory competition required.
Squeezing the adapter slab from 12 adapters to 3 drives staleness to 44%, which
brackets ELORA's measured figures: **42.4%** for stock vLLM (their motivation,
§I) and **48.6%** for their own unified-pool-minus-dependency-manager ablation
(ELORA-WOM, Fig. 15). Note those are two different systems; earlier drafts of
this README conflated them as "~48%".

Adding a CPU tier to each pool (Part B) cuts disk loads from 49 to 11 and nearly
erases the latency penalty — but staleness stays at 46%. **A CPU tier makes each
mistake cheaper; it does not make fewer mistakes.**

### Phase 3 — the key negative result

Give the adapter LRU one signal: "how many GPU-resident KV caches belong to this
adapter?" Prefer evicting adapters with none. This is the cheapest conceivable
fix — one integer across a boundary, no re-architecture. It maps onto vLLM's
open RFC #37003 (context-aware KV retention) and issue #45325 (adapter residency
is not visible to the scheduler today).

**It works on its target metric and fails on latency.** Stale KV falls from 44%
to 33% (count-based) or 5% (size-weighted). Latency moves by ±3% — noise, and
sometimes negative.

The mechanism: the adapter slab's capacity is *fixed and independent*. Declining
to evict adapter A means evicting adapter B instead, so total reloads are
unchanged (~26-30 either way). The signal changes *which* adapter pays, not how
many pay. And stale KV is not destroyed — when its adapter returns it is reused
free — so staleness is an opportunity cost on *space*, and space freed in the KV
pool cannot be lent to the adapter slab.

### Phase 4 — unification is what unlocks it

Merge the pools and the picture changes. The eviction decision now includes a
choice that did not previously exist — *which type* to evict — between a
180 MB adapter (`ADAPTER_MB`, rank-32 all-linear, **ELORA's rank**) and a
conversation that averages ~25 MB and peaks ~250 MB.

| pool MB | policy | p50 | stale KV | swaps |
|---|---|---|---|---|
| 3600 (fits all) | adapter-first / kv-first / joint | 1052 | 0% | 11 |
| 2500 | adapter-first | 1053 | 2.2% | 11 |
| 2500 | kv-first / joint | 1052 | 0% | 11 |
| **1600 (packed)** | adapter-first | 1127 | **21.9%** | 29 |
| **1600 (packed)** | kv-first / joint | **1045** | **0%** | 13 |

At the packed point, unification + "**KV before adapters**" ordering gives
**−7.3% p50** and drives staleness to zero. And `kv-first` (crude ordering)
performs *identically* to `joint` (full dependency-aware). The orphan-count and
stale-priority refinements never get exercised in this workload.

**This appears to contradict ELORA.** Their ELORA-WOM ablation (unified pool,
dependency manager removed) is **1.51× worse TTFT** than full ELORA — i.e. for
them, dependency-awareness matters a lot. The reconciliation is almost certainly
*prefix sharing*: ELORA's dependency tree is a RadixAttention prefix tree, and
their workloads (LMSYS-33K chat, Azure trace) have heavy cross-query prefix
overlap. This simulator's workload has **none** — each conversation is
independent — so "evict the biggest thing" is already near-optimal. The honest
statement:

**Unification is the enabler. In a workload without prefix sharing, ordering
discipline captures nearly all of the benefit and dependency scoring adds
little. Whether that survives prefix sharing is an open, testable question**
(phase 6, the RadixAttention extension in the [Roadmap](#roadmap)).

Phase 5 sharpens this under continuous batching + a squeezed unified pool
(1800 MB): `kv-first` there beats *even the dependency-aware policy* — p50 724
vs 779, stale KV 0.3% vs 11.5% — because the "smart" policy preserves KV it
should have dropped. `kv-first` pays for it with 8× more KV recompute, which
continuous batching absorbs into the step budget. **Ordering discipline is what
matters; dependency scoring without prefix sharing is worse than useless here.**

### Phase 5 — the pathology under the real execution model

Phases 2–4 are batch-at-a-time: pick an adapter, drain its queue, move on.
ELORA — and production — use **continuous batching**: one rolling batch stepped
one token at a time, finished sequences evicted instantly, waiting requests
spliced in (prompt prefilled, chunked, in the same step). `--max-loras` becomes
what it is in vLLM: the max number of *distinct adapters with live sequences in
the batch*, and **an adapter is pinned while any of its sequences is decoding**.

The pathology reproduces — 50% stale KV at `max_loras=3`, bracketing ELORA's
42.4% / 48.6% — but the mechanism is now visible and it is *timing*, not
capacity:

| max_loras | p50 | p95 | TTFT p50 | TPOT | stale KV | disk loads |
|---|---|---|---|---|---|---|
| 12 | 689 | 1076 | 22 | 17.0 | 0.0% | 11 |
| 7 | 747 | 1531 | 24 | 19.6 | 13.8% | 28 |
| 5 | 871 | 1957 | 29 | 23.1 | 27.7% | 54 |
| 3 | 1170 | 2676 | **296** | 26.6 | 50.1% | 88 |

The sharp TTFT knee at `max_loras=3` (22 ms → 296 ms) is **admission stall**: a
waiting request whose adapter would be the 4th distinct adapter in the batch
cannot be spliced in until a pinned adapter drains. Batch-at-a-time produced the
staleness *number* without this mechanism.

**These numbers are identical at `--adapter-mb 20`, `90`, and `180`.** The
pathology depends on the *ratio* of adapters-that-fit to adapters-in-use, not
the absolute MB — so the earlier `ADAPTER_MB = 20` phase 2–4 results were right
in shape, only mislabelled in units. `core.ADAPTER_MB` is now 180 (rank-32
all-linear, matching ELORA); every phase was re-run and re-documented.

**TPOT rises 17 → 27** — a channel that did not exist before. Recompute-prefill
of rebuilt stale KV competes for the per-step token budget with every decoder in
the batch. This is exactly ELORA's stated reason for their 37.8% TPOT win, and
phases 2–4 could not express it (no shared batch).

The **unified pool stays flat** (p50 689 → 706, stale KV 0%) across the whole
sweep — a big idle conversation's KV is traded for an adapter slot mid-stream.
Phase 4's headline, holding under continuous batching. The phase-3 negative
result also holds: `cost-aware` moves stale KV 27.7% → 24.2% but p50 by <1%.

## Calibration

| constant | value | basis |
|---|---|---|
| KV size | 0.125 MB/token | 8B model, GQA, fp16 |
| adapter size | **180 MB** | rank-32 LoRA, all 7 linear targets, fp16 — 90.2M params × 2 B. ELORA states its LoRAs are "rank 32 or 64". |
| prefill / recompute | 0.15 ms/token | compute-bound, parallel |
| decode | 12 ms/token | bandwidth-bound, ~14GB weights read per step |
| adapter swap cold / warm | 300 / 30 ms | disk-or-S3 vs CPU RAM |
| PCIe transfer | 0.105 ms/MB | 40GB/s **plus** ~10µs dispatch per 128KB chunk |

The PCIe figure matters: at vLLM's 128KB swap granularity, per-call dispatch
overhead *dominates* bandwidth. Using the spec-sheet 40GB/s number alone
(0.025 ms/MB) underestimates transfer cost 4.2×.

**Hardware mismatch to keep in mind:** ELORA runs on H800 (= A100/H100 class,
80GB) over PCIe 5.0 at 128GB/s. Our constants target a smaller box (A10-ish,
40GB/s). Absolute latencies are not directly comparable to ELORA's; the *shape*
of the degradation curve is what the comparison rests on.

**Adapter size: corrected from 20 MB to 180 MB.** Earlier drafts used 20 MB
(≈ rank-8, q/v-only — a narrow-adapter regime). First-principles for the
common serving config: rank *r* LoRA on Llama-3.1-8B, all 7 linear targets
(q,k,v,o,gate,up,down) = `32 layers × [4·(4096·r + r·4096) + 2·(4096·r +
r·14336) + (14336·r + r·4096)]` params × 2 B fp16 — that's 90 MB at r=16 and
**180 MB at r=32**. ELORA uses rank 32/64, so 180 MB is the closest match to
the architecture we compare against. **Every phase was re-run**; the separate-
pool pathology is *scale-invariant* (identical numbers at 20/90/180 MB — it's
the fit ratio that matters), and phase 4/5's unified-pool results got *stronger*
because a 180 MB adapter vs a 250 MB conversation is a real either-or choice.

The PCIe / cold / warm figures check out from first principles — `0.105 ms/MB`
is exactly `0.025` (40 GB/s) + `0.080` (10 µs dispatch per 128 KB).

## Methodology

- Every figure is a **mean over 5 seeds**. Single-run comparisons in this problem
  are unreliable; effects here are frequently smaller than seed-to-seed variance.
- Policies within a phase share identical scheduling and differ **only** in
  eviction, so results isolate the eviction decision.
- "Stale KV" = *resident but not currently usable because its adapter was
  evicted*. It is an opportunity cost, not destruction — the "~60% rescued when
  the adapter returns" figure is an *estimate*, not yet instrumented in the sim
  (see [Scope & honesty](#scope--honesty)).

## Scope & honesty

Things this project has *not* earned the right to claim yet, stated plainly so a
reader can calibrate:

1. **One workload.** Every phase 2–5 number is a single shape: 40 conversations,
   12 adapters, Zipf skew 1.0, 5 turns, 6 s gaps, no prefix sharing. The
   "ordering ≈ dependency-aware" result is workload-dependent and probably
   *breaks* once conversations share prefixes (which is ELORA's regime) — phase 6.
2. **Batch-at-a-time in phases 1–4.** Phase 5 adds continuous batching (rolling
   batch, adapter pinning, `--max-loras` as a batch-composition limit) and shows
   the pathology becomes *timing*-driven. Phases 1–4 keep the simpler model on
   purpose — each isolates one variable — so their numbers stand in that frame.
3. **The 60% rescue figure is an estimate.** Not measured in-sim.
4. **Validation is a plan, not a result** until `validation/run_sweep.sh` has
   actually run on a GPU. Once it does, compare against **phase 5**, not phase 2.
5. **No prefix caching / RadixAttention.** ELORA's baseline vLLM has it on; we
   have nothing. So our separate-pool gap partly *is* the "no prefix caching" gap.
6. **Adapter size is now 180 MB** (rank-32, ELORA's rank) — corrected from an
   earlier 20 MB. All phases re-run; separate-pool results are scale-invariant,
   unified-pool results got stronger (see Calibration).

## Limitations

| not modelled | what real systems do | planned? |
|---|---|---|
| continuous batching | one persistent rolling batch; adapters pinned to in-flight sequences | ✅ **phase 5** |
| prefix caching / RadixAttention | SGLang/ELORA match longest shared prefix at any depth | **phase 6** |
| cost-model swapper | ELORA re-scores every cache node every 100ms (swap cost + freq + LRU term) | **phase 7** |
| reclaim pool | vLLM V1 frees blocks lazily; a request returning before reuse pays nothing | — |
| remote KV tiers | LMCache backends: Redis/Valkey, Mooncake, NVMe, S3 | — |
| multi-node routing | biggest real-world lever; invisible to a single-node model | — |
| KV compression | MLA (~90% reduction), FP8 KV (~50%) | — |

## Roadmap

The point of the phases below is *fidelity to what top systems actually do*, so
the comparison to ELORA is like-for-like instead of "our simplified model vs
their real one". Full detail in `ROADMAP.md`.

| phase | adds | why it matters | status |
|---|---|---|---|
| **5** | continuous batching (step clock, in-flight sequences, mid-stream admission, adapter pinning) | ELORA runs entirely under continuous batching; without it our staleness mechanism is structurally different from theirs | ✅ done — `phase5_continuous_batching.py` |
| **6** | RadixAttention prefix tree (shared system prompts, longest-prefix match, per-node LRU) | ELORA's dependency manager *is* a prefix tree; this is the workload regime where dependency-awareness should start to beat plain ordering | ~2 days |
| **7** | ELORA-style cost-model swapper (periodic re-scoring, `Eval_i` = swap cost + visit freq + `1−sigmoid(t)`) vs plain LRU | reproduces ELORA-WOS ablation; tests whether the cost model earns its 1.42× over LRU in our setting | ~1–2 days |

## References

- **ELORA** (HPCA 2026) / **FastLibra** (arXiv 2505.03756) — same authors, same
  system, renamed. Claimed TTFT reduction changed 63% (FastLibra) → 45.7%
  (ELORA); ELORA also reports −37.8% TPOT, +78.9% peak load. Three parts:
  unified pool + RadixAttention dependency tree + 100ms cost-model swapper.
  Ablations: no-dependency-manager (WOM) = 1.51× worse TTFT, 48.6% invalid KV;
  LRU-instead-of-cost-model (WOS) = 1.42× worse. Eval: Llama3-8B/34B/70B on
  1–8× H800, 20/50/100 LoRAs, LMSYS-33K + OPUS-100 + Taskmaster + Azure trace,
  continuous batching throughout.
- **S-LoRA** (arXiv 2311.03285) — unified paged pool (ELORA builds on its
  operator); discards history KV
- **vLLM** — paged allocation; V1 default preemption is RECOMPUTE, not SWAP
- **vLLM issue #45325** — adapter cache residency not exposed to the scheduler
- **vLLM issue #40268** — KV LRU has no memory of block value
- **vLLM RFC #37003** — context-aware KV retention (open)
- **ConServe** — request-granularity eviction, "coarse-grained fate-sharing"
- **FastSwitch** — swap granularity and `cudaMemcpyAsync` dispatch overhead
- **Predibase LoRAX** — Tiered Weight Caching, Adapter Exchange Scheduling
