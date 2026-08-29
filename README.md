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
```
Pure standard library. No GPU, no model, no network.

`validation/` drives a real `vllm serve` with the *identical* workload
(`make_multi_turn_workload`) to check the phase-2 curve on hardware. See
`validation/README.md`. `phase5_*` (planned, see [Roadmap](#roadmap)) adds
continuous batching so the comparison to ELORA is like-for-like.

## The four findings

| phase | what changes | result |
|---|---|---|
| 1 | adapter cache only | **3-6% p50** cold swaps, **~0%** warm — the motivation, not a win |
| 2 | + KV, **separate** pools | stale KV **0% → 44%** as the adapter slab shrinks; p50 +24%, disk loads 11 → 49 |
| 3 | + cross-pool signal | stale KV **44% → 33%** (5% size-weighted), **latency unchanged (±3%)** |
| 4 | + **unified** pool | **27%** lower p50, stale KV **46% → 0%**; ordering-only ≈ dependency-aware *in this workload* |

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

Merge the pools and the picture changes: 27% lower p50, staleness to zero. The
reason is that unification creates a decision that did not previously exist —
*which type* to evict — and the types differ by an order of magnitude (20MB
adapter vs 250MB conversation).

But the ablation is the interesting part. An uncoordinated policy that merely
checks **KV before adapters** performs *identically* to the full dependency-aware
policy at every pressure level (1052 / 1052 at 800MB; 1062 / 1062 at 400MB).
Evicting one 250MB conversation frees what a dozen adapter evictions would, so
ordering alone avoids the thrash. The orphan-count and stale-priority
refinements never get exercised in this workload.

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
(phase 5 / the RadixAttention extension in the [Roadmap](#roadmap)).

## Calibration

| constant | value | basis |
|---|---|---|
| KV size | 0.125 MB/token | 8B model, GQA, fp16 |
| adapter size | 20 MB | rank-16 LoRA |
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

1. **One workload.** Every phase 2–4 number is a single shape: 40 conversations,
   12 adapters, Zipf skew 1.0, 5 turns, 6 s gaps, no prefix sharing. The phase-4
   "ordering ≈ dependency-aware" result is workload-dependent and probably
   *breaks* once conversations share prefixes (which is ELORA's regime).
2. **Batch-at-a-time execution.** ELORA — and every production system — uses
   continuous batching. Ours doesn't. This changes what "evict an adapter" costs
   (a real adapter is *pinned* while any of its sequences is mid-decode) and
   makes staleness a think-time-gap phenomenon. Closing this is phase 5.
3. **The 60% rescue figure is an estimate.** Not measured in-sim.
4. **Validation is a plan, not a result** until `validation/run_sweep.sh` has
   actually run on a GPU.
5. **No prefix caching / RadixAttention.** ELORA's baseline vLLM has it on; we
   have nothing. So our phase-2 gap partly *is* the "no prefix caching" gap.

## Limitations

| not modelled | what real systems do | planned? |
|---|---|---|
| continuous batching | one persistent rolling batch; adapters pinned to in-flight sequences | **phase 5** |
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

| phase | adds | why it matters | rough size |
|---|---|---|---|
| **5** | continuous batching (step clock, in-flight sequences, mid-stream admission, adapter pinning) | ELORA runs entirely under continuous batching; without it our staleness mechanism is structurally different from theirs | ~2–3 days |
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
