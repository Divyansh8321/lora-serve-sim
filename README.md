# Multi-tenant LoRA serving: does coordinating adapter and KV cache eviction pay?

A discrete-event simulator for the memory problem in multi-tenant LoRA serving.
One base model, many LoRA adapters, many concurrent multi-turn conversations,
one GPU. **Seven phases**, each adding one variable, ending with a from-scratch
reconstruction of ELORA's full architecture.

**Headline.** The stale-KV pathology that motivates ELORA/FastLibra (HPCA 2026)
is **real and reproducible** (we measure 44–50% invalid KV vs. their 42–49%).
But decomposing their fix into levers their own ablations don't separate:

- **Pool unification does the heavy lifting** (−7% p50, staleness → 0). A
  cross-pool *signal* without unification moves the staleness metric but not
  latency — capacity isn't fungible until the pools are merged.
- **Prefix-structure-aware eviction *ordering* captures most of the rest.**
  Blind LRU over a prefix tree collapses ~40× under shared prompts; a crude
  "biggest-KV-first, adapters-last" rule already avoids that.
- **The elaborate dependency *scoring* ELORA centres its design on is a real
  but ~2–7% refinement** in these workloads — not the 51% its whole-vs-nothing
  ablation implies.
- **ELORA's 100 ms timer-driven cost-model swapper is net-negative here**
  (−4% to −16% p50, worse under bursts) — on a single smaller GPU, reacting on
  demand beats a proactive timer.

**What ELORA actually claims** (paper, HPCA 2026 — not paraphrased from memory):
a *unified caching pool* + a *dependency-aware cache manager* (a RadixAttention
prefix tree whose nodes are LoRAs and KV blocks) + a *100 ms cost-model swapper*.
Reported: **−45.7% TTFT**, **−37.8% TPOT**, **+78.9% peak load** vs vLLM. Stock
vLLM suffers **42.4%** invalid KV; ELORA's own no-dependency-manager ablation
(ELORA-WOM) still suffers **48.6%**; LRU-instead-of-cost-model (ELORA-WOS) is
1.42× worse TTFT. This simulator reproduces the pathology and isolates the
configurations ELORA's ablation table skips — see below.

## Run

```bash
python phase1_adapter_caching.py     # adapters only
python phase2_separate_pools.py      # + KV, separate pools (real vLLM architecture)
python phase3_cross_pool_signal.py   # + one cross-pool signal   <- key negative result
python phase4_unified_pool.py        # + unified pool (S-LoRA / ELORA-style)
python phase5_continuous_batching.py # + continuous batching (real execution model)
python phase6_radix_prefix.py        # + RadixAttention prefix tree + prefix sharing
python phase7_cost_swapper.py        # + ELORA's 100ms cost-model swapper, bursty traffic
```
Pure standard library. No GPU, no model, no network.

`validation/` drives a real `vllm serve` with the *identical* workload
(`make_multi_turn_workload`) to check the curve on hardware. See
`validation/README.md`. **All 7 phases are built** — phases 5–7 progressively
match ELORA's real execution model so the comparison is like-for-like.

## The findings

| phase | what changes | result |
|---|---|---|
| 1 | adapter cache only | **3-6% p50** cold swaps, **~0%** warm — the motivation, not a win |
| 2 | + KV, **separate** pools | stale KV **0% → 44%** as the adapter slab shrinks; p50 +24%, disk loads 11 → 49 |
| 3 | + cross-pool signal | stale KV **44% → 34%**, **latency unchanged (±5%, often negative)** |
| 4 | + **unified** pool | at the packed point **−7% p50**, stale KV **22% → 0%**; ordering-only ≈ dependency-aware |
| 5 | + **continuous batching** | pathology reproduces (50% stale KV) but is now **timing-driven** — a sharp TTFT knee from *admission stall*; new **TPOT channel** (17→27) from recompute-prefill; unified stays flat |
| 6 | + **RadixAttention prefix tree** + prefix sharing | blind LRU-of-leaves **collapses** (p50 ×40) under sharing; prefix-aware **ordering** avoids it; ELORA-style **dependency scoring** adds a further **2–7%** on top — real, but far short of ELORA's claimed 1.51× |
| 7 | + **ELORA's 100 ms cost-model swapper** + bursty/drifting traffic | the timer-driven swapper is **net-negative** here (−4% to −16% p50, worse as bursts intensify) — proactive swap-out churns; **react-on-demand wins**. Opposite of ELORA's "−1.42× without it" |

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
dependency manager removed) is **1.51× worse TTFT** than full ELORA. The
suspected reconciliation was *prefix sharing* — ELORA's dependency tree is a
RadixAttention prefix tree, and their workloads (LMSYS-33K chat, Azure trace)
have heavy cross-query prefix overlap that this workload lacks. **Phase 6 tests
this directly** and the answer is nuanced — see below.

Phase 5 sharpens the pre-phase-6 picture: under continuous batching + a squeezed
unified pool (1800 MB), `kv-first` beats *even the dependency-aware policy* —
p50 724 vs 779, stale KV 0.3% vs 11.5% — because the "smart" policy preserves
KV it should have dropped. It pays with 8× more KV recompute, which continuous
batching absorbs into the step budget.

### Phase 6 — does the prefix tree change the verdict?

Phase 6 adds a **RadixAttention prefix tree** (KV keyed by token prefix, shared
system-prompt nodes near the root, LoRA-rooted subtrees — ELORA's exact
structure) and a workload where conversations share a system prompt. Three
eviction policies, sweeping shared-prefix length on a generous 3000 MB pool:

| prefix tokens | `ordering` p50 | `dep-aware` p50 | dep vs ordering | `lru-leaf` p50 |
|---|---|---|---|---|
| 0 | 689 | 689 | 0% | 689 |
| 400 | 732 | 715 | **+2.2%** | 711 |
| 800 | 842 | 785 | **+6.7%** | 884 |
| 1600 | 1214 | 1160 | **+4.5%** | **28 553** |

**Two results, and the smaller one is ELORA's:**

1. **Prefix-*structure* awareness matters a lot.** `lru-leaf` — blind LRU over
   leaves — collapses as the shared prefix grows (p50 ×40 at 1600 tokens): it
   evicts shared prefix nodes that many sequences need, forcing everyone to
   re-prefill. Any policy that doesn't blindly LRU-evict a shared node avoids
   this — *including plain `ordering`*, which evicts biggest-single-node-first
   (a shared prefix under an active adapter is neither the biggest node nor
   unpinned).

2. **The dependency *scoring* ELORA layers on top is worth ~2–7%.** `dep-aware`
   (protect an adapter with live shared KV; evict widely-shared prefixes last)
   beats `ordering` by 2–7% p50 and much better TTFT (38 vs 102 ms at 800
   tokens). Real, consistent, grows with sharing — but nowhere near ELORA's
   1.51×.

**Reconciliation with ELORA-WOM.** ELORA-WOM removes the dependency manager
*entirely* — it behaves like our `lru-leaf` (which does collapse ~40×), not
like `ordering`. ELORA's paper never isolates "prefix-aware ordering, no
dependency scoring", which is the cheap policy that captures most of the win.
**Honest read: the pathology is real, prefix structure must be respected, but
simple ordering discipline — not elaborate dependency scoring — does the heavy
lifting.** Dependency scoring is a real but modest (~5%) refinement on top.

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

### Phase 7 — does ELORA's timer-driven swapper help?

ELORA's third component: instead of evicting only when memory fills, a swapper
runs every **100 ms**, re-scores every cache node with a cost model
(`Eval_i` = swap-cost + frequency + soft-recency + a "keep enough LoRAs" floor),
proactively evicts low-scored nodes above a high-water mark, and **prefetches**
during idle windows. ELORA's ablation: replace it with plain LRU and TTFT gets
**1.42× worse**.

This needs bursty traffic to matter, so phase 7 adds `make_bursty_workload`:
inhomogeneous Poisson arrivals (5–10× rate spikes) and a Zipf popularity
ranking that re-shuffles every 60 s. On a 2400 MB pool, 400-token shared
prefix:

| workload | `react-lru` p50 | `react-dep` p50 | `swap-full` p50 | swapper vs react-lru |
|---|---|---|---|---|
| steady | 753 | 745 | 781 | **−3.7%** |
| burst ×5 | 877 | 845 | 981 | **−11.8%** |
| burst ×10 | 1497 | 1460 | 1731 | **−15.6%** |

**The timer-driven swapper is net-negative here — and worse as bursts
intensify. The opposite of ELORA's claim.** The proactive swap-out is churn:
it evicts entries at 92% fill that are needed again seconds later, so adapter
loads climb (14 → 34 vs 14 → 23 for `react-lru`). `swap-noprefetch` is worse
still (−17.8% at ×5), confirming the proactive *swap-out* is the harmful part;
prefetch fires only 5–6 times and doesn't offset it. Term ablations
(`swap-wo-freq`, `swap-wo-swap`) move <0.3% — no single term carries anything
because the whole approach is net-negative in this regime.

**`react-dep`** — phase 6's dependency-aware eviction with *no timer* — is the
consistent winner (+1% to +4% over `react-lru` across every config: tight pool,
long prefix, fast drift).

**Reconciliation with ELORA-WOS's 1.42×:** same pattern as every other phase.
ELORA-WOS keeps the proactive timer + prefetch scaffolding and only swaps the
*scoring* for LRU; our `react-lru` has no timer at all. ELORA's H800 has 8×
our PCIe bandwidth and 80 GB HBM, so aggressive proactive swapping churns far
more cheaply there, and their Azure-trace bursts may have the long idle windows
prefetch needs. **On a single smaller GPU with tight memory, reacting on demand
beats a 100 ms timer.**

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

1. **One workload family.** Every number is one shape: 40 conversations, 12
   adapters, Zipf skew 1.0, 5 turns, 6 s gaps. Phase 6 adds a shared-prefix
   variant but the arrival/adapter structure is unchanged. No Azure-trace
   burstiness, no drifting adapter popularity (both in ELORA's eval).
2. **Batch-at-a-time in phases 1–4.** Phases 5–6 add continuous batching.
   Phases 1–4 keep the simpler model on purpose — each isolates one variable.
3. **The 60% rescue figure is an estimate.** Not measured in-sim.
4. **Validation is a plan, not a result** until `validation/run_sweep.sh` has
   actually run on a GPU. Once it does, compare against **phase 5/6**, not phase 2.
5. **Phase 6's prefix tree is coarse.** KV is keyed by (adapter, group) at the
   prefix and by cid for the continuation — no token-level radix matching, no
   partial-prefix reuse. Enough for the eviction-policy question, not a
   faithful RadixAttention.
6. **Adapter size is 180 MB** (rank-32, ELORA's rank) — corrected from an
   earlier 20 MB. All phases re-run; separate-pool results are scale-invariant,
   unified-pool results got stronger (see Calibration).

## Limitations

| not modelled | what real systems do | planned? |
|---|---|---|
| continuous batching | one persistent rolling batch; adapters pinned to in-flight sequences | ✅ **phase 5** |
| prefix caching / RadixAttention | SGLang/ELORA match longest shared prefix at any depth | ✅ **phase 6** (coarse) |
| cost-model swapper | ELORA re-scores every cache node every 100ms (swap cost + freq + LRU term) | ✅ **phase 7** |
| bursty / drifting workload | Azure Function trace; adapter popularity shifts over time | ✅ **phase 7** (`make_bursty_workload`) |
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
| **6** | RadixAttention prefix tree (shared system prompts, LoRA-rooted subtrees, prefix-aware eviction) | ELORA's dependency manager *is* a prefix tree | ✅ done — `phase6_radix_prefix.py`. Result: blind LRU-of-leaves collapses; prefix-aware *ordering* fixes it; dependency *scoring* adds ~2–7% on top |
| **7** | ELORA-style 100 ms cost-model swapper + idle prefetch, bursty/drifting workload | reproduces ELORA-WOS ablation | ✅ done — `phase7_cost_swapper.py`. Result: the timer-driven swapper is net-negative here (−4% to −16%); react-on-demand wins; `react-dep` (phase 6, no timer) is best |

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
