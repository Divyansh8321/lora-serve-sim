# Multi-tenant LoRA serving: does coordinating adapter and KV cache eviction pay?

A discrete-event simulator for the memory problem in multi-tenant LoRA serving.
One base model, many LoRA adapters, many concurrent multi-turn conversations,
one GPU. Four phases, each isolating one variable.

**Headline:** the stale-KV pathology that motivates recent work (ELORA/FastLibra,
HPCA 2026) is real and reproducible — but **fixing it with cross-pool
coordination alone does not improve latency.** The two pools have to be able to
trade capacity before coordination can pay off. Unification is the enabler;
once unified, simple ordering discipline captures nearly all of the benefit.

## Run

```bash
python phase1_adapter_caching.py     # adapters only
python phase2_separate_pools.py      # + KV, separate pools (real vLLM architecture)
python phase3_cross_pool_signal.py   # + one cross-pool signal   <- key negative result
python phase4_unified_pool.py        # + unified pool (ELORA architecture)
```
Pure standard library. No GPU, no model, no network.

## The four findings

| phase | what changes | result |
|---|---|---|
| 1 | adapter cache only | **3-6% p50** cold swaps, **~0%** warm — the motivation, not a win |
| 2 | + KV, **separate** pools | stale KV **0% → 44%** as the adapter slab shrinks; p50 +24%, disk loads 11 → 49 |
| 3 | + cross-pool signal | stale KV **44% → 33%** (5% size-weighted), **latency unchanged (±3%)** |
| 4 | + **unified** pool | **27%** lower p50, stale KV **46% → 0%** |

### Phase 2 — the pathology is architectural, not incidental

Real vLLM keeps LoRA weights in a pre-allocated slab (`--max-loras`, managed by
`LoRALRUCache`) and KV in a separate `BlockPool`. They never share bytes and
never consult each other. **Two independently-timed LRUs are sufficient to
produce stale KV** — no shared-memory competition required. Squeezing the
adapter slab from 12 adapters to 3 drives staleness to 44%, bracketing the ~48%
invalid-KV figure ELORA reports.

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

**Unification is the enabler; ordering discipline captures nearly all of the
benefit; elaborate dependency scoring adds little on top.**

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

## Methodology

- Every figure is a **mean over 5 seeds**. Single-run comparisons in this problem
  are unreliable; effects here are frequently smaller than seed-to-seed variance.
- Policies within a phase share identical scheduling and differ **only** in
  eviction, so results isolate the eviction decision.
- "Stale KV" = *resident but not currently usable because its adapter was
  evicted*. It is an opportunity cost, not destruction — roughly 60% is rescued
  when the adapter returns.

## Limitations

| not modelled | what real systems do |
|---|---|
| reclaim pool | vLLM V1 frees blocks lazily; a request returning before reuse pays nothing |
| prefix caching / RadixAttention | SGLang matches longest shared prefix at any depth |
| cross-conversation sharing | shared system prompts and tool definitions are identical across users |
| remote KV tiers | LMCache backends: Redis/Valkey, Mooncake, NVMe, S3 |
| multi-node routing | biggest real-world lever; invisible to a single-node model |
| continuous batching | one persistent large batch, not batch-at-a-time |
| KV compression | MLA (~90% reduction), FP8 KV (~50%) |

## References

- **ELORA** (HPCA 2026) / **FastLibra** (arXiv 2505.03756) — same work, two
  versions; claimed TTFT reduction changed 63.4% → 45.7% between them
- **S-LoRA** (arXiv 2311.03285) — unified paged pool; discards history KV
- **vLLM** — paged allocation; V1 default preemption is RECOMPUTE, not SWAP
- **vLLM issue #45325** — adapter cache residency not exposed to the scheduler
- **vLLM issue #40268** — KV LRU has no memory of block value
- **vLLM RFC #37003** — context-aware KV retention (open)
- **ConServe** — request-granularity eviction, "coarse-grained fate-sharing"
- **FastSwitch** — swap granularity and `cudaMemcpyAsync` dispatch overhead
- **Predibase LoRAX** — Tiered Weight Caching, Adapter Exchange Scheduling
