# Multi-tenant LoRA serving: when does smart cache management actually help?

A discrete-event simulator for the memory problem in multi-tenant LoRA serving.
One base model, many LoRA adapters, many concurrent multi-turn conversations,
one GPU's worth of memory. Four phases, each adding one mechanism, each with a
measured finding.

**Headline:** as the serving system gets structurally better — a memory tier, an
I/O overlap — the value of clever eviction policy shrinks. Phase 2 measures a
27% win for dependency-aware eviction; adding a CPU tier drops the same policy's
benefit to 2%. **Memory hierarchy substitutes for policy intelligence.**

## Run

```bash
python phase1_adapter_caching.py     # adapters only
python phase2_joint_kv.py            # + KV caches, no CPU tier  (= vLLM V1 default)
python phase3_memory_tiers.py        # + CPU RAM tier
python phase4_prefetch.py            # + I/O / compute overlap
```
Pure standard library. No GPU, no model, no network. Each file is standalone and
carries its own findings in its docstring.

## The four findings

| phase | mechanism added | measured result |
|---|---|---|
| 1 | adapter cache only | **3-6% p50, 4-8% p95** with cold adapters; **~0%** with warm |
| 2 | + KV caches, evict = destroy | **27%** p50 at high pressure; stale KV peaks **45.6%** |
| 3 | + CPU RAM tier | gap collapses **27% → 2%** (generous tier), back to **17.5%** (minimal) |
| 4 | + queue-driven prefetch | **6.7%**, 95% CI [4.4%, 8.9%] over 24 seeds |

**Phase 1** is the motivation, not a win. Smart adapter caching does cut swaps
(~480 → ~395 per 1200 requests) but swapping is only ~20% of wall-clock time,
so it buys a few percent. If adapters were the whole story this problem would
not be worth solving.

**Phase 2** models vLLM V1's documented default, whose preemption mode is
RECOMPUTE, not SWAP — evicted KV is destroyed and rebuilt. It is also the
baseline ELORA (HPCA 2026) / FastLibra measure against. Measured stale-KV of
45.6% brackets the ~48% invalid-KV figure they report for vLLM, and the 27%
improvement is the same order as their claimed 45.7% TTFT reduction. This is an
independent reproduction of their regime.

**Phase 3** is the contribution. ELORA's improvement is measured against a
baseline that recomputes rather than offloads. Adding a CPU tier to that
baseline captures much of the same benefit with *no policy change at all*.
Mechanism: the smart policy never demotes adapters (it demotes cheap-to-restore
KV instead), so its adapters never cascade to disk — 10 disk loads regardless of
tier size, versus 107 for the naive policy at the tightest setting.
Practical reading: CPU RAM is single-digit $/GB, GPU HBM is ~$100+/GB amortised.
**If you can afford the RAM, buy RAM. If you can't, the policy is what saves you.**

**Phase 4** models the two independent GPU engines (compute and copy) that
phases 1-3 wrongly charged serially. Prefetch uses the compute window for I/O.
No prediction is involved — only adapters whose requests are already queued.
This ships in production as LoRAX's "Adapter Exchange Scheduling" and LMCache's
`PrefetchController`.

## A negative result worth keeping

**Paged (partial) KV eviction loses to atomic eviction.** Trimming blocks off
conversations to free exactly the bytes needed produced *3x more eviction
events*, because the cache then sits permanently 100% full. Atomic eviction —
dropping a whole 375MB conversation to free 20MB — accidentally buys **slack**.

This is why no production system does fine-grained eviction: vLLM and SGLang
allocate KV in fine-grained pages but *evict at request granularity*. The
ConServe paper names the same tension ("coarse-grained fate-sharing" trading
reclamation latency against restoration cost).

Note also that **paging's real win is allocation, not eviction** — it eliminates
internal fragmentation from pre-allocating for max sequence length. This
simulator allocates exactly `tokens x 0.125 MB`, so it already assumes paged
allocation for every policy.

## Calibration

| constant | value | basis |
|---|---|---|
| KV size | 0.125 MB/token | 8B model, GQA, fp16: 2 x 32 layers x 8 heads x 128 dim x 2 bytes |
| adapter size | 20 MB | rank-16 LoRA on an 8B model |
| prefill / recompute | 0.15 ms/token | compute-bound, parallel across prompt |
| decode | 12 ms/token | bandwidth-bound; ~14GB of weights read per step |
| adapter swap (cold) | 300 ms | from disk / object storage |
| adapter swap (warm) | 30 ms | from CPU RAM |
| PCIe transfer | 0.105 ms/MB | 40GB/s bandwidth **plus** ~10us dispatch per 128KB chunk |

The PCIe figure matters: at vLLM's 128KB swap granularity, per-call dispatch
overhead *dominates* bandwidth. Using the naive 40GB/s number (0.025 ms/MB)
underestimates transfer cost by 4.2x. Correcting it did not change phase 3's
conclusion (swap is still ~11x cheaper than recompute) but it is the difference
between copying a spec sheet and modelling the mechanism.

## Methodology notes

- Every reported figure is a **mean over 5-24 seeds**. Phase 4's effect was
  *not* distinguishable from noise at 8 seeds and was at 24. Single-run
  comparisons in this problem are unreliable.
- Policies within a phase share identical `choose_next` scheduling and differ
  **only** in eviction, so results isolate the eviction decision.
- Operating points are named (over-provisioned / cost-optimised /
  aggressively-packed) rather than quoted in raw MB, because what transfers
  between deployments is the **oversubscription ratio**, not the absolute pool size.
- "Stale KV" means *resident but not currently usable because its adapter was
  evicted*. It is not permanently wasted — roughly 60% of it gets rescued when
  the adapter returns — so it is an opportunity cost, not a loss.

## Limitations (deliberately out of scope)

| not modelled | what real systems do | expected effect |
|---|---|---|
| reclaim pool | vLLM V1 frees blocks lazily; a request that returns before its space is reused pays **nothing** | would shrink all gaps further; likely helps under slack, useless under pressure |
| prefix caching / RadixAttention | SGLang matches longest shared prefix at any tree depth; 20-40% lower TTFT on multi-turn | would cut recompute cost sharply |
| cross-conversation sharing | shared system prompts and tool definitions are byte-identical across users | each conversation's KV is private here |
| remote KV tiers | LMCache backends: Redis/Valkey, Mooncake, InfiniStore, S3, NVMe | adds a fourth tier below CPU |
| multi-node routing | NVIDIA Dynamo / vLLM router, consistent hashing for cache locality | biggest real-world lever; invisible to a single-node model |
| continuous batching | one persistent large batch, not batch-at-a-time | this simulator saturates ~3-5 req/s where real systems sustain 8-16 |
| KV compression | MLA (DeepSeek, ~90% reduction), FP8 KV (~50%, <0.5% quality loss) | shrinks the boulders |
| mixed-rank adapters | fast kernels assume uniform rank | serialization cost |

## References

- **ELORA** (HPCA 2026) / **FastLibra** (arXiv 2505.03756) — dependency-aware
  cache manager + performance-driven swapper. *Same work, two versions; the
  claimed TTFT reduction changed from 63.4% to 45.7% between them.*
- **S-LoRA** (arXiv 2311.03285) — unified paged memory pool; discards history KV
- **vLLM / PagedAttention** — paged allocation; V1 default preemption is RECOMPUTE
- **SGLang / RadixAttention** — radix-tree prefix reuse
- **Predibase LoRAX** — Dynamic Adapter Loading, Tiered Weight Caching, Adapter
  Exchange Scheduling
- **LMCache** (arXiv 2510.09665) — KV cache layer between engines and tiered storage
- **ConServe** — request-granularity eviction and coarse-grained fate-sharing
- **FastSwitch** — swap granularity and cudaMemcpyAsync dispatch overhead

## Next step

Validate against real vLLM on a rented GPU: serve a small model with several
adapters, sweep `--gpu-memory-utilization` and adapter count, measure real TTFT,
and compare against this simulator's prediction for the same configuration.
