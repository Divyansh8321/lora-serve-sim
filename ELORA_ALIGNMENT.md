# ELORA, exactly — and where this simulator differs

Source: a fresh read (2026-10-11) of the HPCA 2026 camera-ready
([cse.hkust.edu.hk/~weiwa/papers/elora-hpca26.pdf](https://cse.hkust.edu.hk/~weiwa/papers/elora-hpca26.pdf))
and the earlier FastLibra preprint ([arXiv 2505.03756](https://arxiv.org/abs/2505.03756)).
Section and equation numbers below are from the HPCA version. **No code has been
released**: the paper links none, and no repository could be found. It is "implemented on
vLLM with an extra 7,856 lines of Python and 1,766 of C++" (§VII).

Every earlier comparison in this repo was against our *own reading* of ELORA.
This document pins down what the paper actually specifies, so that
future comparisons are against ELORA rather than against a strawman.

## 1. The design, as specified

### Dependency-aware cache manager (§V)

- **One tree.** A virtual root sits at the top. **LoRA nodes are on the second layer**,
  and each LoRA's KV caches form a subtree under it (a trie keyed by token-sequence
  hash, "similar to SGLang").
- **Nodes are fixed-size memory blocks**, for both LoRAs and KV. A LoRA is split into
  blocks along the rank dimension. The memory-overhead figure implies **16 MB blocks**
  ("232 bytes per 16 MB block", §VIII-L).
- **Swap-out only from leaves in GPU memory. Swap-in only from roots of
  paths in host memory.** A LoRA is the *parent* of its KV, so it cannot
  leave the GPU while any of its KV is resident. **Invalid (stale) KV is
  impossible by construction.** This is the core of the "dependency"
  mechanism. It is a structural invariant, not a scoring rule.
- **Eviction means a swap to host memory** (256 GB host RAM, PCIe 5.0 at 128 GB/s).
  KV blocks are only *deleted* when GPU and host memory are both full (§VII).
- **Swaps are asynchronous** (Torch CUDA streams). A query waiting on its own
  swap-in doesn't block anyone else. Reported swap overhead is ≤ 0.47 ms.

### Performance-driven cache swapper (§VI)

Every **100 ms** the swapper scores every node:

```
Low_lora      = Σ_i [ 1 − (1 − prob_i)^BS ]                    (Eq. 3)
                prob_i = LoRA i's usage-frequency probability (from the tree)
                BS     = recent batch size, last 5 s

LoRA_Eval_i   = min(1, Now_LoRA_i / Low_lora)                  (Eq. 4, HPCA)
              = max(1, Low_lora / Now_LoRA)                    (Eq. 4, FastLibra preprint)
                Now_LoRA_i = number of LoRAs in GPU after swapping node i

Retain_Eval_i = cost_i × visit_i × (1 − sigmoid(t_i))          (Eq. 5)
                cost_i  = transfer cost (PCIe bandwidth × size)
                visit_i = visit-frequency probability
                t_i     = now − last use time

Eval_i        = LoRA_Eval_i × Retain_Eval_i                    (Eq. 6)
```

- **Swap-out.** HPCA: "if the GPU memory is full". FastLibra: proactively above **95%**,
  down to **70%**. Candidates are GPU **leaves**, sorted by *increasing* Eval and
  evicted one at a time. Eval is recomputed after each LoRA swap, because
  `Now_LoRA` changes.
- **Swap-in (prefetch).** When GPU memory is below **70%** used, candidates are the
  **host-memory roots** of each path, sorted by *decreasing* Eval. Prefetch is
  causal: it uses frequency and recency only, never future arrivals.

### Evaluation setup (§III-B, §VIII-A)

| item | value |
|---|---|
| models / GPUs | Llama3-8B on 1 H800, Llama2-34B on 4, Llama3-70B on 8 (80 GB each) |
| LoRAs | 20 / 50 / 100 per model, rank 32 or 64, random HF adapters |
| chatbot | LMSYS-33k. **LoRA assigned by model name** (same as our loader). "Responses using full user history". Rates scaled proportionally, preserving the pattern |
| translation | OPUS-100, one LoRA per language pair. Arrivals from the Microsoft Azure Function trace (top-n functions → n LoRAs) |
| personal agents | Taskmaster, with the same MAFT arrival mapping |
| baselines | **vLLM** (static LoRA memory ratio 0.2, LRU, prefix caching), **S-LoRA** (unified pool, no history-KV reuse). SGLang excluded (multi-LoRA could not reuse history KV) |
| metrics | mean TTFT and TPOT. **Peak load = max QPS with mean TTFT < 500 ms**. 10 rates from 0 to peak, 20 repeats |
| headline | −45.7% TTFT, −37.8% TPOT, +78.9% peak load vs vLLM. vLLM shows 42.4% invalid KV |
| ablations | WOM (no tree, swapper kept) 1.51× TTFT, 48.6% invalid. WOS (tree + plain LRU) 1.42×. Dropping single cost terms (WOL/WOC/WOV/WOU) 1.19–1.25× |
| vs oracle vLLM | brute-force best LoRA ratio: ELORA still −38.7% TTFT |

## 2. Where this simulator differed (before `phase7_elora.py`)

> **Status 2026-10-11:** the alignment plan in §4 is built as `phase7_elora.py`
> (it replaced the heuristic phase 7, now `archive/phase7_heuristic_swapper.py`).
> The table below describes phases 6 and the archived phase 7. §5 lists what
> `phase7_elora.py` still does differently from the paper.

| aspect | ELORA | this repo (as of 2026-10-11) | impact |
|---|---|---|---|
| **eviction semantics** | swap to host memory; delete only when host full | **dropped**; KV must be recomputed. Phases 6–7 have no host tier | Makes every eviction expensive, so it penalises proactive eviction and over-protects KV. Probably the largest single gap |
| **LoRA miss** | async swap-in from host over PCIe (~1.4 ms for 180 MB at 128 GB/s) | **300 ms** cold load, every time (phases 6–7) | Overweights adapter churn ~200× |
| **dependency mechanism** | structural: LoRA is the parent of its KV, leaf-only eviction | adapters live outside the tree in a separate dict and can be evicted while their KV is resident. `dep-aware` is a ranking heuristic | We never tested ELORA's actual mechanism |
| **granularity** | fixed 16 MB blocks | whole conversation / whole prefix per node | Coarser eviction; size-based heuristics behave differently |
| **cost model** | multiplicative Eq. 3–6 | additive `0.01·mb + hits + 5·recency + 10·floor + log2(sharers)` | Different ranking. `swap-full` is not ELORA's swapper |
| **swapper triggers** | out: full (HPCA) or >95%→70% (FastLibra); in: <70% | out: >92%→75% or full; in: <60% | Thresholds differ |
| **prefetch** | causal, by Eval, from host memory | **reads future arrival times** from the workload, and only fires before each (adapter, group)'s first request | Oracle and mostly inert |
| **vLLM baseline** | static ratio 0.2, LRU, prefix caching | `max_loras × adapter` slab; vLLM-faithful CPU LoRA cache only since this audit; no prefix caching in the separate-pool phase | Baseline not yet comparable |
| **workload** | real prompts, full history, scaled real rates, 20–100 LoRAs | prompt tokens only since this audit; synthetic runs use 12 adapters, 40–60 conversations | Much smaller scale |
| **metric** | mean TTFT/TPOT; peak QPS at TTFT < 500 ms | p50/p95 end-to-end latency, p50 TTFT | Not directly comparable |

## 3. What the paper leaves unspecified

Any reimplementation, including the planned GPU one, must choose these and say so:

1. **The direction of Eq. 4.** The two versions disagree. HPCA's `min(1, Now/Low)`
   is ≤ 1, so it *shrinks* a node's value when too few LoRAs are loaded.
   FastLibra's `max(1, Low/Now)` is ≥ 1, so it *boosts* it. What `Now_LoRA_i` means for
   a KV node (whose swap doesn't change the LoRA count) isn't defined either.
2. **The units of `t_i`** in `sigmoid(t_i)`. For t ≥ 0, `1 − sigmoid(t)` lies in
   (0, 0.5], and its slope depends entirely on the unit (ms vs s), which is never stated.
3. **The window behind `visit_i` and `prob_i`** ("from recorded data").
4. **`cost_i` for fixed-size blocks.** If every node is one 16 MB block, `cost_i`
   is the same for all of them, unless LoRA and KV blocks differ in size.
5. **What "full" means** in the HPCA swap-out trigger.
6. **How 50 and 100 LoRAs are built** from LMSYS's ~20 model names.
7. **LMSYS turn timing.** The dataset has one timestamp per battle, and the paper
   doesn't say how turns are spaced.
8. **"LoRA memory ratio 0.2" in vLLM.** vLLM sizes its LoRA slab from `max_loras`
   and `max_lora_rank`, not from a ratio flag. The exact baseline config needs to
   be reconstructed.
9. **What counts as "invalid KV" in vLLM (42.4%).** In vLLM V1, blocks of finished
   requests stay hashed in the free queue and are reclaimed lazily on demand.
   Whether ELORA counted those as invalid matters: if so, they occupy memory that
   is free in practice. This is measurable on a GPU.

## 4. Alignment plan (a "phase 8" built for fidelity)

1. **Tree:** virtual root → LoRA node → KV blocks (16 MB), with leaf-only swap-out and
   root-only swap-in, so the dependency invariant is enforced structurally.
2. **Host tier for both kinds of node**, priced at PCIe (`pcie_ms_per_mb`), async.
   Delete only when host memory is full. Cold disk loads only on the first-ever use.
3. **Eq. 3–6 implemented literally**, with a `--eq4 {hpca,fastlibra}` flag and an
   explicit `--t-unit` for the sigmoid. `Low_lora` from a 5 s batch-size window.
4. **Triggers** selectable as `hpca` (full / <70%) or `fastlibra` (>95%→70% / <70%).
5. **Baselines:** vLLM-faithful (static split, LRU, prefix caching, CPU LoRA cache,
   lazily reclaimed cached blocks), S-LoRA (unified pool, no history KV),
   ELORA-WOM (no tree, same swapper), ELORA-WOS (tree + LRU).
6. **Workload and metric:** real prompt tokens with full history, 20/50/100 LoRAs,
   rates swept to find **peak QPS at mean TTFT < 500 ms**, mean TTFT/TPOT reported
   alongside the percentiles.

Even then, the simulator can only show whether the *mechanism* reproduces. The
magnitudes need the GPU runs (`VALIDATION_PLAN.md`).

## 5. What `phase7_elora.py` implements, and where it still departs

Implemented as specified: 16 MB blocks; a LoRA → shared prefix → per-conversation
KV-chain tree; swap-out from GPU leaves only, swap-in from host roots only;
evicted KV to host memory (3.2× GPU), deleted only when host is full; async
host↔GPU transfers on one PCIe channel each way; Eq. 3–6 with both Eq. 4
readings (`--eq4`), both swap-out triggers (`--trigger`), a free `t_i` unit
(`--t-unit-ms`), a 5 s batch-size window and causal prefetch below 70% use.
Baselines: vLLM (fixed slab = batch cap, LRU, prefix caching, KV deleted),
vLLM + CPU KV offload, S-LoRA (no history reuse), ELORA-WOS, ELORA-WOM,
ELORA without the host KV tier. Peak load (`--peak`) uses ELORA's definition.

Still different:

- A LoRA is one node, not rank-split 16 MB blocks.
- GPU memory is freed as soon as a swap-out starts, not when the copy ends.
- Host RAM holds every adapter after its first disk load.
- `visit_i` and `prob_i` are exponentially decayed counts (half-life
  `--visit-halflife-ms`, default 10 s). The paper does not give the window.
- `cost_i` is the PCIe transfer time of the node. For KV blocks it is the
  same for every block, as the paper's fixed-size design implies.
- vLLM's prefix caching is modelled at 16 MB-block granularity (vLLM uses
  16-token blocks).
- The workload is synthetic chat traffic (or the three real traces through
  `run_on_real_traces.py`), not ELORA's exact LMSYS/OPUS/Taskmaster + Azure mix.

Baseline note: the paper's vLLM baseline is vLLM *plus* LRU offload of
history KV and LoRAs to main memory (FastLibra §6.1), with a 0.2 LoRA
memory ratio. That is our `vllm-offload` at the default `max_loras`.

Result: ELORA beats an oracle-tuned vLLM by 21–87% TTFT. All of it comes from
the host-memory KV tier; vLLM with CPU KV offload matches it, and the tree and
the swapper change nothing measurable. Against `vllm-offload` at the paper's
0.2 ratio ELORA wins 77–94% (20–100 adapters); against `vllm-offload` with a
tuned slab, ~1% steady and 14–17% under 5× bursts. See `phase7_elora.py`'s docstring and
the README.

