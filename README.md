# lora-serve-sim

**A discrete-event simulator of multi-tenant LoRA serving, used to take apart
ELORA (HPCA 2026): where does its speedup actually come from?**

One base model, many LoRA adapters, many multi-turn conversations, one GPU.
The simulator models GPU memory (adapters + KV cache), host memory, PCIe
transfers, continuous batching and prefix caching. It compares **vLLM**,
**vLLM + CPU KV offload**, **S-LoRA** and a from-the-paper reimplementation of
**ELORA**, including its ablations.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/simulator-stdlib%20only-green)
![gpu](https://img.shields.io/badge/GPU-not%20required-lightgrey)

---

## TL;DR

> ELORA's reported gains reproduce in simulation, but against its own
> baseline they come almost entirely from **two things that are not ELORA's
> novel parts**: keeping evicted KV in **CPU memory**, and vLLM's adapter
> memory being **fixed at 20%**. Give vLLM CPU KV offload and a well-sized
> adapter slab and ELORA's lead shrinks to **~1% (steady) to 14–17% (bursty)**.
> ELORA's dependency tree and cost-model swapper change TTFT by **< 1%**.

## Key results

Phase 7, chat workload (80 conversations × 5 turns, ~200 prompt / ~200 reply
tokens, 400-token shared system prompt), 6 GB GPU budget, mean of 5 seeds.
Mean time to first token (TTFT):

| design | 20 adapters, steady | 50 adapters, 5× bursts | 100 adapters, 5× bursts |
|---|---:|---:|---:|
| vLLM + offload, adapter memory 20% (**ELORA's baseline**) | 2,575 ms | 13,355 ms | 19,605 ms |
| vLLM + offload, adapter memory tuned | 608 ms | 1,364 ms | 1,633 ms |
| **ELORA** (reimplemented) | **600 ms** | **1,128 ms** | **1,387 ms** |
| ELORA without the swapper (paper: 1.42× slower) | 601 ms | 1,134 ms | 1,390 ms |
| ELORA without the dependency tree (paper: 1.51× slower) | 600 ms | 1,128 ms | 1,387 ms |
| ELORA without CPU memory | 1,540 ms | 8,421 ms | 9,636 ms |

| claim in the paper | here |
|---|---|
| −45.7% TTFT vs vLLM | −77% to −94% vs the same baseline; **−1% to −17% vs a tuned one** |
| +78.9% peak load vs vLLM | +97% vs the 20% setting; **+19% vs tuned vLLM; 0% vs vLLM + offload** |
| WOS / WOM ablations 1.42× / 1.51× | **1.00× / 1.00×** |
| vLLM wastes 42.4% of KV on "invalid" entries | reproduced (up to 76%), but it costs **~nothing**: the space is reclaimed by LRU when needed |

On three real datasets (LMSYS-Chat, Taskmaster, Mooncake, with BurstGPT
timing) the designs tie or ELORA loses, mainly because these public traces have
no real gaps between a conversation's turns. Details:
[`REAL_DATA_FINDINGS.md`](REAL_DATA_FINDINGS.md).

**Real-GPU check:** a harness that replays the same workload token-for-token
against `vllm serve` on one A100 (vLLM vs vLLM + native CPU KV offload, three
adapter-slot settings) is in [`validation/GPU_RUN.md`](validation/GPU_RUN.md).

## Quick start

```bash
git clone https://github.com/Divyansh8321/lora-serve-sim && cd lora-serve-sim
python phase7_elora.py                                  # ELORA vs vLLM / S-LoRA / ablations (~40 s)
python phase7_elora.py --n-adapters 50 --burst 5        # the paper's adapter counts
python phase7_elora.py --hw elora-aggressive --peak     # H800-class timings, peak load
python run_on_real_traces.py --workload taskmaster      # real traces (where to get the data: real_traces.py docstring)
```

Pure Python standard library. No GPU, no model weights, no network.

## How it is built: seven phases, one variable at a time

| phase | adds | finding |
|---|---|---|
| [1](phase1_adapter_caching.py) | adapter cache | cold adapter loads cost 3–6% p50; warm ~0% |
| [2](phase2_separate_pools.py) | KV cache in a separate pool (vLLM's layout) | shrinking the adapter slab leaves up to 44% of KV "stale" (its adapter is gone) |
| [3](phase3_cross_pool_signal.py) | a signal between the two pools | stale KV drops, latency does not |
| [4](phase4_unified_pool.py) | one unified pool | evicting KV before adapters: −7% to −24% p50 as memory tightens |
| [5](phase5_continuous_batching.py) | continuous batching, realistic chat | separate ≈ unified (±5%); vLLM's real cost is the `max_loras` batch cap |
| [6](phase6_radix_prefix.py) | radix prefix tree, shared system prompts | plain LRU is never meaningfully beaten |
| [7](phase7_elora.py) | **ELORA as specified**: LoRA→prefix→KV tree, 16 MB blocks, host-memory tier, async PCIe, Eq. 3–6 swapper | table above |

Every choice the paper leaves open (Eq. 4's direction, the swap trigger, the
unit of `t_i`, the visit window) is a flag, and none of them moves the result
by more than 1%. See [`ELORA_ALIGNMENT.md`](ELORA_ALIGNMENT.md).

## Repository map

| path | what |
|---|---|
| `core.py` | requests, conversations, workload generators, hardware profiles |
| `phase1_…py` – `phase7_elora.py` | the seven simulators (each runnable, results in its docstring) |
| `real_traces.py`, `run_on_real_traces.py` | loaders for LMSYS / Taskmaster / Mooncake / BurstGPT and the real-trace runs |
| `validation/` | real-GPU harness: adapter generator, token-exact workload driver, run + compare scripts |
| `sweeps/`, `archive/` | parameter sweeps; the first (heuristic) phase 7 |
| [`ELORA_ALIGNMENT.md`](ELORA_ALIGNMENT.md) | ELORA's equations, the ambiguities, and every remaining departure |
| [`REAL_DATA_FINDINGS.md`](REAL_DATA_FINDINGS.md) | real-trace results and caveats |
| [`AUDIT.md`](AUDIT.md), [`ROADMAP.md`](ROADMAP.md) | change log of fixes; history and plans |

## References

- J. Shi et al., *ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA
  LLM Serving*, HPCA 2026 ([pdf](https://cse.hkust.edu.hk/~weiwa/papers/elora-hpca26.pdf));
  earlier version *FastLibra*, [arXiv:2505.03756](https://arxiv.org/abs/2505.03756).
- vLLM, S-LoRA, SGLang (RadixAttention), Mooncake, BurstGPT; full list at the
  [end of this file](#full-reference-list).

---

# Detailed write-up

**Hardware profiles.** Phases 5–7 take `--hw {ours,a100,elora,elora-aggressive,
elora-conservative}`: `ours` is a single ~A10 (PCIe 40 GB/s); `elora` is the
H800 regime (PCIe 128 GB/s, async overlapped swaps, evict-on-full); `a100` is a
rough A100 guess used only to predict the GPU run.

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

Merge the pools and the eviction decision gains a choice that did not
previously exist: *which type* to evict, a 180 MB adapter (`ADAPTER_MB`,
rank-32 all-linear, **ELORA's rank**) or a conversation's KV. All three
policies run in the same unified pool. (`adapter-first` was printed as
"separate" in earlier versions. It never meant separate pools.)

Original workload, no user prompts (a conversation's KV averages ~25 MB):

| pool MB | `adapter-first` | `kv-first` | `joint` | stale KV (`adapter-first`) |
|---|---|---|---|---|
| 3600 (fits all) | 1052 | 1052 | 1052 | 0% |
| 2500 | 1053 | 1052 | 1052 | 2.2% |
| **1600 (packed)** | 1127 | **1045** | **1045** | **21.9%** |

With real-sized prompts (`--input-tokens 600`, a conversation's KV ~400 MB):

| pool MB | `adapter-first` | `kv-first` = `joint` | change |
|---|---|---|---|
| 20000 | 1781 | 1781 | 0% |
| 12000 | 1845 | 1781 | −3.5% |
| 8000 | 2054 | 1799 | −12.4% |
| 5000 | 2567 | 1963 | −23.5% |

(p50 end-to-end latency in ms, mean of 5 seeds.) **Evicting KV before adapters
wins, and wins more with realistic prompts.** An adapter is shared, so evicting
it charges a reload to every conversation that uses it next. `kv-first`
(crude ordering) is *identical* to `joint` (dependency-aware). Caveat: phase 4
runs batch-at-a-time, so p95 is ~10–20 s of queueing that a continuous
batcher would not have. Use it to see the type-choice, and phase 5 for latency.

### Phase 6 — does the prefix tree change the verdict?

Phase 6 adds a **RadixAttention prefix tree** (KV keyed by token prefix,
shared system-prompt nodes, LoRA-rooted subtrees) and a workload where
conversations share a system prompt. Three policies on a 3000 MB pool:

| prefix tokens | `lru-leaf` | `ordering` | `dep-aware` |
|---|---|---|---|
| 0 | 689 | 689 | 689 |
| 400 | 707 | 731 | 714 |
| 800 | 796 | 841 | 785 |
| 1600 | 1122 | 1216 | 1161 |

(p50 ms, mean of 5 seeds, re-run after the audit)

**Plain LRU (SGLang's own rule) is never meaningfully beaten.** It ties or
beats `ordering` everywhere and is within 1–3% of our `dep-aware`. The 1800 MB
squeezed pool and `--hw elora-aggressive` both give all three within ~2%.

**With realistic prompts** (`--input-tokens 600`, pools 6000/10000 MB),
`dep-aware` is **16–54% worse** than `ordering`, and `lru-leaf` ≈ `ordering`.
Our heuristic's "small nodes first" rule evicts many live conversations once
KV is realistically sized.

**Withdrawn:** an earlier version reported that `lru-leaf` "collapses ~40×"
(p50 28,553 ms at 1600 tokens), that "prefix-aware ordering does the heavy
lifting", and that ELORA-WOM ≈ our collapsing `lru-leaf`. All three came
from a bug: the leaf test ignored residency, so shared prefixes could never be
evicted. ELORA-WOM also keeps ELORA's cost-model swapper, so it is not blind
LRU in any case.

**What this does not test:** ELORA's actual dependency mechanism. In
ELORA, a LoRA is the *parent* of its KV, so leaf-only eviction makes stale
KV impossible, and eviction swaps to host memory instead of destroying.
Neither is modelled here (`ELORA_ALIGNMENT.md`).

### Phase 5 — the pathology under the real execution model

Phases 2–4 are batch-at-a-time. ELORA and production systems use **continuous
batching**: one rolling batch stepped one token at a time, with waiting requests
spliced in. `--max-loras` is what it is in vLLM: the adapter slab's slot count
*and* the max number of distinct adapters in one batch. An adapter is pinned
while any of its sequences is running.

The default workload is now **chat**: 20 adapters (ELORA's smallest setup),
~200 new prompt tokens and ~200 reply tokens per turn, 15 s between turns,
6000 MB of GPU memory (keeping every conversation would need ~19 GB of KV).
Both architectures get the same total memory and the **same host-RAM adapter
cache** (every adapter, as an operator would set `--max-cpu-loras`).

| max_loras | slab % | SEPARATE (vLLM) mean / TTFT | UNIFIED same cap | stale KV (separate) |
|---|---|---|---|---|
| 20 | 60% | 4570 / 188 | 4280 / 156 | 0% |
| 10 | 30% | 4216 / 175 | 4303 / 190 | 6% |
| 6 | 18% | 4923 / 977 | 5149 / 1098 | 29% |
| 4 | 12% | 7815 / 3950 | 8179 / 4223 | 46% |

(mean latency / mean TTFT in ms, mean of 5 seeds.) Unified with no batch cap:
4280 / 156.

- **Separate and unified are within ~5% at every slab size.** The separate
  pool is sometimes faster.
- **Stale KV reaches 46% and costs nothing measurable.** It sits in otherwise
  unused KV space and is reclaimed by LRU when needed.
- **The cliff at small `max_loras` is the batch cap, and it hits both.** With
  200-token replies, a request for a 5th adapter waits seconds for a batch
  slot. Only a unified pool can lift the cap (no fixed slab), and that wins big
  (TTFT 3950 → 156 ms) only against a badly sized vLLM. A tuned vLLM
  (`max_loras` 10) matches it. The no-cap row ignores the extra kernel cost of
  more adapters per batch, so it is optimistic.
- On `--hw elora-aggressive` sequences finish ~3× sooner, the cap barely binds,
  and the two are within ±4% everywhere. `--workload toy` (the original
  workload) gives separate 691 → 763 ms and unified 691 → 720 ms as
  `max_loras` goes 12 → 3.

**Withdrawn:** +66% (then +22%) separate-pool penalties and the "capacity
fungibility" explanation. The first came from giving the separate pool extra
KV memory and no CPU adapter cache. The second came from the reverse
asymmetry: the unified pool had no CPU adapter cache, so its evicted adapters
cost 300 ms disk loads.

### Phase 7 — ELORA as the paper specifies it

`phase7_elora.py` replaces the original phase 7, which was our own
approximation (evicted KV deleted, every adapter miss a 300 ms disk load,
an invented additive score, a prefetcher that read future arrivals). It is
kept at `archive/phase7_heuristic_swapper.py`.

The rebuild follows ELORA §V–VII: GPU memory in 16 MB blocks; a tree of
LoRA → shared prefix → each conversation's chain of KV blocks; swap-out only
from leaves and swap-in only from roots (so a LoRA cannot leave while its KV
is on GPU); evicted KV goes to **host memory** (3.2× GPU, ELORA's ratio) and
comes back over PCIe **asynchronously**; and the 100 ms swapper scores nodes
with Eq. 3–6 (`ELORA_ALIGNMENT.md` has the formulas and every choice the
paper leaves open, exposed as flags).

Designs: `vllm` (fixed slab, LRU, prefix caching, KV deleted on eviction),
`vllm-offload` (the same plus CPU KV offload), `slora` (unified, no history
reuse), `elora`, `elora-wos` (tree + LRU, no swapper), `elora-wom` (no
dependency tree), `elora-nohost` (evicted KV deleted). "Oracle vLLM" is the best
`max_loras` of {6 (ELORA's 0.2 ratio), 10, 14, 20}.

Change in mean TTFT vs oracle vLLM (chat workload with a 400-token shared
system prompt, 6000 MB GPU, mean of 5 seeds):

| design | ours, steady | ours, burst ×5 | H800-class, steady | H800-class, burst ×5 |
|---|---|---|---|---|
| `elora` | **−62%** | **−87%** | **−21%** | **−36%** |
| `elora-wos` | −61% | −87% | −21% | −36% |
| `elora-wom` | −62% | −87% | −21% | −36% |
| `elora-nohost` | −1% | −7% | −7% | −22% |
| `vllm-offload` (tuned) | −61% | −85% | −21% | −36% |
| `slora` | +106% | +42% | +23% | +15% |

- **Peak load** (H800-class, highest req/s with mean TTFT < 500 ms): ELORA
  5.99, tuned vLLM 5.05 (+19%), vLLM at the 0.2 ratio 3.04 (+97%; paper
  +78.9%), vLLM + offload 5.99.
- **ELORA beats a tuned vLLM by about what the paper says** (paper: −38.7% vs
  oracle vLLM; −37.8% TPOT, here −5% to −36%).
- **The win is the host-memory KV tier.** Remove it (`elora-nohost`) and most
  of the win goes. Add only it to vLLM (`vllm-offload`) and vLLM matches ELORA.
  The one exception is tight memory plus bursts on slow hardware, where the
  unified pool is worth ~15% (TTFT 920 vs 794 ms).
- **The tree and the swapper add nothing measurable.** `elora`, `elora-wos`
  and `elora-wom` are within 1% in every cell, and that holds for both
  directions of Eq. 4, both swap-out triggers, `t_i` in ms / s / minutes, and
  6000 or 12000 MB. Without the tree, stale KV is still ~0%: a 180 MB LoRA is
  expensive to move, so it is almost never the eviction victim. Prefetch rarely
  fires, because with history kept the GPU is almost always over 70% full.
- **This does not reproduce ELORA's ablations** (WOM 1.51×, WOS 1.42×). Either
  their workloads stress something this one does not, or their ablations change
  more than the paper describes.

**Archived v1 results** (heuristic swapper, deleted KV): the swapper was −5% to
−22% at our engine and +0.4–2.9% at ELORA's with zero prompts, and plain
reactive LRU beat it at both engines with realistic prompts. Those runs had no
host tier, which is the one mechanism that turns out to matter.

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

**Hardware profiles (`core.py`, used by phases 5–7 via `--hw`).** The
calibration table above is the `OURS` profile. `ELORA_AGGRESSIVE` /
`ELORA_CONSERVATIVE` model ELORA's H800:

| field | `OURS` | `ELORA_*` | source |
|---|---|---|---|
| PCIe transfer | 0.105 ms/MB | 0.008 ms/MB | PCIe 5.0 @ 128 GB/s (ELORA Table II) vs our ~40 GB/s |
| decode / token | 12 ms | ×0.18 (agg) / ×0.35 (cons) | H800 HBM ~3.35 TB/s vs our implied ~600 GB/s; **swept as a band** since it's the biggest lever |
| prefill / token | 0.15 ms | ×0.13 / ×0.30 | same, FLOPs-scaled |
| adapter cold load | 300 ms | 300 ms | disk/S3 latency — not a GPU property, held constant |
| adapter warm load | 30 ms | 2.3 ms | PCIe-bound, scales with bandwidth |
| swap engine | sync, charged prefetch, evict-at-92% | async overlapped, evict-on-full | ELORA §VI-C, §VII ("no extra swapping overhead") |

`mb_per_token` and `adapter_mb` are model architecture — **identical in every
profile**. Absolute latencies at `--hw ours` are not directly comparable to
ELORA's; the *shape* of each curve, and *whether a policy's sign flips between
profiles*, is what the comparison rests on.

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

1. **One synthetic workload family for phases 2–5.** 40 conversations, 12
   adapters, Zipf skew 1.0, 5 turns, 6 s gaps, Poisson arrivals. Phase 6's
   shared prefixes and phase 7's bursts + drift were originally synthetic
   stand-ins for real traffic — **now also tested against real data** (see
   #8 below and `REAL_DATA_FINDINGS.md`), which found the real trace is far
   *more* long-tailed (7,373 distinct adapters, most used once) than our
   synthetic Zipf(1.0) assumption, and that real bursts move phase 7's numbers
   much less than our synthetic ones did.
2. **Batch-at-a-time in phases 1–4.** Phases 5–7 add continuous batching.
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
7. **The `ELORA_*` hardware profiles are estimates.** PCIe is a solid figure
   (128 GB/s, from the paper); the decode/prefill scaling is a swept
   *aggressive/conservative band*, not a measured value, now also confirmed
   continuous with `--sweep-scale` (never crosses zero at any speed tried —
   not just the two endpoints). And even the H800 profile keeps our
   batch-at-a-time step model — so phase 7's "swapper flips positive at
   ELORA's engine" is the *direction* ELORA reports, at a plausible constant
   band, not a reproduction of their 1.42×.
8. **Real-trace results (see `REAL_DATA_FINDINGS.md`) mix real systems for
   two of three datasets.** BurstGPT's arrivals are overlaid onto Mooncake's
   and Taskmaster's conversations (each a different real system) — the best
   available combination, not one ground-truth trace. LMSYS is the exception:
   real timing and real identity both come from the same source, no overlay.
   Adapter identity is inferred for Mooncake, genuine for Taskmaster and
   LMSYS. `lru-leaf`'s multi-hour real-trace latency is a genuine simulator
   output (verified, not a bug) but not a literal production forecast — a
   real system would shed load long before that point.
9. **We could not obtain or reconstruct ELORA's third dataset** (OPUS-100
   translation + their specific, unpublished Azure Function trace slice) —
   untested entirely.
10. **No public dataset we found carries BOTH real prefix-sharing structure
    AND real adapter identity.** Mooncake has the first and Taskmaster/LMSYS
    the second. After the audit, staleness is ~0% on all three. The earlier
    "0.00% → 99.40%" prefix-injection control and Mooncake's "42.5%" were
    both the `lru-leaf` bug, not data properties.
11. **Intra-conversation turn gaps are ~1 ms on Taskmaster and LMSYS** vs the
    6,000 ms our synthetic workload uses. A data limitation for LMSYS (one
    timestamp per arena battle); **our own bug for Taskmaster** (placeholder
    per-turn values the BurstGPT overlay then preserved). Re-testing with
    realistic 6 s gaps held the headline numbers but moved adapter loads
    861 → 6,096 — it materially affects the workload and should be fixed
    before drawing further Taskmaster conclusions.
12. **Phase 6's `dep-aware` is a heuristic, and phase 7 is a reconstruction.**
    `phase7_elora.py` follows the paper, but ELORA's code is not public and the
    paper leaves several choices open (`ELORA_ALIGNMENT.md` §3). Phase 7 treats a
    LoRA as one node rather than rank-split blocks, frees evicted GPU memory
    as soon as the copy starts, and assumes host RAM holds every adapter.

## Limitations

| not modelled | what real systems do | planned? |
|---|---|---|
| continuous batching | one persistent rolling batch; adapters pinned to in-flight sequences | ✅ **phase 5** |
| prefix caching / RadixAttention | SGLang/ELORA match longest shared prefix at any depth | ✅ **phase 6** (coarse) |
| cost-model swapper | ELORA re-scores every cache node every 100 ms (Eq. 3–6) | ✅ **phase 7** (`phase7_elora.py`) |
| host-memory KV tier | ELORA swaps evicted KV to host RAM; LMCache / vLLM offloading do the same | ✅ **phase 7** (`elora`, `vllm-offload`) |
| bursty / drifting workload | Azure Function trace; adapter popularity shifts over time | ✅ **phase 7** (`make_bursty_workload`); ✅ real BurstGPT trace (`run_on_real_traces.py`) |
| real prefix-sharing structure | actual shared system-prompt overlap across users | ✅ real Mooncake trace (`run_on_real_traces.py`) |
| reclaim pool | vLLM V1 frees blocks lazily; a request returning before reuse pays nothing | — |
| remote KV tiers | LMCache backends: Redis/Valkey, Mooncake, NVMe, S3 | — |
| multi-node routing | biggest real-world lever; invisible to a single-node model | — |
| KV compression | MLA (~90% reduction), FP8 KV (~50%) | — |
| hardware regime | ELORA is on H800 (128 GB/s PCIe, 80 GB HBM); we're A10-class | ✅ **phases 5–7** `--hw` |

## Roadmap

The point of the phases below is *fidelity to what top systems actually do*, so
the comparison to ELORA is like-for-like instead of "our simplified model vs
their real one". Full detail in `ROADMAP.md`.

| phase | adds | why it matters | status |
|---|---|---|---|
| **5** | continuous batching (step clock, in-flight sequences, mid-stream admission, adapter pinning) | ELORA runs entirely under continuous batching; without it our staleness mechanism is structurally different from theirs | ✅ done — `phase5_continuous_batching.py` |
| **6** | RadixAttention prefix tree (shared system prompts, LoRA-rooted subtrees, prefix-aware eviction) | ELORA's dependency manager *is* a prefix tree | ✅ done — `phase6_radix_prefix.py`. Result (audited): plain LRU is never meaningfully beaten; our dependency heuristic is ±3% without prompts and worse with realistic prompts |
| **7** | ELORA as specified: LoRA-parent tree, 16 MB blocks, host-memory tier, async transfers, Eq. 3–6, vLLM / vLLM-offload / S-LoRA / WOS / WOM baselines | the first like-for-like test of ELORA's mechanism | ✅ done — `phase7_elora.py` (replaced the heuristic v1, now in `archive/`). Result: −21% to −87% TTFT vs oracle vLLM, all from the host KV tier |

## Full reference list

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
- **Mooncake FAST'25 trace** (github.com/kvcache-ai/Mooncake) — real anonymized
  request trace with prefix-block hash IDs for studying KV-cache sharing; used
  in `real_traces.py` / `REAL_DATA_FINDINGS.md`
- **BurstGPT** (github.com/HPMLL/BurstGPT, CC-BY-4.0) — 10M+ real ChatGPT/GPT-4
  request logs from Azure OpenAI with real arrival timestamps; used for real
  bursty-arrival timing in `real_traces.py`
