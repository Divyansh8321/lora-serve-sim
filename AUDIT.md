# Code audit (2026-10-11)

Bugs found in this simulator's own earlier versions, what was fixed, and how
each fix changed the results. Kept for the record; the README reports only
post-fix numbers.

An independent review re-derived each claim by instrumenting the code.

| # | finding | fixed in | effect on results |
|---|---|---|---|
| 1 | `lru-leaf` used structural `is_leaf()`; children are never pruned, so **shared prefixes could never be evicted** and filled the pool | `phase6_radix_prefix.py` (`LRULeaf`) | the "40× collapse", "ordering does the heavy lifting", the ELORA-WOM reconciliation, the Taskmaster "0% → 99.4% stale" test and Mooncake's "42.5% stale" **were all this bug** (old code reproduces 99.40% exactly; fixed: 0%). Mooncake stale is now 0.4% |
| 2 | **No user-prompt tokens** anywhere; `real_traces.py` read Mooncake's `input_length` and discarded it (median prompt 6,909 tokens), so KV was **~20× too small** | `core.Request.input_tokens`, workload generators (`mean_input_tokens`), phases 5–7 (`--input-tokens`), all three loaders | realistic prompts make `dep-aware` and the swapper clearly worse (see above) |
| 3 | Phase 5's separate pool got 4000 MB of KV **on top of** its slab (peak use 972 MB, zero KV evictions) and **no CPU adapter cache**, which is a config vLLM rejects (`max_cpu_loras ≥ max_loras`, default `= max_loras`, `vllm/config/lora.py`) | `phase5_continuous_batching.trial` defaults: memory-matched, vLLM-default CPU cache | separate-pool penalty +66% → **+22%**; TTFT knee 296 → 55 ms. The "admission stall" explanation was wrong (unified runs the same `max_loras` cap at 24 ms) |
| 4 | decode growth resized KV **without making room**, overrunning the pool (up to ~70 MB) | phases 5–7 | numbers move < 1–2% |
| 5 | phase 7: a request blocked on an async adapter transfer could be marked **finished unserved**, or wait for an unrelated arrival | `Phase7Sim.step` (now `archive/phase7_heuristic_swapper.py`) | latent; verified identical outputs on the shipped workloads |
| 6 | phase 7 prefetch **reads future arrivals** and only fires once per (adapter, group) | phase 7 rebuilt as `phase7_elora.py` (causal prefetch by Eq. 3–6); the old file is `archive/phase7_heuristic_swapper.py` | — |
| 9 | phase 5's unified pool had **no CPU adapter cache** while the separate pool did, so unified reloaded adapters from disk (300 ms) and separate from host RAM (30 ms) | both pools get the same host-RAM adapter cache | toy: separate +22% → +7% p50 at `max_loras=3`; chat: separate ≈ unified |
| 10 | phase 4 printed the adapter-first policy as **"separate"**, though it runs in the unified pool | renamed `adapter-first` | naming only |
| 7 | `ADAPTER_MB` formula treats k/v as 4096→4096; with GQA they are 4096→1024 (exact rank-32 size ≈ 168 MB) | comment in `core.py` | cosmetic |
| 8 | `validation/serve.sh` defaulted `MAX_CPU_LORAS=0`, which real vLLM rejects at startup | defaults to `MAX_LORAS` | the GPU harness would not have started |
