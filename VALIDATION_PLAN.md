# Validating the simulator against real vLLM

Goal: confirm the phase-2 pathology (stale KV from a squeezed adapter slab) is
real on hardware, and check whether the phase-3 negative result holds — that a
cross-pool signal moves staleness but not latency.

Budget: ~1 weekend, ~$20-40 of GPU time.

## 1. Machine

Lambda Labs (or RunPod / Vast.ai). An **A10 24GB (~$0.75/hr)** is enough for an
8B model with several adapters; an **A100 40GB (~$1.30/hr)** gives more headroom
if you want longer conversations. Tear the instance down when you stop — idle
instances still bill.

## 2. Environment — a throwaway env, not your main one

```bash
conda create -n vllm-val python=3.11 -y
conda activate vllm-val
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```

## 3. Adapters

You need several distinct LoRA adapters for one base model. Either pull existing
ones from the Hub, or train N tiny throwaway adapters (rank 32, all-linear
targets, a few hundred steps each on any small dataset) — quality is irrelevant,
only their memory footprint and identity matter.

Verify each is ~180 MB, matching `ADAPTER_MB` in `core.py` and ELORA's stated
rank of 32/64.

## 4. Serve

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-cpu-loras 8 \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

The two knobs that matter:
- `--max-loras` — GPU adapter slab size. **This is the phase-2 x-axis.**
- `--gpu-memory-utilization` — total pool, which indirectly sizes the KV region.

## 5. Workload driver

Write a client that mirrors `make_multi_turn_workload`: N conversations, each
issuing several turns with think-time gaps between them, adapters drawn from a
Zipf distribution. Send each turn with the full prior history so the KV cache
grows realistically. Record per-request TTFT and end-to-end latency.

Match the simulator's parameters exactly — same conversation count, turns,
token counts, arrival rate — so the comparison is apples to apples.

## 6. The sweeps

**Sweep A (phase 2, the important one).** Hold everything fixed, vary
`--max-loras` over 12 / 7 / 5 / 3 with more adapters in play than slots.
Prediction from the simulator: p50 rises and adapter reloads climb sharply as
the slab shrinks. Plot measured p50/p95 against the simulator's curve.

**Sweep B (phase 2b).** Vary `--max-cpu-loras` (0 vs 8 vs 16) at a fixed tight
`--max-loras`. Prediction: latency penalty largely disappears while the
underlying eviction rate does not change — the CPU tier makes mistakes cheaper,
not rarer.

**Sweep C (phase 4).** vLLM will not unify the pools, so this is not directly
testable. What *is* testable is the premise: confirm from metrics that adapter
and KV memory are independently sized and cannot borrow from each other.

## 7. What to measure

- **TTFT and end-to-end latency** per request (from your client)
- **Adapter reload counts** — scrape `vllm:lora_requests_info` from `/metrics`.
  Note the known gap (vLLM issue #45325): adapters vanish from this metric once
  idle, so you may need to infer reloads from TTFT spikes on adapter switches.
- **Preemption counts** — vLLM logs preemptions; these are your KV evictions.

## 8. Success criteria — decide these before you run

- **Strong:** measured p50 tracks the simulator's predicted curve within ~20%
  across the `--max-loras` sweep, and the CPU-tier effect reproduces.
- **Useful:** the *shape* matches (monotonic degradation, sharp knee at the same
  slab size) even if absolute numbers differ. Recalibrate constants and say so.
- **Negative but publishable:** the pathology does not appear — which would mean
  vLLM's reclaim pool or prefix caching is absorbing it. That is a real finding
  and should be reported, not buried.

## 9. Writing it up

Whatever happens, record the exact vLLM version, flags, model, adapter ranks,
and workload parameters. A validation nobody can reproduce is worth little.

If the numbers diverge, the honest move is to report the divergence and explain
which simplification caused it — the reclaim pool and prefix caching are the two
most likely culprits, and both are already named in the README's limitations.

## 10. Hardware-profile checkpoint (done in-sim, pre-GPU)

Phases 5–7 take `--hw {ours,elora-aggressive,elora-conservative}` so the ELORA
comparison is apples-to-apples on the constants, not just the policy. Recorded
findings:

- **Phase 5**: the stale-KV pathology is hardware-robust — identical *shape*
  at `--hw ours` and `--hw elora` (fit-ratio effect, not bandwidth).
- **Phase 6**: dependency-scoring's benefit *shrinks* as HW gets faster
  (+6.7% → +2.2% at 800 shared tokens).
- **Phase 7**: the 100 ms swapper is net-negative at our engine model
  (−5% to −22%), **net-positive at ELORA's** (+0.4% to +2.9%).
  `phase7 --sweep` isolates `--swap-out-when full` (a policy choice) as the
  single change that flips the sign; `--sweep-const` shows no hardware
  constant alone does. This reproduces the *direction* of ELORA-WOS's 1.42×
  at a plausible H800 constant band — not a reproduction of the magnitude.

When the GPU run happens, the swapper comparison should be checked against
`--hw` matched to the actual rented card (A10 → `ours`; A100/H100 → an
`elora`-like profile), not the default.
