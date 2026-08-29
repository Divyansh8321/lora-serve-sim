# Validation harness — simulator vs. real vLLM

Turns `VALIDATION_PLAN.md` into runnable code. The goal is Sweep A: confirm the
phase-2 stale-KV pathology (p50 rises, adapter reloads climb as `--max-loras`
shrinks) reproduces on real hardware, and check whether the shape matches the
simulator's predicted curve.

Nothing here needs a GPU except the actual sweep run. Everything else —
building the workload, the sim's predicted curve, parsing results — runs
locally.

## Files

| file | what it does | needs GPU |
|---|---|---|
| `serve.sh` | one `vllm serve` invocation, slab size via `MAX_LORAS` env | yes |
| `workload.py` | drives `/v1/chat/completions` with the **exact** `make_multi_turn_workload` structure from `core.py`; records TTFT + e2e per turn to JSONL | needs a server |
| `metrics.py` | snapshot/diff vLLM `/metrics` (preemptions = KV evictions) | needs a server |
| `run_sweep.sh` | orchestrates: for each `MAX_LORAS`, start server → snapshot → run all seeds → snapshot → stop | yes |
| `compare.py` | tabulates measured vs. the simulator's own phase-2 `trial()` output | no |

## Preview the predicted curve now (no GPU)

```bash
python validation/compare.py --predict-only --max-loras 12 7 5 3
```

This is the target. Measured p50 should track it (within ~20% = "strong",
same shape = "useful").

## On the GPU box

### 1. Env

```bash
conda create -n vllm-val python=3.11 -y && conda activate vllm-val
pip install vllm
pip install -r validation/requirements.txt   # client-side: openai, httpx
vllm --version
```

### 2. Adapters

Need ≥12 distinct rank-16 LoRA adapters for one base model (the workload uses
12 by default; more slots than the tightest `--max-loras` is the whole point).
Pull from the Hub or train tiny throwaways — quality is irrelevant, only that
each is ~20 MB (matches `ADAPTER_MB` in `core.py`).

```bash
export LORA_MODULES="lora0=/adapters/a0 lora1=/adapters/a1 ... lora11=/adapters/a11"
export ADAPTER_NAMES="lora0 lora1 lora2 lora3 lora4 lora5 lora6 lora7 lora8 lora9 lora10 lora11"
```

`ADAPTER_NAMES` order maps 1:1 onto the simulator's integer `adapter_id`, so
keep it stable across the whole sweep.

### 3. Run Sweep A

```bash
./validation/run_sweep.sh          # SWEEP="12 7 5 3", SEEDS="0 1 2 3 4"
```

Or manually, one cell at a time:

```bash
MAX_LORAS=5 ./validation/serve.sh &        # wait for /health
python validation/metrics.py snapshot --out validation/results/m_5_before.json
for s in 0 1 2 3 4; do
  python validation/workload.py --adapters $ADAPTER_NAMES --max-loras 5 --seed $s
done
python validation/metrics.py snapshot --out validation/results/m_5_after.json
python validation/metrics.py diff validation/results/m_5_{before,after}.json
```

### 4. Compare

```bash
python validation/compare.py --max-loras 12 7 5 3 --seeds 0 1 2 3 4
```

Reads `validation/results/run_maxloras{ML}_seed{S}.jsonl`, aggregates measured
p50/p95/TTFT over seeds, and prints the `p50_err%` against the simulator.

## Sweep B (phase 2b — CPU tier)

Fix a tight slab, vary the CPU adapter tier:

```bash
for c in 0 8 16; do
  MAX_LORAS=3 MAX_CPU_LORAS=$c ./validation/serve.sh &   # wait, run seeds, stop
done
```

Prediction: latency penalty largely disappears as `MAX_CPU_LORAS` grows, but
the preemption/eviction rate does **not** drop — cheaper mistakes, not fewer.
The sim's phase-2 Part B shows p50 flat (~1052→1072) while stale KV still
climbs to 46%.

## Sweep C (phase 4 premise)

vLLM won't unify the pools, so phase 4 isn't directly testable. What *is*:
confirm from `/metrics` and startup logs that adapter memory and KV memory are
sized independently and can't borrow from each other. Note the configured KV
cache blocks vs. the LoRA slab; neither grows into the other.

## Caveats baked into the harness

- **`time-scale`**: `workload.py` replays the sim's arrival schedule in real
  time (turn gaps ~6 s, total span ~108 s per seed). If the box can't keep up,
  raise `--time-scale` and record it — the comparison is then only
  shape-valid, not magnitude-valid.
- **Token accounting**: we set `max_tokens` to the sim's `output_tokens` and
  grow the prompt with turn index, but the real tokenizer decides actual
  prompt length. Prefix caching / RadixAttention on the shared system prompt
  may absorb some KV pressure — that's exactly the "negative but publishable"
  outcome in `VALIDATION_PLAN.md` §8. Report it, don't bury it.
- **Adapter reload counts** aren't reliably in `/metrics` (vLLM issue #45325).
  Infer them from TTFT spikes on adapter switches in the JSONL.
- Record `vllm --version`, all flags, model id, adapter ranks, and any
  `--time-scale` into `validation/results/` — a validation nobody can
  reproduce is worth little.
