# GPU run — phase 7 on a real A100

**Question:** on a real GPU, does vLLM's CPU KV offload cut time to first token
(TTFT) by roughly what `phase7_elora.py` predicts? Is vLLM's adapter-slot count
(`--max-loras`) as important as the simulator says?

ELORA itself has no public code, so it is not run here. The simulator says
ELORA ≈ tuned vLLM + offload, so these two vLLM configurations are the claims
that can be checked.

## What runs

- **Model:** Llama-3.1-8B-Instruct.
- **Adapters:** 20 random rank-32 LoRAs, about 168 MB each.
- **Workload:** phase 7's chat workload, token for token:
  - 80 conversations × 5 turns.
  - About 200 new prompt tokens and 200 reply tokens per turn.
  - A 400-token shared system prompt.
  - Steady and 5× bursty arrivals.

**Matrix (6 server configs × 2 bursts × 3 seeds = 36 runs of about 4–5 min each):**

| | max_loras 6 (ELORA's 0.2 ratio) | 10 | 14 |
|---|---|---|---|
| vLLM (prefix caching) | ✓ | ✓ | ✓ |
| vLLM + native CPU KV offload (19 GB) | ✓ | ✓ | ✓ |

**KV capacity:** the GPU KV cache is fixed with `--num-gpu-blocks-override` to
the simulator's KV region, (6000 MB − slots × 180 MB) / 0.125 MB per token.
For 6, 10 and 14 slots that is 2,460, 2,100 and 1,740 blocks. An 80 GB card
therefore behaves like the simulator's 6 GB budget, and the results do not
depend on the card's size.

## Measured results (run 2026-10-13)

RunPod A100-SXM4-80GB, vLLM 0.31.0, 36/36 runs, 400/400 requests each, 0
errors, about 3 h 15 min and ~$5. Raw JSONL, server logs and the full log are
in [`a100_results/`](a100_results/); `compare.txt` is the output of
`compare_p7.py`. Mean TTFT in ms (TPOT in brackets), 3 seeds:

| config | steady: sim a100 | steady: measured | bursty: sim a100 | bursty: measured |
|---|---:|---:|---:|---:|
| vLLM ml=6 | 1,443 | 2,077 (21.4) | 5,266 | 4,401 (23.8) |
| vLLM ml=10 | 572 | 362 (23.5) | 1,427 | 2,506 (31.7) |
| vLLM ml=14 | 597 | 355 (24.6) | 2,122 | 5,345 (32.0) |
| vLLM + offload ml=6 | 879 | 1,796 (20.5) | 2,979 | 3,876 (22.3) |
| vLLM + offload ml=10 | 406 | 301 (22.2) | 580 | 1,474 (27.4) |
| vLLM + offload ml=14 | 405 | 246 (22.3) | 541 | 3,211 (27.8) |

| effect | sim a100 | measured |
|---|---:|---:|
| offload vs best plain vLLM, steady | −29% | −31% |
| offload vs best plain vLLM, bursty | −62% | −41% |
| best plain vLLM vs ml=6, steady | −60% | −83% |
| best plain vLLM vs ml=6, bursty | −73% | −43% |
| best offload vs offload ml=6 (ELORA's baseline), steady | −54% | −86% |
| best offload vs offload ml=6, bursty | −81% | −62% |

**Verdict:** the signs all match and the sizes are within about 20 points.
The adapter-slot setting and the CPU KV tier together explain the large
gains, and too many slots hurt under bursts (ml=14 worse than ml=10).

**Caveats:**
- Seed 0 always runs first after a server start and is 2–4× slower than
  seeds 1 and 2 in every config. It reproduced in the smoke test (3,749 vs
  3,714 ms), so it is either warm-up or a heavier seed; the two cannot be
  separated here. Without seed 0, offload's steady gain is −2% (best offload
  179 ms vs best plain 182 ms), but its bursty gain holds (990 vs 1,645 ms,
  −40%).
- `late` is large (0.6–5.5 s) because a turn is sent only after the previous
  turn of that conversation finishes. That is the intended closed-loop
  behaviour, but it means bursty cells see less load than scheduled.
- TPOT is 20–32 ms, against 15–21 ms in the a100 profile.

## Predicted results (before the run)

From `python validation/compare_p7.py --predict-only`. Mean TTFT in ms, rough
A100 timing profile, 3 seeds:

| config | steady | 5× bursts |
|---|---|---|
| vLLM ml=6 | ~1,440 | ~5,270 |
| vLLM ml=10 | ~570 | ~1,430 |
| vLLM ml=14 | ~600 | ~2,120 |
| vLLM + offload ml=6 | ~880 | ~2,980 |
| vLLM + offload ml=10 | ~410 | ~580 |
| vLLM + offload ml=14 | ~410 | ~540 |

Headline predictions:
- **Offload vs best vLLM:** about −30% TTFT steady, −60% bursty.
- **Tuning slots from 6 to 10 (plain vLLM):** about −60% steady, −73% bursty.

Seeds matter: with 5 seeds instead of 3, the ml=6 steady cell is about
2,500 ms. Compare cells run with the same seeds.

The A100 timings are a guess. Expect the **direction** to hold; the
**sizes** could be off by up to about 2×.

## Cost and time

| step | time |
|---|---|
| pod start, `pip install vllm` | 10 min |
| model download (16 GB) and adapters | 10 min |
| smoke test (`SMOKE=1`: ml=6, offload off and on, 1 seed) | 15 min |
| full matrix (36 runs + 6 server restarts) | 2 h 50 min |
| copy results off, buffer | 15 min |
| **total** | **about 3 h 40 min** |

**Cost:** one A100 80 GB at about $1.2–1.9/h is **about $5–8**; stop at $12.
The only choice is community or secure cloud; check today's prices.

To roughly halve the cost, skip the smoke test once the first config works,
or run `SEEDS="0 1"`: about 2.5 h, $3–5.

A cheaper 48 GB card (L40S) would also work, because the KV cache is capped.
But the predictions above assume A100 speeds.

## Steps

### 0. Before renting (free)

1. Accept the Llama 3.1 license on Hugging Face and create a read token.
2. Push this repo to GitHub, or plan to `scp` it to the pod.
3. Run `python validation/compare_p7.py --predict-only` locally and keep
   the output.

### 1. Pod

Use RunPod (or similar) with:
- 1× A100 80 GB, PyTorch / CUDA 12 image.
- 50 GB disk.
- At least 32 GB RAM. Offload needs 19 GB of pinned host RAM.

```bash
git clone <your repo> && cd lora-serve-sim
pip install uv && uv venv /root/venv && . /root/venv/bin/activate
uv pip install vllm==0.31.0   # the version this harness was checked against
uv pip install -r validation/requirements.txt safetensors huggingface_hub numpy
vllm --version
vllm serve --help | grep -E "kv-offloading|num-gpu-blocks-override"   # both must exist
export HF_TOKEN=<token>
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
python validation/make_adapters.py --out /workspace/adapters --n 20
# paste the two export lines it prints:
export LORA_MODULES="lora0=/workspace/adapters/lora0 ..."
export ADAPTER_NAMES="lora0 ... lora19"
```

**If `--kv-offloading-*` is missing** (older vLLM), pass the connector
directly:

```bash
EXTRA_ARGS='--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":9500}}'
```

Leave `OFFLOAD_GB=0` and do the off/on runs by hand. Check the key names
against that version's docs.

**Do not use LMCache.** It does not include the adapter in its cache key
(LMCache issue #2961), so it can hand one adapter's KV to another and make
results look better than they are.

### 2. Smoke test (stop here if anything is off)

```bash
SMOKE=1 ./validation/run_p7.sh 2>&1 | tee validation/results/smoke.log
```

Check all of the following:
- Every config starts. If vLLM refuses `--enable-lora` together with
  offloading, offload + LoRA cannot be tested on this version. Run only
  `OFFLOAD=0` and note it.
- The server log reports the expected number of GPU KV blocks.
- Each run prints `0 errors`.
- `late` stays small relative to TTFT. Turns are sent when the previous turn
  finishes, so some lateness is normal under load.
- Offload runs are not slower than plain runs at the same slot count. If they
  are, offload may not be engaging; check the server log.

### 3. Full run

```bash
nohup ./validation/run_p7.sh > validation/results/full.log 2>&1 &
tail -f validation/results/full.log
```

### 4. Results

```bash
python validation/compare_p7.py | tee validation/results/compare.txt
tar czf p7_results.tgz validation/results
# copy p7_results.tgz to your laptop, THEN terminate the pod (stopped pods still bill for disk)
```

## Reading the outcome

| what you see | meaning |
|---|---|
| Offload beats best plain vLLM by tens of %, and ml=6 is much worse than ml=10 | Simulator confirmed: ELORA's gain is mostly the CPU KV tier plus its baseline's slot setting |
| Same direction, much smaller effects | Real recompute is cheaper than modelled. Recalibrate `prefill_ms_per_token` from the measured TTFT |
| Offload no better than plain | Either offload is not engaging (check logs), or prefix caching already keeps enough. A real negative result: report it |
| Many errors or timeouts | Discard the cell. Rerun with fewer seeds, or check `MAX_MODEL_LEN` |
