#!/usr/bin/env bash
# Phase 7 GPU run: vLLM vs vLLM + CPU KV offload, at several adapter-slot
# counts, on phase 7's chat workload. See validation/GPU_RUN.md.
#
# For each (MAX_LORAS, offload on/off): start the server with the KV cache
# fixed to the simulator's KV region, run every (burst, seed), stop.
#
#   ./validation/run_p7.sh                 # full matrix
#   SMOKE=1 ./validation/run_p7.sh         # ~15 min check: ml=6, offload off and on, 1 seed
#
# Env knobs:
#   MLS      adapter slot counts            (default "6 10 14"; 6 = ELORA's 0.2 ratio)
#   OFFLOAD  offload settings, GB of host   (default "0 19"; 19 GB ~ 3.2 x the 6 GB budget)
#   BURSTS   default "1 5"
#   SEEDS    default "0 1 2"
#   GPU_MB   simulated GPU budget for adapters + KV (default 6000, phase 7's)
#   ADAPTER_NAMES / LORA_MODULES as printed by make_adapters.py
#
# KV blocks = (GPU_MB - MLS * 180 MB) / 0.125 MB per token / 16 tokens per block,
# i.e. exactly the KV region phase7_elora.py gives vLLM at that slot count.

set -euo pipefail
cd "$(dirname "$0")/.."

MLS="${MLS:-6 10 14}"
OFFLOAD="${OFFLOAD:-0 19}"
BURSTS="${BURSTS:-1 5}"
SEEDS="${SEEDS:-0 1 2}"
GPU_MB="${GPU_MB:-6000}"
PORT="${PORT:-8000}"
RESULTS="validation/results"
: "${LORA_MODULES:?export LORA_MODULES (see make_adapters.py output)}"
: "${ADAPTER_NAMES:?export ADAPTER_NAMES (see make_adapters.py output)}"
N_ADAPTERS=$(wc -w <<<"$ADAPTER_NAMES")
mkdir -p "$RESULTS"

if [[ "${SMOKE:-0}" == "1" ]]; then MLS="6"; BURSTS="1"; SEEDS="0"; fi

vllm --version | tee "$RESULTS/vllm_version.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$RESULTS/gpu.txt"

for OFF in $OFFLOAD; do
for ML in $MLS; do
  BLOCKS=$(python3 -c "print(int(($GPU_MB - $ML * 180) / 0.125 / 16))")
  TAG="ml${ML}_off${OFF}"
  echo "=================  $TAG  (KV blocks $BLOCKS)  ================="
  MAX_LORAS=$ML MAX_CPU_LORAS=$N_ADAPTERS NUM_GPU_BLOCKS=$BLOCKS OFFLOAD_GB=$OFF PORT=$PORT \
    ./validation/serve.sh > "$RESULTS/serve_$TAG.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

  echo "waiting for server..."
  UP=0
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then UP=1; break; fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then
      echo "server died; tail of $RESULTS/serve_$TAG.log:"; tail -30 "$RESULTS/serve_$TAG.log"; exit 1
    fi
    sleep 5
  done
  if [[ $UP != 1 ]]; then
    echo "server not healthy after 10 min; tail of $RESULTS/serve_$TAG.log:"; tail -30 "$RESULTS/serve_$TAG.log"; exit 1
  fi
  grep -i -E "num_gpu_blocks|GPU KV cache size|offload" "$RESULTS/serve_$TAG.log" | head -5 || true

  for B in $BURSTS; do
  for S in $SEEDS; do
    python3 validation/workload_p7.py --base-url "http://localhost:${PORT}/v1" \
      --adapters $ADAPTER_NAMES --n-adapters 20 --seed "$S" --burst "$B" --tag "$TAG" \
      --out "$RESULTS/p7_${TAG}_b${B}_s${S}.jsonl"
  done
  done

  kill $SERVE_PID 2>/dev/null || true
  wait $SERVE_PID 2>/dev/null || true
  trap - EXIT
  sleep 5
done
done

echo
echo "done. now:  python3 validation/compare_p7.py"
