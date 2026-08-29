#!/usr/bin/env bash
# Sweep A driver: for each MAX_LORAS value, (re)start the server, wait for it,
# snapshot /metrics, run the workload for every seed, snapshot again, stop.
#
# Assumes:
#   - a GPU box with `vllm` installed in the ACTIVE env
#   - client deps installed:  pip install -r validation/requirements.txt
#   - LORA_MODULES exported as "name=path" pairs for >=12 adapters, e.g.
#       export LORA_MODULES="lora0=/adapters/a0 lora1=/adapters/a1 ... lora11=/adapters/a11"
#   - ADAPTER_NAMES matching, space-separated, in adapter_id order
#
#   ./validation/run_sweep.sh
#
# Results land in validation/results/. Then:
#   python validation/compare.py --max-loras 12 7 5 3 --seeds 0 1 2 3 4

set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP="${SWEEP:-12 7 5 3}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}/v1"
ADAPTER_NAMES="${ADAPTER_NAMES:-lora0 lora1 lora2 lora3 lora4 lora5 lora6 lora7 lora8 lora9 lora10 lora11}"
RESULTS="validation/results"
mkdir -p "$RESULTS"

vllm --version | tee "$RESULTS/vllm_version.txt"

for ML in $SWEEP; do
  echo "=================  MAX_LORAS=$ML  ================="
  MAX_LORAS=$ML PORT=$PORT ./validation/serve.sh > "$RESULTS/serve_maxloras${ML}.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

  echo "waiting for server..."
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then break; fi
    sleep 5
    if ! kill -0 $SERVE_PID 2>/dev/null; then
      echo "server died; see $RESULTS/serve_maxloras${ML}.log"; exit 1
    fi
  done

  python validation/metrics.py snapshot --url "http://localhost:${PORT}/metrics" \
    --out "$RESULTS/metrics_maxloras${ML}_before.json"

  for S in $SEEDS; do
    echo "--- seed $S ---"
    python validation/workload.py \
      --base-url "$BASE_URL" \
      --adapters $ADAPTER_NAMES \
      --max-loras "$ML" --seed "$S" \
      --out "$RESULTS/run_maxloras${ML}_seed${S}.jsonl"
  done

  python validation/metrics.py snapshot --url "http://localhost:${PORT}/metrics" \
    --out "$RESULTS/metrics_maxloras${ML}_after.json"
  python validation/metrics.py diff \
    "$RESULTS/metrics_maxloras${ML}_before.json" \
    "$RESULTS/metrics_maxloras${ML}_after.json" \
    | tee "$RESULTS/metrics_maxloras${ML}_diff.txt"

  kill $SERVE_PID 2>/dev/null || true
  wait $SERVE_PID 2>/dev/null || true
  trap - EXIT
  sleep 5
done

echo
echo "sweep done. now:"
echo "  python validation/compare.py --max-loras $SWEEP --seeds $SEEDS"
