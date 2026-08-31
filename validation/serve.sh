#!/usr/bin/env bash
# Launch vLLM with a LoRA slab sized for one sweep cell.
#
# Sweep A (the important one): run this once per MAX_LORAS in 12 7 5 3, each
# time driving it with validation/workload.py, then tear it down before the
# next value. The simulator's phase-2 x-axis IS this knob.
#
#   MAX_LORAS=5 ./validation/serve.sh
#
# Env knobs (all optional):
#   MODEL          base model (default meta-llama/Llama-3.1-8B-Instruct)
#   MAX_LORAS      GPU adapter slab, # of adapters (default 5)
#   MAX_CPU_LORAS  CPU adapter tier (default 0 -> phase-2 Part A; set 8/16 for Part B / Sweep B)
#   MAX_LORA_RANK  default 32 (all-linear targets -> ~180MB, matches core.ADAPTER_MB)
#   GPU_MEM_UTIL   default 0.85
#   PORT           default 8000
#   LORA_MODULES   space-separated name=path pairs; if unset you must pass --lora-modules yourself
#
# Record the exact printed command + `vllm --version` into validation/results/.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_LORAS="${MAX_LORAS:-5}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
PORT="${PORT:-8000}"

cmd=(vllm serve "$MODEL"
  --enable-lora
  --max-loras "$MAX_LORAS"
  --max-cpu-loras "$MAX_CPU_LORAS"
  --max-lora-rank "$MAX_LORA_RANK"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --port "$PORT")

if [[ -n "${LORA_MODULES:-}" ]]; then
  cmd+=(--lora-modules $LORA_MODULES)
fi

echo "vllm version: $(vllm --version 2>/dev/null || echo '???')"
echo "+ ${cmd[*]}"
exec "${cmd[@]}"
