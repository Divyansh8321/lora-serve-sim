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
#   MAX_CPU_LORAS  CPU adapter tier (default = MAX_LORAS, vLLM's own default; vLLM
#                  REJECTS values below MAX_LORAS -- vllm/config/lora.py -- so there
#                  is no "no CPU tier" config on real vLLM. Set 8/16 for Sweep B)
#   MAX_LORA_RANK  default 32 (all-linear targets -> ~180MB, matches core.ADAPTER_MB)
#   GPU_MEM_UTIL   default 0.85
#   PORT           default 8000
#   LORA_MODULES   space-separated name=path pairs; if unset you must pass --lora-modules yourself
#   NUM_GPU_BLOCKS fix the GPU KV cache to this many 16-token blocks (phase 7 run: makes the
#                  KV region match the simulator's, independent of how big the GPU is)
#   OFFLOAD_GB     >0 turns on vLLM's native CPU KV offloading with this many GB of host RAM
#                  (--kv-offloading-backend native). NOT LMCache: LMCache does not key its
#                  cache by LoRA adapter (LMCache issue #2961), so it can serve one adapter's
#                  KV to another.
#   MAX_MODEL_LEN  default 8192
#   EXTRA_ARGS     appended verbatim (e.g. a --kv-transfer-config for older vLLM versions)
#
# Record the exact printed command + `vllm --version` into validation/results/.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_LORAS="${MAX_LORAS:-5}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-$MAX_LORAS}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
PORT="${PORT:-8000}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-}"
OFFLOAD_GB="${OFFLOAD_GB:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

cmd=(vllm serve "$MODEL"
  --enable-lora
  --max-loras "$MAX_LORAS"
  --max-cpu-loras "$MAX_CPU_LORAS"
  --max-lora-rank "$MAX_LORA_RANK"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-prefix-caching
  --port "$PORT")

if [[ -n "$NUM_GPU_BLOCKS" ]]; then
  cmd+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS")
fi
if [[ "$OFFLOAD_GB" != "0" ]]; then
  cmd+=(--kv-offloading-backend native --kv-offloading-size "$OFFLOAD_GB")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  cmd+=($EXTRA_ARGS)
fi

if [[ -n "${LORA_MODULES:-}" ]]; then
  cmd+=(--lora-modules $LORA_MODULES)
fi

echo "vllm version: $(vllm --version 2>/dev/null || echo '???')"
echo "+ ${cmd[*]}"
exec "${cmd[@]}"
