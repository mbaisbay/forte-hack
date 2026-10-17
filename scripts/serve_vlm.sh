#!/usr/bin/env bash
# Start the local model server. Bound to loopback only; never expose this port.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/models/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# The GPU is shared: this is the fraction of TOTAL VRAM vLLM may take. It must be free at start.
GPU_UTIL="${GPU_UTIL:-0.85}"

exec "$ROOT/.venv-vllm/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image": 48, "video": 0}' \
  --max-num-seqs 16
