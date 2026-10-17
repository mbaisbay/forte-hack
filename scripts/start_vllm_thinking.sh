#!/usr/bin/env bash
# Diagnostic: Qwen3-VL-8B-Thinking (Apache-2.0) with its reasoning trace returned separately.
cd ~/fortebank-hack
export HF_HOME=$HOME/fortebank-hack/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec .venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 --reasoning-parser qwen3 \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} \
  --limit-mm-per-prompt '{"image": 48, "video": 0}' --max-num-seqs 16
