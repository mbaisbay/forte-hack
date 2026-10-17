#!/usr/bin/env bash
# Experimental: Qwen3-VL-32B (AWQ 4-bit, Apache-2.0 base) for the bigger-model comparison.
cd ~/fortebank-hack
export HF_HOME=$HOME/fortebank-hack/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec .venv-vllm/bin/vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
  --host 127.0.0.1 --port 8000 --dtype auto --max-model-len 16384 \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image": 48, "video": 0}' --max-num-seqs 4
