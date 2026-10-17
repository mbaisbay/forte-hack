#!/usr/bin/env bash
# One-time setup on the GPU machine. Needs internet. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export HF_HOME="${HF_HOME:-$ROOT/models/hf}"

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# App environment (light: no torch)
uv venv -q --python 3.12 .venv
uv sync -q

# Model server environment (separate, it pins its own torch/CUDA)
uv venv -q --python 3.12 .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install -q "vllm>=0.11" "huggingface_hub[cli]"
.venv-vllm/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__, 'gpu ok', torch.cuda.is_available())"

# Weights (~17 GB for 8B bf16)
.venv-vllm/bin/hf download "$MODEL"
echo "setup done: model=$MODEL HF_HOME=$HF_HOME"
