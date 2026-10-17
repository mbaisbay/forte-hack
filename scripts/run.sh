#!/usr/bin/env bash
# Start API + worker. Expects the model server (serve_vlm.sh) to be running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# .env fills in only what the environment does not already set (so VLM_MODEL=... bash scripts/run.sh wins).
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -z "${!key+x}" ] && export "$key=$value"
  done < .env
fi
: "${API_KEY:?set API_KEY in .env}"
export HF_HUB_OFFLINE=1

.venv/bin/python -m app.worker &
WORKER=$!
trap 'kill $WORKER 2>/dev/null' EXIT
# One API process: the worker is separate, so the API never loads a model.
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8080}" --workers 1
