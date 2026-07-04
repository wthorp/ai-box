#!/bin/bash
# Compatibility wrapper for the Python DeepSWE harness.

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  RUNNER_ENV=(
    -e AI_BOX_RUNNER=1
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
    -e QWEN_BASE_URL="${QWEN_BASE_URL:-}"
    -e INFERENCE_SERVICE="${INFERENCE_SERVICE:-}"
    -e INFERENCE_PORT="${INFERENCE_PORT:-}"
    -e EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-}"
  )
  while IFS='=' read -r name _; do
    [[ "$name" == QSA_* ]] || continue
    RUNNER_ENV+=(-e "$name=${!name}")
  done < <(env)
  exec docker compose run --rm \
    "${RUNNER_ENV[@]}" \
    runner scripts/deepswe-run.sh "$@"
fi

set -euo pipefail

AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
if [[ "${INFERENCE_SERVICE:-}" == "tabbyapi" ]]; then
  INFERENCE_PORT="${INFERENCE_PORT:-5000}"
else
  INFERENCE_PORT="${INFERENCE_PORT:-8080}"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:${INFERENCE_PORT}/v1}"
if [[ "${INFERENCE_SERVICE:-}" == "tabbyapi" ]]; then
  export QWEN_BASE_URL="${QWEN_BASE_URL:-$OPENAI_BASE_URL}"
fi

exec python3 scripts/deepswe.py run "$@"
