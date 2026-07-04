#!/bin/bash
# Wrapper for goal-based DeepSWE staging runs.

set -euo pipefail

FORCE_HOST=0
RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--host" ]]; then
    FORCE_HOST=1
  else
    RUNNER_ARGS+=("$1")
  fi
  shift
done

if [[ "$FORCE_HOST" -eq 0 && -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  RUNNER_ENV=(
    -e AI_BOX_RUNNER=1
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
    -e QWEN_BASE_URL="${QWEN_BASE_URL:-}"
    -e EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-}"
    -e DEEPSWE_DIR="${DEEPSWE_DIR:-}"
    -e AI_BOX_HOST_DIR="${ROOT}"
  )
  while IFS='=' read -r name _; do
    [[ "$name" == QSA_* ]] || continue
    RUNNER_ENV+=(-e "$name")
  done < <(env)
  exec docker compose run --rm \
    "${RUNNER_ENV[@]}" \
    runner scripts/deepswe-goal-run.sh "${RUNNER_ARGS[@]}"
fi

AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:8080/v1}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-$OPENAI_BASE_URL}"
export DEEPSWE_DIR="${DEEPSWE_DIR:-/deep-swe}"

exec python3 scripts/deepswe_goal_run.py "${RUNNER_ARGS[@]}"
