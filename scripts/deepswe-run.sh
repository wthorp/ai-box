#!/bin/bash
# Run DeepSWE through Pier against the currently loaded local OpenAI-compatible
# endpoint. Use mini-swe-agent for quant/model comparisons; use codex later for
# full agent + MCP/skills experiments.
#
# Usage:
#   ./deepswe-run.sh --n-tasks 5 --sample-seed 0
#   ./deepswe-run.sh --agent codex --model openai/local --n-tasks 1

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/deepswe-run.sh "$@"
fi

set -euo pipefail

AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

AGENT=mini-swe-agent
MODEL=openai/local
TASK_PATH="${DEEPSWE_DIR:-/deep-swe}/tasks"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --path) TASK_PATH="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -d "$TASK_PATH" ]]; then
  echo "ERROR: DeepSWE tasks not found at $TASK_PATH" >&2
  echo "Clone https://github.com/datacurve-ai/deep-swe to DEEPSWE_DIR on the host." >&2
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1}"

mkdir -p "${EVAL_RESULTS_DIR:-/eval-results}"

pier run \
  -p "$TASK_PATH" \
  --agent "$AGENT" \
  --model "$MODEL" \
  "${EXTRA_ARGS[@]}"
