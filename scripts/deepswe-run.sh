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
  exec docker compose run --rm \
    -e AI_BOX_RUNNER=1 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}" \
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
    runner scripts/deepswe-run.sh "$@"
fi

set -euo pipefail

AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

AGENT=mini-swe-agent
AGENT_IMPORT_PATH=""
MODEL=openai/local
TASK_PATH="${DEEPSWE_DIR:-/deep-swe}/tasks"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT="$2"; shift 2 ;;
    --agent-import-path) AGENT_IMPORT_PATH="$2"; shift 2 ;;
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
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:8080/v1}"

mkdir -p "${EVAL_RESULTS_DIR:-/eval-results}"

pier_args=(
  -p "$TASK_PATH" \
  --model "$MODEL" \
)
if [[ -n "$AGENT_IMPORT_PATH" ]]; then
  pier_args+=(--agent-import-path "$AGENT_IMPORT_PATH")
else
  pier_args+=(--agent "$AGENT")
fi
pier_args+=("${EXTRA_ARGS[@]}")

pier run "${pier_args[@]}"
