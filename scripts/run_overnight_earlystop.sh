#!/bin/bash
set -euo pipefail

if [[ -d /data/ai/ai-box ]]; then
  cd /data/ai/ai-box
else
  cd /workspace
fi

export INFERENCE_SERVICE=tabbyapi
export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:5000/v1}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-$OPENAI_BASE_URL}"

export QSA_EARLY_STOP="${QSA_EARLY_STOP:-1}"
export QSA_FAIL_SCORE_ABORT="${QSA_FAIL_SCORE_ABORT:-0.70}"
export QSA_LOOP_ABORT_REPEATS="${QSA_LOOP_ABORT_REPEATS:-5}"
export QSA_NO_EDIT_ABORT_STEP="${QSA_NO_EDIT_ABORT_STEP:-45}"
export QSA_VALIDATION_GRACE_STEPS="${QSA_VALIDATION_GRACE_STEPS:-12}"
export QSA_STALE_ABORT_STEPS="${QSA_STALE_ABORT_STEPS:-10}"

export QSA_MAX_STEPS="${QSA_MAX_STEPS:-80}"
export QSA_MAX_TOKENS="${QSA_MAX_TOKENS:-512}"
export QSA_LLM_TIMEOUT_SEC="${QSA_LLM_TIMEOUT_SEC:-240}"
export QSA_TEMPERATURE="${QSA_TEMPERATURE:-0.05}"
export QSA_TOP_P="${QSA_TOP_P:-0.8}"
export QSA_REPEAT_PENALTY="${QSA_REPEAT_PENALTY:-1.12}"
export QSA_REPEAT_LAST_N="${QSA_REPEAT_LAST_N:-4096}"
export QSA_DRY_MULTIPLIER="${QSA_DRY_MULTIPLIER:-1.0}"
export QSA_DRY_BASE="${QSA_DRY_BASE:-1.75}"
export QSA_DRY_ALLOWED_LENGTH="${QSA_DRY_ALLOWED_LENGTH:-3}"
export QSA_DRY_PENALTY_LAST_N="${QSA_DRY_PENALTY_LAST_N:-4096}"
export QSA_MIROSTAT="${QSA_MIROSTAT:-0}"

RESULTS_DIR="${RESULTS_DIR:-/data/ai/local/eval-results/overnight-earlystop-20260602}"
N_TASKS="${N_TASKS:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-20260602}"

mkdir -p "$RESULTS_DIR"

exec ./scripts/deepswe-run.sh \
  --agent qwen-sverklo \
  --mcp-profile sverklo \
  --model openai/local \
  --task-path /data/ai/deep-swe/tasks \
  --results-dir "$RESULTS_DIR" \
  --job-name qwen-sverklo-gemma-tabbyapi-earlystop \
  --n-tasks "$N_TASKS" \
  --n-concurrent 1 \
  --sample-seed "$SAMPLE_SEED" \
  --quiet-yes \
  -- \
  --agent-kwarg max_steps="$QSA_MAX_STEPS" \
  --agent-kwarg max_tokens="$QSA_MAX_TOKENS" \
  --agent-kwarg temperature="$QSA_TEMPERATURE" \
  --agent-kwarg top_p="$QSA_TOP_P" \
  --agent-kwarg repeat_penalty="$QSA_REPEAT_PENALTY" \
  --agent-kwarg repeat_last_n="$QSA_REPEAT_LAST_N" \
  --agent-kwarg dry_multiplier="$QSA_DRY_MULTIPLIER" \
  --agent-kwarg dry_base="$QSA_DRY_BASE" \
  --agent-kwarg dry_allowed_length="$QSA_DRY_ALLOWED_LENGTH" \
  --agent-kwarg dry_penalty_last_n="$QSA_DRY_PENALTY_LAST_N" \
  --agent-kwarg mirostat="$QSA_MIROSTAT"
