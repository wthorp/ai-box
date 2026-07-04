#!/bin/bash
# deepswe-sweep.sh — fail-fast DeepSWE sweep across Q4..Q8.
#
# Behavior:
#   - runs quants in order: Q4, Q5, Q6, Q8
#   - for each quant, uses the largest known context
#   - sweeps even CPU_MOE_LAYERS from highest to lowest
#   - appends one TSV row per attempt
#   - stops immediately on the first DeepSWE failure or incomplete job
#
# Defaults are conservative so the script stays deterministic and readable.
#
# Usage:
#   ./deepswe-sweep.sh
#   ./deepswe-sweep.sh --n-tasks 5 --sample-seed 0
#   ./deepswe-sweep.sh --min-gpu-moe 4
#   ./deepswe-sweep.sh --output /tmp/deepswe-results.tsv
#   ./deepswe-sweep.sh --mcp-profile sverklo --n-tasks 5
#   ./deepswe-sweep.sh --service tabbyapi --quants GEMMA4_DENSE_EXL3 --context 32768
#   ./deepswe-sweep.sh --service tabbyapi --context 65536 --step-limit 40
#   ./deepswe-sweep.sh --service tabbyapi --keep-going --n-tasks 5

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  RUNNER_ARGS=()
  while [[ $# -gt 0 ]]; do
    RUNNER_ARGS+=("$1")
    shift
  done
  exec docker compose run --rm \
    -e AI_BOX_RUNNER=1 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}" \
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
    -e QWEN_BASE_URL="${QWEN_BASE_URL:-}" \
    -e QSA_AUTOFIX_AFTER_DIFF="${QSA_AUTOFIX_AFTER_DIFF:-}" \
    -e QSA_AUTOFIX_MAX_RUNS="${QSA_AUTOFIX_MAX_RUNS:-}" \
    -e QSA_AUTOFIX_TIMEOUT_SEC="${QSA_AUTOFIX_TIMEOUT_SEC:-}" \
    -e QSA_AUTOFIX_OUTPUT_LIMIT="${QSA_AUTOFIX_OUTPUT_LIMIT:-}" \
    -e QSA_SQLGLOT_DIALECTS="${QSA_SQLGLOT_DIALECTS:-}" \
    -e AI_BOX_HOST_DIR="$ROOT" \
    runner scripts/deepswe-sweep.sh "${RUNNER_ARGS[@]}"
fi

set -euo pipefail

AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

source "${AI_BOX}/scripts/lib/env.sh"
source "${AI_BOX}/scripts/lib/models.sh"

if [[ -f .env ]] && grep -q '^MODELS_DIR=' .env; then
  export COMPOSE_MODELS_DIR
  COMPOSE_MODELS_DIR=$(env_value MODELS_DIR .env)
  export MODELS_DIR="$COMPOSE_MODELS_DIR"
fi

compose() {
  MODELS_DIR="${COMPOSE_MODELS_DIR:-${MODELS_DIR:-/data/ai/models}}" \
    AI_BOX_HOST_DIR="${AI_BOX_HOST_DIR:-$AI_BOX}" \
    docker compose "$@"
}

SERVICE=rotorquant
PORT=8080
USER_SET_PORT=0
TASK_PATH="${DEEPSWE_DIR:-/deep-swe}/tasks"
AGENT=mini-swe-agent
AGENT_IMPORT_PATH="scripts.pier_agents.mini_swe_agent_run:MiniSweAgentRun"
ENVIRONMENT_IMPORT_PATH="scripts.pier_envs.docker_llm_proxy:DockerLlmProxyEnvironment"
MODEL=openai/local
N_TASKS=1
SAMPLE_SEED=0
N_CONCURRENT=1
OUTPUT_FILE=""
MOE_LIST=""
QUANT_LIST="Q4,Q5,Q6,Q8"
USER_SET_QUANTS=0
CONTEXT_OVERRIDE=""
LOAD_TIMEOUT=1800
MIN_GPU_MOE=4
EXTRA_PIER_ARGS=()
DEBUG_HARNESS=0
USER_SET_AGENT=0
USER_SET_AGENT_IMPORT=0
HARDEN_REWARDS=1
MCP_PROFILE=none
MCP_URL=""
KEEP_GOING=0
FAILURES=0
STEP_LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service) SERVICE="$2"; shift 2 ;;
    --port) PORT="$2"; USER_SET_PORT=1; shift 2 ;;
    --path) TASK_PATH="$2"; shift 2 ;;
    --agent) AGENT="$2"; USER_SET_AGENT=1; shift 2 ;;
    --agent-import-path) AGENT_IMPORT_PATH="$2"; USER_SET_AGENT_IMPORT=1; shift 2 ;;
    --environment-import-path) ENVIRONMENT_IMPORT_PATH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --n-tasks) N_TASKS="$2"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --moe) MOE_LIST="$2"; shift 2 ;;
    --quants) QUANT_LIST="$2"; USER_SET_QUANTS=1; shift 2 ;;
    --context) CONTEXT_OVERRIDE="$2"; shift 2 ;;
    --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
    --min-gpu-moe) MIN_GPU_MOE="$2"; shift 2 ;;
    --debug-harness) DEBUG_HARNESS=1; shift ;;
    --mcp-profile) MCP_PROFILE="$2"; shift 2 ;;
    --mcp-url) MCP_URL="$2"; shift 2 ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --step-limit) STEP_LIMIT="$2"; shift 2 ;;
    --no-harden-rewards) HARDEN_REWARDS=0; shift ;;
    --) shift; EXTRA_PIER_ARGS+=("$@"); break ;;
    *) EXTRA_PIER_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$USER_SET_AGENT" -eq 1 && "$USER_SET_AGENT_IMPORT" -eq 0 ]]; then
  AGENT_IMPORT_PATH=""
fi

has_agent_kwarg_prefix() {
  local prefix=$1
  local index arg next
  for index in "${!EXTRA_PIER_ARGS[@]}"; do
    arg="${EXTRA_PIER_ARGS[$index]}"
    next="${EXTRA_PIER_ARGS[$((index + 1))]:-}"
    if [[ "$arg" == --agent-kwarg="${prefix}"* || "$arg" == "${prefix}"* ]]; then
      return 0
    fi
    if [[ "$arg" == "--agent-kwarg" && "$next" == "${prefix}"* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$SERVICE" == "tabbyapi" ]]; then
  if [[ "$USER_SET_PORT" -eq 0 ]]; then
    PORT=5000
  fi
  if [[ "$USER_SET_QUANTS" -eq 0 ]]; then
    QUANT_LIST="GEMMA4_DENSE_EXL3,GEMMA4_MOE_410_EXL3,GEMMA4_MOE_510_EXL3"
  fi
  if [[ -z "$MOE_LIST" ]]; then
    MOE_LIST="0"
  fi
  if ! has_agent_kwarg_prefix "model_class="; then
    EXTRA_PIER_ARGS+=(--agent-kwarg model_class=litellm)
  fi
fi

if [[ -n "$STEP_LIMIT" ]] && ! has_agent_kwarg_prefix "step_limit="; then
  EXTRA_PIER_ARGS+=(--agent-kwarg "step_limit=${STEP_LIMIT}")
fi

if [[ ! -d "$TASK_PATH" ]]; then
  echo "ERROR: DeepSWE tasks not found at $TASK_PATH" >&2
  echo "Clone https://github.com/datacurve-ai/deep-swe to DEEPSWE_DIR on the host." >&2
  exit 1
fi

if [[ -z "$MOE_LIST" ]]; then
  if [[ "$MIN_GPU_MOE" =~ ^[0-9]+$ ]] && (( MIN_GPU_MOE >= 0 && MIN_GPU_MOE <= 48 )); then
    if (( MIN_GPU_MOE % 2 == 1 )); then
      MIN_GPU_MOE=$((MIN_GPU_MOE - 1))
    fi
    MOE_LIST=""
    for ((moe=48-MIN_GPU_MOE; moe>=0; moe-=2)); do
      MOE_LIST+="${moe},"
    done
    MOE_LIST="${MOE_LIST%,}"
  else
    echo "ERROR: --min-gpu-moe must be between 0 and 48" >&2
    exit 1
  fi
fi

IFS=',' read -ra QUANTS <<< "$QUANT_LIST"
IFS=',' read -ra MOES <<< "$MOE_LIST"

RESULTS_DIR="${EVAL_RESULTS_DIR:-/eval-results}/deepswe-sweep-$(date +%Y%m%d-%H%M%S)"
if [[ -z "$OUTPUT_FILE" ]]; then
  OUTPUT_FILE="${RESULTS_DIR}/results.tsv"
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"
RESULTS_DIR="$(cd "$(dirname "$OUTPUT_FILE")" && pwd)"
JOBS_ROOT="${RESULTS_DIR}/jobs"
mkdir -p "$RESULTS_DIR" "$JOBS_ROOT"

if [[ "$HARDEN_REWARDS" -eq 1 || "$MCP_PROFILE" != "none" ]]; then
  SOURCE_TASK_PATH="$TASK_PATH"
  overlay_args=(
    --task-path "$TASK_PATH"
    --results-dir "$RESULTS_DIR"
    --mcp-profile "$MCP_PROFILE"
  )
  if [[ -n "$MCP_URL" ]]; then
    overlay_args+=(--mcp-url "$MCP_URL")
  fi
  if [[ "$HARDEN_REWARDS" -eq 0 ]]; then
    overlay_args+=(--no-harden-rewards)
  fi
  TASK_PATH=$(python3 scripts/deepswe.py overlay \
    "${overlay_args[@]}")
fi

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2
}

if [[ "${SOURCE_TASK_PATH:-}" != "" ]]; then
  log "Using task overlay: $TASK_PATH (source: $SOURCE_TASK_PATH, mcp_profile=$MCP_PROFILE, harden_rewards=$HARDEN_REWARDS)"
fi

quant_context() {
  local quant=$1
  if [[ -n "$CONTEXT_OVERRIDE" ]]; then
    echo "$CONTEXT_OVERRIDE"
    return 0
  fi
  case "$quant" in
    Q4|Q4_K_XL|Q5|Q5_K_XL|Q6|Q6_K_XL|Q8|Q8_K_XL) echo 262144 ;;
    GEMMA4_*|GEMINI_*|QWEN3_CODER_NEXT*|QWEN_CODER_NEXT*) echo "${TABBY_CONTEXT_SIZE:-32768}" ;;
    *) echo 262144 ;;
  esac
}

free_available_mib() {
  free -m | awk '/^Mem:/ {print $7}'
}

gpu_memory_csv() {
  local out
  out=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ')
  [[ "$out" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  echo "$out"
}

wait_healthy_or_exit() {
  local timeout=$1 elapsed=0 state oom exit_code restarts start_restarts
  local health_path="/health"
  if [[ "$SERVICE" == "tabbyapi" ]]; then
    health_path="/v1/models"
  fi
  start_restarts=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.RestartCount}}' 2>/dev/null || echo 0)
  while [ "$elapsed" -lt "$timeout" ]; do
    if curl -sf "http://localhost:${PORT}${health_path}" > /dev/null 2>&1; then
      return 0
    fi
    state=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.State.Status}}' 2>/dev/null || echo unknown)
    oom=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.State.OOMKilled}}' 2>/dev/null || echo false)
    exit_code=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.State.ExitCode}}' 2>/dev/null || echo 0)
    restarts=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.RestartCount}}' 2>/dev/null || echo "$start_restarts")
    if [[ "$state" == "exited" || "$oom" == "true" || "$exit_code" != "0" || "$restarts" != "$start_restarts" ]]; then
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  return 1
}

wait_vram_stable() {
  local timeout=$1 elapsed=0 last="" stable=0 current
  while [ "$elapsed" -lt "$timeout" ]; do
    current=$(gpu_memory_csv 2>/dev/null || echo "")
    if [[ -n "$current" && "$current" == "$last" ]]; then
      stable=$((stable + 1))
      if [ "$stable" -ge 3 ]; then
        return 0
      fi
    else
      stable=0
      last="$current"
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  return 1
}

write_debug_harness_snapshot() {
  local job_dir=$1 phase=$2
  [[ "$DEBUG_HARNESS" -eq 1 ]] || return 0
  python3 scripts/deepswe_harness.py debug-snapshot \
    --job-dir "$job_dir" \
    --phase "$phase" \
    --service "$SERVICE" || true
}

container_name() {
  echo "ai-box-${SERVICE}-1"
}

write_service_snapshot() {
  local job_dir=$1 phase=${2:-after}
  local out_dir="${job_dir}/service-${phase}"
  local container
  container=$(container_name)
  mkdir -p "$out_dir"
  docker inspect "$container" > "${out_dir}/inspect.json" 2> "${out_dir}/inspect.err" || true
  docker logs --tail 2000 "$container" > "${out_dir}/logs.txt" 2> "${out_dir}/logs.err" || true
  docker stats --no-stream "$container" > "${out_dir}/stats.txt" 2> "${out_dir}/stats.err" || true
}

append_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$@" >> "$OUTPUT_FILE"
}

result_blank_fields=("" "" "" "" "" "" "" "" "" "")
resource_blank_fields=("" "" "" "" "" "")

if [[ ! -f "$OUTPUT_FILE" ]]; then
  python3 scripts/deepswe_harness.py tsv-header > "$OUTPUT_FILE"
fi

ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
  compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1 || true
}
trap cleanup EXIT

log "=== deepswe_sweep: quants=${QUANTS[*]} moes=${MOES[*]} n_tasks=$N_TASKS sample_seed=$SAMPLE_SEED step_limit=${STEP_LIMIT:-none} output=$OUTPUT_FILE ==="
log "Tasks: $TASK_PATH"

export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:${PORT}/v1}"
if [[ "$SERVICE" == "tabbyapi" ]]; then
  export QWEN_BASE_URL="${QWEN_BASE_URL:-$OPENAI_BASE_URL}"
fi

for quant in "${QUANTS[@]}"; do
  ctx=$(quant_context "$quant")
  if [[ "$SERVICE" == "tabbyapi" ]]; then
    model=$(model_tabby "$quant")
    if ! model_tabby_exists "$quant"; then
      log "FAIL $quant — EXL3 model directory missing under ${MODELS_DIR:-/models}"
      append_row "$quant" "$ctx" "-" "-" "${quant}-missing" "-" "missing_model" "1" "${result_blank_fields[@]}" "${resource_blank_fields[@]}" "model directory not found: $model"
      if [[ "$KEEP_GOING" -eq 1 ]]; then
        FAILURES=$((FAILURES + 1))
        continue
      fi
      exit 1
    fi
  else
    model=$(model_gguf "$quant")
  fi
  if [[ "$SERVICE" != "tabbyapi" ]] && ! model_exists "$quant"; then
    log "FAIL $quant — model missing under ${MODELS_DIR:-/models}"
    append_row "$quant" "$ctx" "-" "-" "${quant}-missing" "-" "missing_model" "1" "${result_blank_fields[@]}" "${resource_blank_fields[@]}" "model not found"
    if [[ "$KEEP_GOING" -eq 1 ]]; then
      FAILURES=$((FAILURES + 1))
      continue
    fi
    exit 1
  fi

  for moe in "${MOES[@]}"; do
    if [[ "$SERVICE" == "tabbyapi" ]]; then
      gpu_moe="-"
      job_name="${quant}-ctx${ctx}-tabbyapi"
    else
      gpu_moe=$((48 - moe))
      job_name="${quant}-ctx${ctx}-moe${moe}"
    fi
    job_dir="${JOBS_ROOT}/${job_name}"
    free_before=$(free_available_mib)

    log "=== $job_name ==="
    if [[ "$SERVICE" == "tabbyapi" ]]; then
      set_env_value TABBY_MODEL_NAME "$model" .env
      set_env_value TABBY_CONTEXT_SIZE "$ctx" .env
      set_env_value TABBY_CACHE_SIZE "$ctx" .env
      set_env_value TABBY_BACKEND "${TABBY_BACKEND:-exllamav3}" .env
      tool_format=$(model_tool_format "$quant")
      if [[ -n "$tool_format" ]]; then
        set_env_value TABBY_TOOL_FORMAT "$tool_format" .env
      fi
    else
      set_env_value MODEL_FILE "$model" .env
      set_env_value CPU_MOE_LAYERS "$moe" .env
      set_env_value CONTEXT_SIZE "$ctx" .env
    fi

    compose stop "$SERVICE" > /dev/null 2>&1 || true
    compose rm -sf "$SERVICE" > /dev/null 2>&1 || true
    sleep 3
    compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1 || true

    if ! wait_healthy_or_exit "$LOAD_TIMEOUT"; then
      write_service_snapshot "$job_dir" "load-failed"
      free_after_load=$(free_available_mib)
      mem=$(gpu_memory_csv 2>/dev/null || echo ",")
      vram_used=${mem%,*}
      vram_total=${mem#*,}
      vram_pct=""
      if [[ "$vram_used" =~ ^[0-9]+$ && "$vram_total" =~ ^[0-9]+$ && "$vram_total" -gt 0 ]]; then
        vram_pct=$(python3 -c "print(round(100*${vram_used}/${vram_total}, 1))")
      fi
      append_row "$quant" "$ctx" "$moe" "$gpu_moe" "$job_name" "$job_dir" "load_failed" "1" "${result_blank_fields[@]}" "$free_before" "$free_after_load" "" "$vram_used" "$vram_total" "$vram_pct" "server did not become healthy"
      log "FAIL $job_name — model server never became healthy"
      if [[ "$KEEP_GOING" -eq 1 ]]; then
        FAILURES=$((FAILURES + 1))
        continue
      fi
      exit 1
    fi

    wait_vram_stable 600 || true
    free_after_load=$(free_available_mib)
    mem=$(gpu_memory_csv 2>/dev/null || echo ",")
    vram_used=${mem%,*}
    vram_total=${mem#*,}
    vram_pct=""
    if [[ "$vram_used" =~ ^[0-9]+$ && "$vram_total" =~ ^[0-9]+$ && "$vram_total" -gt 0 ]]; then
      vram_pct=$(python3 -c "print(round(100*${vram_used}/${vram_total}, 1))")
    fi

    pier_exit=0
    set +e
    write_debug_harness_snapshot "$job_dir" "before-pier"
    harness_args=(
      --task-path "$TASK_PATH"
      --model "$MODEL"
      --job-name "$job_name"
      --jobs-dir "$JOBS_ROOT"
      --n-tasks "$N_TASKS"
      --sample-seed "$SAMPLE_SEED"
      --n-concurrent "$N_CONCURRENT"
      --quiet-yes
      --agent "$AGENT"
      --agent-import-path "$AGENT_IMPORT_PATH"
      --environment-import-path "$ENVIRONMENT_IMPORT_PATH"
    )
    if [[ "$DEBUG_HARNESS" -eq 1 ]]; then
      harness_args+=(--debug-harness)
    fi
    mapfile -d '' pier_args < <(python3 scripts/deepswe_harness.py pier-args "${harness_args[@]}" -- "${EXTRA_PIER_ARGS[@]}")
    pier "${pier_args[@]}"
    pier_exit=$?
    write_debug_harness_snapshot "$job_dir" "after-pier"
    set -e
    write_service_snapshot "$job_dir" "after-pier"

    mapfile -d '' summary_fields < <(python3 scripts/deepswe_harness.py summarize-job --job-dir "$job_dir" --pier-exit "$pier_exit")
    finished_at=${summary_fields[0]:-}
    total_trials=${summary_fields[1]:-}
    completed_trials=${summary_fields[2]:-}
    errored_trials=${summary_fields[3]:-}
    cancelled_trials=${summary_fields[4]:-}
    passed_trials=${summary_fields[5]:-}
    pass_rate_pct=${summary_fields[6]:-}
    running_trials=${summary_fields[7]:-}
    pending_trials=${summary_fields[8]:-}
    exception_summary=${summary_fields[9]:-}
    status=${summary_fields[10]:-missing_result}
    note=${summary_fields[11]:-result summary missing}

    free_after_completion=$(free_available_mib)
    append_row "$quant" "$ctx" "$moe" "$gpu_moe" "$job_name" "$job_dir" "$status" "$pier_exit" "$finished_at" "$total_trials" "$completed_trials" "$errored_trials" "$cancelled_trials" "$passed_trials" "$pass_rate_pct" "$running_trials" "$pending_trials" "$exception_summary" "$free_before" "$free_after_load" "$free_after_completion" "$vram_used" "$vram_total" "$vram_pct" "$note"

    log "$status $job_name exit=$pier_exit passed=${passed_trials:-0}/${total_trials:-0}${pass_rate_pct:+ (${pass_rate_pct}%)} errored=${errored_trials:-0} vram=${vram_used}/${vram_total} (${vram_pct}%)"

    if [[ "$status" != "ok" ]]; then
      if [[ "$KEEP_GOING" -eq 1 ]]; then
        log "Continuing after failure because --keep-going is set."
        FAILURES=$((FAILURES + 1))
        continue
      fi
      log "Stopping on first failure or incomplete run."
      exit 1
    fi
  done
done

if [[ "$FAILURES" -gt 0 ]]; then
  log "Completed requested DeepSWE runs with failures=$FAILURES."
  exit 1
fi

log "All requested DeepSWE runs completed successfully."
trap - EXIT
cleanup
