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

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm \
    -e AI_BOX_RUNNER=1 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}" \
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
    runner scripts/deepswe-sweep.sh "$@"
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
  MODELS_DIR="${COMPOSE_MODELS_DIR:-${MODELS_DIR:-/data/ai/models}}" docker compose "$@"
}

SERVICE=turboquant
PORT=8080
TASK_PATH="${DEEPSWE_DIR:-/deep-swe}/tasks"
AGENT=mini-swe-agent
AGENT_IMPORT_PATH=""
MODEL=openai/local
N_TASKS=1
SAMPLE_SEED=0
N_CONCURRENT=1
OUTPUT_FILE=""
MOE_LIST=""
QUANT_LIST="Q4,Q5,Q6,Q8"
LOAD_TIMEOUT=1800
MIN_GPU_MOE=4
EXTRA_PIER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service) SERVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --path) TASK_PATH="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --agent-import-path) AGENT_IMPORT_PATH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --n-tasks) N_TASKS="$2"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --moe) MOE_LIST="$2"; shift 2 ;;
    --quants) QUANT_LIST="$2"; shift 2 ;;
    --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
    --min-gpu-moe) MIN_GPU_MOE="$2"; shift 2 ;;
    --) shift; EXTRA_PIER_ARGS+=("$@"); break ;;
    *) EXTRA_PIER_ARGS+=("$1"); shift ;;
  esac
done

[[ "$SERVICE" == "rotorquant" && "$PORT" == "8080" ]] && PORT=8082

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

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2
}

quant_context() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL|Q5|Q5_K_XL|Q6|Q6_K_XL|Q8|Q8_K_XL) echo 262144 ;;
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
  start_restarts=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.RestartCount}}' 2>/dev/null || echo 0)
  while [ "$elapsed" -lt "$timeout" ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
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

append_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$@" >> "$OUTPUT_FILE"
}

parse_result_json() {
  local path=$1
  python3 - "$path" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("\t".join(["", "", "", "", "", ""]))
    raise SystemExit(0)
with open(path) as f:
    data = json.load(f)
stats = data.get("stats") or {}
print("\t".join([
    str(data.get("finished_at") or ""),
    str(stats.get("n_total_trials") or data.get("n_total_trials") or ""),
    str(stats.get("n_completed_trials") or ""),
    str(stats.get("n_errored_trials") or ""),
    str(stats.get("n_running_trials") or ""),
    str(stats.get("n_pending_trials") or ""),
]))
PY
}

if [[ ! -f "$OUTPUT_FILE" ]]; then
  cat > "$OUTPUT_FILE" <<'EOF'
quant	context	cpu_moe	gpu_moe	job_name	job_dir	status	pier_exit	finished_at	total_trials	completed_trials	errored_trials	running_trials	pending_trials	free_ram_before_mib	free_ram_after_load_mib	free_ram_after_completion_mib	vram_used_mib	vram_total_mib	vram_pct	note
EOF
fi

ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
  compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1 || true
}
trap cleanup EXIT

log "=== deepswe_sweep: quants=${QUANTS[*]} moes=${MOES[*]} n_tasks=$N_TASKS sample_seed=$SAMPLE_SEED output=$OUTPUT_FILE ==="
log "Tasks: $TASK_PATH"

export OPENAI_API_KEY="${OPENAI_API_KEY:-notneeded}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://172.17.0.1:${PORT}/v1}"

for quant in "${QUANTS[@]}"; do
  model=$(model_gguf "$quant")
  ctx=$(quant_context "$quant")
  if ! model_exists "$quant"; then
    log "FAIL $quant — model missing under ${MODELS_DIR:-/models}"
    append_row "$quant" "$ctx" "-" "-" "${quant}-missing" "-" "missing_model" "1" "" "" "" "" "" "" "" "" "" "" "" "model not found"
    exit 1
  fi

  for moe in "${MOES[@]}"; do
    gpu_moe=$((48 - moe))
    job_name="${quant}-ctx${ctx}-moe${moe}"
    job_dir="${JOBS_ROOT}/${job_name}"
    free_before=$(free_available_mib)

    log "=== $job_name ==="
    set_env_value MODEL_FILE "$model" .env
    set_env_value CPU_MOE_LAYERS "$moe" .env
    set_env_value CONTEXT_SIZE "$ctx" .env

    compose stop "$SERVICE" > /dev/null 2>&1 || true
    compose rm -sf "$SERVICE" > /dev/null 2>&1 || true
    sleep 3
    compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1 || true

    if ! wait_healthy_or_exit "$LOAD_TIMEOUT"; then
      free_after_load=$(free_available_mib)
      mem=$(gpu_memory_csv 2>/dev/null || echo ",")
      vram_used=${mem%,*}
      vram_total=${mem#*,}
      vram_pct=""
      if [[ "$vram_used" =~ ^[0-9]+$ && "$vram_total" =~ ^[0-9]+$ && "$vram_total" -gt 0 ]]; then
        vram_pct=$(python3 -c "print(round(100*${vram_used}/${vram_total}, 1))")
      fi
      append_row "$quant" "$ctx" "$moe" "$gpu_moe" "$job_name" "$job_dir" "load_failed" "1" "" "" "" "" "" "" "$free_before" "$free_after_load" "" "$vram_used" "$vram_total" "$vram_pct" "server did not become healthy"
      log "FAIL $job_name — model server never became healthy"
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
    pier_args=(
      -p "$TASK_PATH" \
      --model "$MODEL" \
      --job-name "$job_name" \
      --jobs-dir "$JOBS_ROOT" \
      --n-tasks "$N_TASKS" \
      --sample-seed "$SAMPLE_SEED" \
      --n-concurrent "$N_CONCURRENT" \
      --quiet \
      --yes \
    )
    if [[ -n "$AGENT_IMPORT_PATH" ]]; then
      pier_args+=(--agent-import-path "$AGENT_IMPORT_PATH")
    else
      pier_args+=(--agent "$AGENT")
    fi
    pier_args+=("${EXTRA_PIER_ARGS[@]}")
    pier run "${pier_args[@]}"
    pier_exit=$?
    set -e

    result_json="${job_dir}/result.json"
    read -r finished_at total_trials completed_trials errored_trials running_trials pending_trials < <(parse_result_json "$result_json")

    status="ok"
    note=""
    if [[ "$pier_exit" -ne 0 ]]; then
      status="pier_failed"
      note="pier exited nonzero"
    elif [[ ! -f "$result_json" ]]; then
      status="missing_result"
      note="result.json missing"
    elif [[ "${errored_trials:-0}" != "0" ]]; then
      status="deep_swe_failed"
      note="DeepSWE reported errored trials"
    elif [[ -z "$finished_at" ]]; then
      status="incomplete"
      note="job did not finish"
    fi

    free_after_completion=$(free_available_mib)
    append_row "$quant" "$ctx" "$moe" "$gpu_moe" "$job_name" "$job_dir" "$status" "$pier_exit" "$finished_at" "$total_trials" "$completed_trials" "$errored_trials" "$running_trials" "$pending_trials" "$free_before" "$free_after_load" "$free_after_completion" "$vram_used" "$vram_total" "$vram_pct" "$note"

    log "$status $job_name exit=$pier_exit completed=${completed_trials:-0}/${total_trials:-0} errored=${errored_trials:-0} vram=${vram_used}/${vram_total} (${vram_pct}%)"

    if [[ "$status" != "ok" ]]; then
      log "Stopping on first failure or incomplete run."
      exit 1
    fi
  done
done

log "All requested DeepSWE runs completed successfully."
trap - EXIT
cleanup
