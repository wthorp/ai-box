#!/bin/bash
# multi_quant_eval.sh
# Downloads UD-Q5_K_XL / UD-Q6_K_XL / UD-Q8_K_XL, probes max context per quant,
# and runs the Aider polyglot benchmark on each. Downloads next quant while
# the current eval runs.
#
# Usage: ./multi_quant_eval.sh [--languages python,go,rust,javascript] [--threads 1]
#
# Prerequisites: docker compose up -d turboquant (done automatically per quant)

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/multi_quant_eval.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"
source "${AI_BOX}/scripts/lib/probe_context.sh"

MODELS_DIR="${MODELS_DIR:-/models}"
HF_BASE="https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main"
RESULTS_DIR="${EVAL_RESULTS_DIR}/multi-quant-$(date +%Y%m%d-%H%M%S)"
NPARTS=3
ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
}
trap cleanup EXIT

BENCH_LANGUAGES="${BENCH_LANGUAGES:-python,go,rust,javascript}"
BENCH_THREADS="${BENCH_THREADS:-1}"
BENCH_EDIT_FORMAT="${BENCH_EDIT_FORMAT:-whole}"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2; }

download_quant() {
  local quant=$1
  log "Download start: $quant"
  for i in $(seq 1 $NPARTS); do
    local part; part=$(printf "%05d" "$i")
    local total; total=$(printf "%05d" "$NPARTS")
    local fname="Qwen3-Coder-Next-${quant}-${part}-of-${total}.gguf"
    local dest="${MODELS_DIR}/${fname}"
    if [ -f "$dest" ]; then
      log "  $fname already present, skipping"
    else
      log "  Fetching $fname ..."
      wget -q --show-progress -c \
        "${HF_BASE}/${quant}/${fname}" \
        -O "${dest}.tmp" \
        && mv "${dest}.tmp" "$dest"
    fi
  done
  log "Download done: $quant"
}

wait_healthy() {
  local timeout=${1:-300}
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
      return 0
    fi
    local state
    state=$(docker inspect --format='{{.State.Status}}' ai-box-turboquant-1 2>/dev/null || echo "unknown")
    if [ "$state" = "exited" ]; then
      log "  turboquant exited early"
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  return 1
}

probe_context() {
  local model_file=$1
  local -a ctx_sizes=(262144 131072 65536 32768 16384)
  [[ "$model_file" =~ Q8 ]] && ctx_sizes+=(8192)
  for ctx in "${ctx_sizes[@]}"; do
    log "  Probing -c $ctx ..."
    set_env_value MODEL_FILE "$model_file" "${AI_BOX}/.env"
    set_env_value CONTEXT_SIZE "$ctx" "${AI_BOX}/.env"
    docker compose up -d turboquant > /dev/null 2>&1
    if wait_healthy 300; then
      local used total
      mem=$(gpu_memory_csv) || return 1
      used=$(echo "$mem" | cut -d, -f1)
      total=$(echo "$mem" | cut -d, -f2)
      log "  OK: -c $ctx | VRAM ${used}/${total} MiB"
      echo "${ctx}:${used}:${total}"
      return 0
    else
      log "  OOM or timeout at -c $ctx, backing off"
      docker compose stop turboquant > /dev/null 2>&1
      sleep 10
    fi
  done
  log "ERROR: no context size loaded for $model_file"
  return 1
}

run_bench() {
  local quant=$1
  local ctx=$2
  local run_name
  run_name="$(date +%Y%m%d-%H%M)-${quant}"
  local out="${RESULTS_DIR}/${quant}.txt"

  log "Bench start: $quant (ctx=$ctx, langs=${BENCH_LANGUAGES})"
  {
    echo "=== $quant  ctx=$ctx  $(date) ==="
    docker compose run --rm --no-deps bench \
      "$run_name" \
      --model "openai/${quant}" \
      --exercises-dir /bench/exercises \
      --languages "$BENCH_LANGUAGES" \
      --edit-format "$BENCH_EDIT_FORMAT" \
      --threads "$BENCH_THREADS" \
      --num-ctx "$ctx" \
      2>&1
  } | tee "$out"
  log "Bench done: $quant → $out"
}

QUANTS=("UD-Q5_K_XL" "UD-Q6_K_XL" "UD-Q8_K_XL")

log "=== Multi-quant Aider benchmark starting ==="
log "Results → $RESULTS_DIR"
log "Languages: $BENCH_LANGUAGES | threads: $BENCH_THREADS | edit-format: $BENCH_EDIT_FORMAT"

download_quant "${QUANTS[0]}" &
DL_PID=$!

for i in "${!QUANTS[@]}"; do
  quant="${QUANTS[$i]}"

  log "Waiting for $quant download..."
  wait $DL_PID

  DL_PID=0
  if [ $((i + 1)) -lt ${#QUANTS[@]} ]; then
    next="${QUANTS[$((i + 1))]}"
    download_quant "$next" &
    DL_PID=$!
  fi

  log "Switching turboquant → $quant"
  docker compose stop turboquant > /dev/null 2>&1
  sleep 5

  gguf_first="Qwen3-Coder-Next-${quant}-00001-of-00003.gguf"
  ctx_info=$(probe_context "$gguf_first")
  ctx=$(echo "$ctx_info" | cut -d: -f1)

  run_bench "$quant" "$ctx"
done

[ "$DL_PID" -ne 0 ] && wait $DL_PID 2>/dev/null || true

log "Restoring original .env"
docker compose stop turboquant > /dev/null 2>&1
restore_env_file "$ENV_SNAPSHOT" .env
trap - EXIT
docker compose up -d --force-recreate turboquant > /dev/null 2>&1

log "=== All benchmarks complete ==="
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MULTI-QUANT AIDER BENCHMARK SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for quant in "${QUANTS[@]}"; do
  out="${RESULTS_DIR}/${quant}.txt"
  [ -f "$out" ] || continue
  echo ""
  echo "── $quant ──"
  grep -E "pass|fail|error|correct|pct|%|score|Percent" "$out" \
    | grep -v "^===" | head -10
done
echo ""
echo "Full results: $RESULTS_DIR"
