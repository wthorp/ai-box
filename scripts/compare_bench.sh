#!/bin/bash
# compare_bench.sh
# Smoke-bench turboquant vs rotorquant across Q4/Q5/Q6/Q8.
# For each (server, quant) pair: probe max context, run 20-exercise smoke
# bench (python+go), then print a side-by-side summary table.
#
# Usage:
#   ./compare_bench.sh                           # all quants, both servers
#   ./compare_bench.sh --quants Q4,Q5            # subset of quants
#   ./compare_bench.sh --servers turboquant       # one server only
#   ./compare_bench.sh --num-tests 10            # fewer exercises

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/compare_bench.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"
source "${AI_BOX}/scripts/lib/probe_context.sh"

MODELS_DIR="${MODELS_DIR:-/models}"
RESULTS_DIR="${EVAL_RESULTS_DIR}/compare-$(date +%Y%m%d-%H%M%S)"
ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
}
trap cleanup EXIT

QUANT_LIST="Q4,Q5,Q6,Q8"
SERVER_LIST="turboquant,rotorquant"
NUM_TESTS=20
LANGUAGES="python,go"
THREADS=4
RESUME_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --quants)     QUANT_LIST="$2";  shift 2 ;;
    --servers)    SERVER_LIST="$2"; shift 2 ;;
    --num-tests)  NUM_TESTS="$2";   shift 2 ;;
    --languages)  LANGUAGES="$2";   shift 2 ;;
    --threads)    THREADS="$2";     shift 2 ;;
    --resume)     RESUME_DIR="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra QUANTS  <<< "$QUANT_LIST"
IFS=',' read -ra SERVERS <<< "$SERVER_LIST"

mkdir -p "$RESULTS_DIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2; }

model_file() {
  local quant=$1
  case "$quant" in
    Q4) echo "Qwen3-Coder-Next-UD-Q4_K_XL.gguf" ;;
    Q5) echo "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf" ;;
    Q6) echo "Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf" ;;
    Q8) echo "Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf" ;;
    *)  echo "Unknown quant: $quant" >&2; return 1 ;;
  esac
}

server_port() {
  case "$1" in
    turboquant)  echo 8080 ;;
    rotorquant)  echo 8082 ;;
    *) echo "Unknown server: $1" >&2; return 1 ;;
  esac
}

stop_servers() {
  docker compose --profile rotorquant stop turboquant rotorquant \
    > /dev/null 2>&1 || true
  sleep 5
}

wait_healthy() {
  local port=$1 timeout=${2:-300} elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      return 0
    fi
    sleep 5; elapsed=$((elapsed + 5))
  done
  return 1
}

probe_context() {
  local server=$1 model=$2
  local port; port=$(server_port "$server")
  local -a ctx_sizes=(262144 131072 65536 32768 16384)
  [[ "$model" =~ Q8 ]] && ctx_sizes+=(8192)
  for ctx in "${ctx_sizes[@]}"; do
    log "  [$server] probing -c $ctx ..."
    set_env_value MODEL_FILE "$model" "${AI_BOX}/.env"
    set_env_value CONTEXT_SIZE "$ctx" "${AI_BOX}/.env"
    docker compose --profile rotorquant up -d "$server" > /dev/null 2>&1
    if wait_healthy "$port" 300; then
      local used total
      mem=$(gpu_memory_csv) || return 1
      used=$(echo "$mem" | cut -d, -f1)
      total=$(echo "$mem" | cut -d, -f2)
      log "  [$server] OK: -c $ctx | VRAM ${used}/${total} MiB"
      echo "${ctx}:${used}:${total}"
      return 0
    fi
    log "  [$server] OOM or timeout at -c $ctx, backing off"
    docker compose --profile rotorquant stop "$server" > /dev/null 2>&1
    sleep 10
  done
  log "  [$server] ERROR: no context size loaded"
  return 1
}

run_bench() {
  local server=$1 quant=$2 ctx=$3
  local port; port=$(server_port "$server")
  local label="${quant}-${server}"
  local run_name; run_name="$(date +%Y%m%d-%H%M)-${label}"
  local out="${RESULTS_DIR}/${label}.txt"

  log "Bench: $label (ctx=$ctx, tests=$NUM_TESTS, langs=$LANGUAGES)"
  {
    echo "=== $label  ctx=$ctx  $(date) ==="
    timeout 7200 docker compose --profile rotorquant run --rm --no-deps \
      -e OPENAI_API_BASE="http://localhost:${port}/v1" \
      bench \
      "$run_name" \
      --model "openai/${label}" \
      --exercises-dir /bench/exercises \
      --languages "$LANGUAGES" \
      --edit-format whole \
      --threads "$THREADS" \
      --num-ctx "$ctx" \
      --num-tests "$NUM_TESTS" \
      2>&1 || true
  } | tee "$out"
  log "Done: $label → $out"
}

extract_pass_rate() {
  local f=$1
  grep "pass_rate_1:" "$f" 2>/dev/null | tail -1 | awk '{print $2}'
}

extract_secs_per_case() {
  local f=$1
  grep "seconds_per_case:" "$f" 2>/dev/null | tail -1 | awk '{print $2}'
}

log "=== compare_bench starting ==="
log "Quants: ${QUANTS[*]} | Servers: ${SERVERS[*]}"
log "num-tests: $NUM_TESTS | languages: $LANGUAGES | threads: $THREADS"
log "Results → $RESULTS_DIR"
[[ -n "$RESUME_DIR" ]] && log "Resuming from: $RESUME_DIR"

declare -A RESULT_CTX RESULT_VRAM RESULT_PASS RESULT_SPC

if [[ -n "$RESUME_DIR" ]]; then
  for f in "${RESUME_DIR}"/*.txt; do
    [[ -f "$f" ]] || continue
    k=$(basename "$f" .txt)
    RESULT_PASS[$k]=$(extract_pass_rate "$f")
    RESULT_SPC[$k]=$(extract_secs_per_case "$f")
    ctx=$(grep "ctx=" "$f" | head -1 | grep -oP 'ctx=\K[0-9]+' || echo "")
    RESULT_CTX[$k]="${ctx}"
    log "Loaded prior result: $k (pass_rate=${RESULT_PASS[$k]:-?})"
    cp "$f" "${RESULTS_DIR}/" 2>/dev/null || true
  done
fi

stop_servers

for quant in "${QUANTS[@]}"; do
  mf=$(model_file "$quant")
  if [ ! -f "${MODELS_DIR}/${mf}" ]; then
    log "SKIP $quant — model file not found: ${MODELS_DIR}/${mf}"
    continue
  fi

  for server in "${SERVERS[@]}"; do
    key="${quant}-${server}"

    if [[ -f "${RESULTS_DIR}/${key}.txt" ]] && grep -q "pass_rate_1:" "${RESULTS_DIR}/${key}.txt" 2>/dev/null; then
      log "SKIP $key — results already present"
      RESULT_PASS[$key]=$(extract_pass_rate "${RESULTS_DIR}/${key}.txt")
      RESULT_SPC[$key]=$(extract_secs_per_case "${RESULTS_DIR}/${key}.txt")
      ctx=$(grep "ctx=" "${RESULTS_DIR}/${key}.txt" | head -1 | grep -oP 'ctx=\K[0-9]+' || echo "")
      RESULT_CTX[$key]="${ctx}"
      continue
    fi

    log "=== $key ==="
    stop_servers

    ctx_info=$(probe_context "$server" "$mf")
    ctx=$(echo "$ctx_info" | cut -d: -f1)
    vram=$(echo "$ctx_info" | cut -d: -f2)
    total=$(echo "$ctx_info" | cut -d: -f3)
    RESULT_CTX[$key]="$ctx"
    RESULT_VRAM[$key]="${vram}/${total}"

    run_bench "$server" "$quant" "$ctx"
    RESULT_PASS[$key]=$(extract_pass_rate "${RESULTS_DIR}/${key}.txt")
    RESULT_SPC[$key]=$(extract_secs_per_case "${RESULTS_DIR}/${key}.txt")

    stop_servers
  done
done

log "Restoring original .env"
restore_env_file "$ENV_SNAPSHOT" .env
trap - EXIT
docker compose up -d --force-recreate turboquant > /dev/null 2>&1

log "=== All complete ==="
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  %-24s  %8s  %14s  %10s  %10s\n" \
  "Run" "ctx" "VRAM (MiB)" "pass_rate" "sec/case"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for quant in "${QUANTS[@]}"; do
  for server in "${SERVERS[@]}"; do
    key="${quant}-${server}"
    printf "  %-24s  %8s  %14s  %10s  %10s\n" \
      "$key" \
      "${RESULT_CTX[$key]:-—}" \
      "${RESULT_VRAM[$key]:-—}" \
      "${RESULT_PASS[$key]:- —}" \
      "${RESULT_SPC[$key]:- —}"
  done
  echo "  ──────────────────────────────────────────────────────────────────────────"
done
echo ""
echo "Full results: $RESULTS_DIR"
