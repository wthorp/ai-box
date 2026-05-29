#!/bin/bash
# smoke_bench.sh
# Quick benchmark: 20 exercises across python+go at max context.
# Completes in ~20 min. Use to validate a model/quant before a full run.
#
# Usage:
#   ./smoke_bench.sh                        # targets turboquant (port 8080)
#   ./smoke_bench.sh --server rotorquant    # targets rotorquant (port 8082)
#   ./smoke_bench.sh --port 8080            # explicit port
#   ./smoke_bench.sh --model my-model-name  # override model label

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/smoke_bench.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"
source "${AI_BOX}/scripts/lib/probe_context.sh"

MODEL_FILE=$(env_value MODEL_FILE "${AI_BOX}/.env")
export MODEL_FILE
ENV_SNAPSHOT=$(save_env_file "${AI_BOX}/.env")
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" "${AI_BOX}/.env"
}
trap cleanup EXIT

RESULTS_DIR="${EVAL_RESULTS_DIR}/smoke-$(date +%Y%m%d-%H%M%S)"
SERVER="turboquant"
PORT=""
MODEL_LABEL=""
NUM_TESTS=20
LANGUAGES="python,go"

while [[ $# -gt 0 ]]; do
  case $1 in
    --server)   SERVER="$2";      shift 2 ;;
    --port)     PORT="$2";        shift 2 ;;
    --model)    MODEL_LABEL="$2"; shift 2 ;;
    --num-tests) NUM_TESTS="$2";  shift 2 ;;
    --languages) LANGUAGES="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$PORT" ]]; then
  case "$SERVER" in
    turboquant)  PORT=8080 ;;
    rotorquant)  PORT=8082 ;;
    *) echo "Unknown server '$SERVER'; use --port explicitly"; exit 1 ;;
  esac
fi

[[ -z "$MODEL_LABEL" ]] && MODEL_LABEL="$SERVER"

mkdir -p "$RESULTS_DIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2; }

log "=== Smoke bench: server=$SERVER port=$PORT model=$MODEL_LABEL ==="
log "Languages: $LANGUAGES | num-tests: $NUM_TESTS"
log "Results → $RESULTS_DIR"

# ── Verify server is healthy ──────────────────────────────────────────────────
if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  log "ERROR: server not healthy at http://localhost:${PORT}/health"
  exit 1
fi

# ── Probe VRAM at max context (fill GPU; 8192 only for Q8) ───────────────────
probe_context() {
  local -a ctx_sizes=(262144 131072 65536 32768 16384)
  [[ "${MODEL_FILE:-}" =~ Q8 ]] && ctx_sizes+=(8192)
  for ctx in "${ctx_sizes[@]}"; do
    log "  Probing -c $ctx ..."
    if [[ "$SERVER" == "turboquant" || "$SERVER" == "rotorquant" ]]; then
      set_env_value CONTEXT_SIZE "$ctx" "${AI_BOX}/.env"
      docker compose restart "$SERVER" > /dev/null 2>&1
      local elapsed=0
      while [ $elapsed -lt 180 ]; do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then break; fi
        sleep 5; elapsed=$((elapsed + 5))
      done
      if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        log "  OOM or timeout at -c $ctx, backing off"
        continue
      fi
    fi
    local used total
    mem=$(gpu_memory_csv) || return 1
    used=$(echo "$mem" | cut -d, -f1)
    total=$(echo "$mem" | cut -d, -f2)
    log "  OK: -c $ctx | VRAM ${used}/${total} MiB"
    echo "${ctx}:${used}:${total}"
    return 0
  done
  log "ERROR: no context size worked"
  return 1
}

ctx_info=$(probe_context)
ctx=$(echo "$ctx_info" | cut -d: -f1)

# ── Run benchmark subset ──────────────────────────────────────────────────────
run_name="$(date +%Y%m%d-%H%M)-smoke-${MODEL_LABEL}"
out="${RESULTS_DIR}/results.txt"

log "Bench start (ctx=$ctx, tests=$NUM_TESTS, langs=$LANGUAGES)"
{
  echo "=== smoke-bench  server=$SERVER  ctx=$ctx  $(date) ==="
  OPENAI_API_BASE="http://localhost:${PORT}/v1" \
    docker compose run --rm --no-deps \
      -e OPENAI_API_BASE="http://localhost:${PORT}/v1" \
      bench \
      "$run_name" \
      --model "openai/${MODEL_LABEL}" \
      --exercises-dir /bench/exercises \
      --languages "$LANGUAGES" \
      --edit-format whole \
      --threads 4 \
      --num-ctx "$ctx" \
      --num-tests "$NUM_TESTS" \
      2>&1
} | tee "$out"

log "Done → $out"
restore_env_file "$ENV_SNAPSHOT" "${AI_BOX}/.env"
trap - EXIT
if [[ "$SERVER" == "turboquant" || "$SERVER" == "rotorquant" ]]; then
  docker compose up -d --force-recreate "$SERVER" > /dev/null 2>&1
fi
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SMOKE BENCH SUMMARY  ($MODEL_LABEL, ctx=$ctx)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -E "pass_rate|percent|correct|score|total_tests|seconds_per" "$out" | head -15
echo ""
echo "Full results: $RESULTS_DIR"
