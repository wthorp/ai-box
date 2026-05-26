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

set -euo pipefail

AI_BOX="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${BENCH_RESULTS_DIR:-/data/ai/local/eval-results}/smoke-$(date +%Y%m%d-%H%M%S)"
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

# ── Probe VRAM at max context ─────────────────────────────────────────────────
probe_context() {
  for ctx in 65536 32768 16384 8192; do
    log "  Probing -c $ctx ..."
    # Adjust .env and restart if running the managed server
    if [[ "$SERVER" == "turboquant" || "$SERVER" == "rotorquant" ]]; then
      sed -i "s|^CONTEXT_SIZE=.*|CONTEXT_SIZE=${ctx}|" "${AI_BOX}/.env"
      (cd "$AI_BOX" && docker compose restart "$SERVER") > /dev/null 2>&1
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
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
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
  (cd "$AI_BOX" && OPENAI_API_BASE="http://localhost:${PORT}/v1" \
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
      2>&1)
} | tee "$out"

log "Done → $out"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SMOKE BENCH SUMMARY  ($MODEL_LABEL, ctx=$ctx)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -E "pass_rate|percent|correct|score|total_tests|seconds_per" "$out" | head -15
echo ""
echo "Full results: $RESULTS_DIR"
