#!/bin/bash
# light_tune.sh — MoE sweep at max VRAM-filling context per setting.
# For each --n-cpu-moe: probe largest context that loads, then measure tok/s.
# Target: use ~all 24 GB VRAM. Small contexts (8192) are only for Q8 / --allow-small-ctx.
#
# Usage:
#   ./light_tune.sh
#   ./light_tune.sh --moe 34,35,36,37
#   ./light_tune.sh --ctx 65536          # fixed ctx (skip per-moe probe)
#   ./light_tune.sh --allow-small-ctx    # allow probing down to 8192 (Q8 etc.)

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/light_tune.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"
if [[ -f .env ]] && grep -q '^MODELS_DIR=' .env; then
  export COMPOSE_MODELS_DIR=$(grep '^MODELS_DIR=' .env | cut -d= -f2-)
  export MODELS_DIR="$COMPOSE_MODELS_DIR"
fi
compose() {
  MODELS_DIR="${COMPOSE_MODELS_DIR:-${MODELS_DIR:-/data/ai/models}}" docker compose "$@"
}
source "${AI_BOX}/scripts/lib/probe_context.sh"

SERVICE=turboquant
PORT=8080
MOE_LIST="34,35,36,37"
CTX=""              # empty = probe max per moe
MAX_TOKENS=48
MIN_CTX=16384
MIN_VRAM_PCT=75     # warn if GPU memory used is below this % (MoE-on-CPU uses host RAM too)
ALLOW_SMALL_CTX=0
PROMPT="Write one short sentence about sorting algorithms."

while [[ $# -gt 0 ]]; do
  case $1 in
    --service)          SERVICE="$2"; shift 2 ;;
    --port)             PORT="$2";    shift 2 ;;
    --moe)              MOE_LIST="$2"; shift 2 ;;
    --ctx)              CTX="$2";     shift 2 ;;
    --min-ctx)          MIN_CTX="$2"; shift 2 ;;
    --tokens)           MAX_TOKENS="$2"; shift 2 ;;
    --min-vram-pct)     MIN_VRAM_PCT="$2"; shift 2 ;;
    --allow-small-ctx)  ALLOW_SMALL_CTX=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ "$SERVICE" == "rotorquant" && "$PORT" == "8080" ]] && PORT=8082

export ALLOW_SMALL_CTX
export MODEL_FILE
MODEL_FILE=$(env_value MODEL_FILE .env)

ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
}
trap cleanup EXIT

ORIG_MOE=$(env_value CPU_MOE_LAYERS .env)
ORIG_CTX=$(env_value CONTEXT_SIZE .env)
RESULTS="${EVAL_RESULTS_DIR:-/eval-results}/light-tune-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
OUT="${RESULTS}/results.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/run.log" >&2; }

wait_healthy() {
  local timeout=${1:-60} elapsed=0
  while [ $elapsed -lt $timeout ]; do
    curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
    sleep 3
    elapsed=$((elapsed + 3))
  done
  return 1
}

mlock_kib() {
  local pid
  pid=$(docker compose exec -T "$SERVICE" sh -c 'pgrep -f llama-server | head -1' 2>/dev/null || true)
  [[ -z "$pid" ]] && echo "n/a" && return
  docker compose exec -T "$SERVICE" sh -c \
    "grep -E 'Mlocked:' /proc/${pid}/smaps_rollup 2>/dev/null | awk '{print \$2}' | head -1" \
    2>/dev/null || echo "n/a"
}

run_completion() {
  local t0 t1 elapsed body
  t0=$(date +%s.%N)
  body=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n \
      --arg p "$PROMPT" \
      --argjson n "$MAX_TOKENS" \
      '{model:"local",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')" \
    2>/dev/null) || return 1
  t1=$(date +%s.%N)
  elapsed=$(python3 -c "print(round($t1 - $t0, 3))")
  local comp
  comp=$(echo "$body" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  [[ "$comp" == "0" || -z "$comp" ]] && comp=$(echo "$body" | jq -r '.choices[0].message.content | length' 2>/dev/null)
  python3 -c "c=float('$comp'); e=float('$elapsed'); print(f'{c} {e} {c/e if e>0 else 0:.1f}')" 2>/dev/null
}

vram_pct() {
  local used total
  used=$(echo "$1" | cut -d/ -f1)
  total=$(echo "$1" | cut -d/ -f2)
  python3 -c "print(round(100*float('$used')/float('$total'), 1))" 2>/dev/null || echo "?"
}

echo -e "moe\tctx\tvram_mib\tvram_pct\tmlock_kib\tcompletion_tokens\tseconds\ttok_per_s" > "$OUT"

log "=== light_tune: service=$SERVICE model=$MODEL_FILE ==="
log "MoE sweep: $MOE_LIST | min_ctx=$MIN_CTX | target: max ctx per MoE (GPU+CPU RAM; warn if GPU <${MIN_VRAM_PCT}%)"
[[ -n "$CTX" ]] && log "Fixed context: $CTX (probe disabled)" \
  || log "Per-moe max context probe (8192 only with Q8 or --allow-small-ctx)"
log "Baseline .env: CPU_MOE_LAYERS=$ORIG_MOE CONTEXT_SIZE=$ORIG_CTX"

IFS=',' read -ra MOES <<< "$MOE_LIST"
BEST_MOE="" BEST_TPS=0

for moe in "${MOES[@]}"; do
  log "--- n-cpu-moe=$moe ---"
  if [[ -n "$CTX" ]]; then
    set_env_value CPU_MOE_LAYERS "$moe" .env
    set_env_value CONTEXT_SIZE "$CTX" .env
    docker compose stop "$SERVICE" > /dev/null 2>&1 || true
    sleep 5
    docker compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1
    if ! wait_healthy 360; then
      log "  FAIL: health timeout at ctx=$CTX"
      echo -e "${moe}\t${CTX}\tOOM\t-\t-\t-\t-\t-" >> "$OUT"
      continue
    fi
    run_ctx=$CTX
    vram=$(gpu_memory_csv | tr ',' '/') || {
      log "  FAIL: nvidia-smi unavailable"
      echo -e "${moe}\t${CTX}\tGPU_UNAVAILABLE\t-\t-\t-\t-\t-" >> "$OUT"
      continue
    }
  else
    log "  Probing max context (filling VRAM)..."
    if ! probe_max_context "$SERVICE" "$PORT" "$moe" "$MIN_CTX"; then
      log "  FAIL: no context >= $MIN_CTX loaded"
      echo -e "${moe}\t-\tFAIL\t-\t-\t-\t-\t-" >> "$OUT"
      continue
    fi
    run_ctx=$PROBE_CTX
    vram=$PROBE_VRAM
    log "  Loaded -c $run_ctx | VRAM ${vram} MiB"
  fi

  pct=$(vram_pct "$vram")
  if python3 -c "import sys; sys.exit(0 if float('${pct}') >= float('${MIN_VRAM_PCT}') else 1)" 2>/dev/null; then
    :
  else
    log "  WARN: VRAM ${pct}% < ${MIN_VRAM_PCT}% target — context may be too small for this quant"
  fi

  sleep 2
  mlock=$(mlock_kib)
  if ! stats=$(run_completion); then
    log "  FAIL: completion error"
    echo -e "${moe}\t${run_ctx}\t${vram}\t${pct}\t${mlock}\t-\t-\t-" >> "$OUT"
    continue
  fi
  read -r comp sec tps <<< "$stats"
  log "  ctx=$run_ctx VRAM ${vram} (${pct}%) | ${comp} tok in ${sec}s => ${tps} tok/s"
  echo -e "${moe}\t${run_ctx}\t${vram}\t${pct}\t${mlock}\t${comp}\t${sec}\t${tps}" >> "$OUT"
  if python3 -c "import sys; sys.exit(0 if float('${tps}') > float('${BEST_TPS}') else 1)"; then
    BEST_TPS=$tps
    BEST_MOE=$moe
  fi
done

log "Restoring original .env"
restore_env_file "$ENV_SNAPSHOT" .env
trap - EXIT
docker compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LIGHT TUNE (max VRAM context per MoE — compare tok/s at full GPU use)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
column -t "$OUT" 2>/dev/null || cat "$OUT"
echo ""
if [[ -n "$BEST_MOE" ]]; then
  echo "Best tok/s in sweep: CPU_MOE_LAYERS=${BEST_MOE} (${BEST_TPS} tok/s at each row's ctx)"
  echo "For production: ./start.sh $SERVICE  # max ctx + MoE from VRAM model"
fi
echo "Full log: $RESULTS"
