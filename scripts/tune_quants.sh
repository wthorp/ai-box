#!/bin/bash
# tune_quants.sh — Ideal CPU_MOE_LAYERS + CONTEXT_SIZE per quant (Q4/Q5/Q6/Q8).
#
# Method (fixed): for each quant and each MoE value, probe the largest context
# that loads with that MoE (min 65536 for Q4–Q6), wait for stable VRAM, benchmark.
# Pick the MoE+ctx with >=85% GPU VRAM and best tok/s.
#
# Usage:
#   ./tune_quants.sh
#   ./tune_quants.sh --quants Q5,Q6,Q8
#   ./tune_quants.sh --quants Q4 --moe 24,28,32

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/tune_quants.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"
# Runner sets MODELS_DIR=/models for file checks; nested compose needs the host path from .env
if [[ -f .env ]] && grep -q '^MODELS_DIR=' .env; then
  export COMPOSE_MODELS_DIR=$(grep '^MODELS_DIR=' .env | cut -d= -f2-)
  export MODELS_DIR="$COMPOSE_MODELS_DIR"
fi
compose() {
  MODELS_DIR="${COMPOSE_MODELS_DIR:-${MODELS_DIR:-/data/ai/models}}" docker compose "$@"
}
source "${AI_BOX}/scripts/lib/probe_context.sh"
source "${AI_BOX}/scripts/lib/models.sh"

SERVICE=rotorquant
PORT=8080
QUANT_LIST="Q4,Q5,Q6,Q8"
MOE_LIST="20,24,28,32,36,40"
MAX_TOKENS=32
TARGET_VRAM_PCT=85
MIN_VRAM_SPREAD=1500   # warn if all MoE points within this MiB (stale readings)
PROMPT="Explain quicksort in one paragraph."

while [[ $# -gt 0 ]]; do
  case $1 in
    --quants) QUANT_LIST="$2"; shift 2 ;;
    --moe)    MOE_LIST="$2";   shift 2 ;;
    --tokens) MAX_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/run.log" >&2; }

vram_pct() {
  python3 -c "print(round(100*float('$1')/float('$2'), 1))" 2>/dev/null
}

quant_min_ctx() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL) echo 65536 ;;
    Q5|Q5_K_XL) echo 32768 ;;
    Q6|Q6_K_XL) echo 16384 ;;
    Q8|Q8_K_XL) echo 8192 ;;
    *)          echo 32768 ;;
  esac
}

quant_probe_vram_pct() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL) echo 70 ;;
    Q5|Q5_K_XL) echo 45 ;;   # moe~36 lands ~71%; moe~32 ~54%
    Q6|Q6_K_XL) echo 40 ;;   # moe~44 @32k ~65%; allow moe~40 ~48%
    Q8|Q8_K_XL) echo 40 ;;
    *)          echo 45 ;;
  esac
}

quant_target_vram_pct() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL) echo 85 ;;
    Q5|Q5_K_XL) echo 65 ;;
    Q6|Q6_K_XL) echo 60 ;;
    Q8|Q8_K_XL) echo 55 ;;
    *)          echo 65 ;;
  esac
}

# Lower n-cpu-moe = more GPU experts = higher VRAM %. Heavier quants need higher moe #s.
quant_moe_list() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL) echo "20,24,28,32,36,40" ;;
    Q5|Q5_K_XL) echo "32,36,40" ;;       # 28 OOMs @32k; 44 too little GPU
    Q6|Q6_K_XL) echo "40,44,48" ;;       # 36 OOMs @32k
    Q8|Q8_K_XL) echo "44,48" ;;         # 40 OOMs @16k
    *)          echo "32,36,40,44" ;;
  esac
}

quant_max_ctx() {
  local quant=$1
  case "$quant" in
    Q5|Q5_K_XL) echo 32768 ;;
    Q6|Q6_K_XL) echo 32768 ;;   # probe cap; min_ctx 16384
    Q8|Q8_K_XL) echo 16384 ;;
    *)          echo 262144 ;;
  esac
}

run_completion_tps() {
  local t0 t1 body comp
  t0=$(date +%s.%N)
  body=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
      '{model:"local",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')" \
    2>/dev/null) || return 1
  t1=$(date +%s.%N)
  comp=$(echo "$body" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  python3 -c "c=float('$comp'); e=float('$t1')-float('$t0'); print(f'{c} {e} {c/e if e>0 else 0:.2f}')" 2>/dev/null
}

hard_stop_service() {
  compose stop "$SERVICE" > /dev/null 2>&1 || true
  compose rm -sf "$SERVICE" > /dev/null 2>&1 || true
  sleep 8
}

reload_for_benchmark() {
  local moe=$1 ctx=$2 model=$3 elapsed=0
  set_env_value MODEL_FILE "$model" .env
  set_env_value CPU_MOE_LAYERS "$moe" .env
  set_env_value CONTEXT_SIZE "$ctx" .env
  export MODEL_FILE="$model"
  recreate_service "$SERVICE"
  while [ $elapsed -lt 420 ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      break
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || return 1
  wait_vram_stable 12000 90 420
}

update_memory_md() {
  local summary=$1 run_dir=$2
  local memory="${AI_BOX}/MEMORY.md"
  [[ -f "$memory" ]] || return 0
  python3 - "$memory" "$summary" "$run_dir" <<'PY'
import sys
from datetime import date

memory_path, summary_path, run_dir = sys.argv[1:4]
rows = []
with open(summary_path) as f:
    header = f.readline()
    for line in f:
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 8 or cols[2] in ("-", ""):
            continue
        quant, moe, ctx, _, _, tps, note = cols[0], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]
        rows.append(f"| {quant} | {moe} | {ctx} | {note} (~{tps} tok/s) |")

block = [
    "",
    f"Updated: {date.today()} — `{run_dir}`",
    "",
    "| Quant | CPU_MOE | CONTEXT | Note |",
    "|-------|---------|---------|------|",
    *rows,
    "",
]
with open(memory_path) as f:
    lines = f.read().splitlines()
out, skip = [], False
for line in lines:
    if "<!-- TUNE_QUANTS:START -->" in line:
        out.append(line)
        out.extend(block)
        skip = True
        continue
    if "<!-- TUNE_QUANTS:END -->" in line:
        skip = False
        out.append(line)
        continue
    if not skip:
        out.append(line)
with open(memory_path, "w") as f:
    f.write("\n".join(out) + "\n")
PY
}

ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
}
trap cleanup EXIT

ORIG_MODEL=$(env_value MODEL_FILE .env)
ORIG_MOE=$(env_value CPU_MOE_LAYERS .env)
ORIG_CTX=$(env_value CONTEXT_SIZE .env)

RESULTS="${EVAL_RESULTS_DIR:-/eval-results}/tune-quants-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
DETAIL="${RESULTS}/detail.tsv"
SUMMARY="${RESULTS}/recommendations.tsv"
echo -e "quant\tmoe\tctx\tvram_used\tvram_total\tvram_pct\ttok_per_s" > "$DETAIL"
echo -e "quant\tmodel\tcpu_moe_layers\tcontext_size\tvram_pct\ttok_per_s\tnote" > "$SUMMARY"

IFS=',' read -ra QUANTS <<< "$QUANT_LIST"
IFS=',' read -ra MOES <<< "$MOE_LIST"

log "=== tune_quants (per-MoE max ctx): quants=${QUANTS[*]} default_moe=${MOES[*]} ==="

for quant in "${QUANTS[@]}"; do
  if ! model_exists "$quant"; then
    log "SKIP $quant — model not found under ${MODELS_DIR:-/models}"
    echo -e "${quant}\t-\t-\t-\t-\t-\tSKIP missing model" >> "$SUMMARY"
    continue
  fi
  mf=$(model_gguf "$quant")
  min_ctx=$(quant_min_ctx "$quant")
  probe_vram_pct=$(quant_probe_vram_pct "$quant")
  TARGET_VRAM_PCT=$(quant_target_vram_pct "$quant")
  IFS=',' read -ra QUANT_MOES <<< "$(quant_moe_list "$quant")"
  export MODEL_FILE="$mf"
  ALLOW_SMALL_CTX=0
  [[ "$quant" == "Q8" || "$mf" =~ Q8 ]] && ALLOW_SMALL_CTX=1
  export ALLOW_SMALL_CTX

  log "=== $quant ($mf) min_ctx=$min_ctx probe>=${probe_vram_pct}% pick>=${TARGET_VRAM_PCT}% moe=${QUANT_MOES[*]} ==="
  set_env_value MODEL_FILE "$mf" .env
  hard_stop_service

  best_moe="" best_ctx="" best_tps=0 best_pct=0
  best_vram_moe="" best_vram_ctx="" best_vram_pct=0 best_vram_tps=0
  fallback_moe="" fallback_ctx="" fallback_tps=0
  vram_min=999999 vram_max=0

  for moe in "${QUANT_MOES[@]}"; do
    log "  $quant moe=$moe — probe max context (min $min_ctx, need >=${probe_vram_pct}% VRAM) ..."
    set_env_value CPU_MOE_LAYERS "$moe" .env
    PROBE_MIN_VRAM_PCT=$probe_vram_pct
    PROBE_MAX_CTX=$(quant_max_ctx "$quant")
    export PROBE_MIN_VRAM_PCT PROBE_MAX_CTX
    if ! probe_max_context "$SERVICE" "$PORT" "$moe" "$min_ctx"; then
      log "    retry probe without VRAM gate ..."
      PROBE_MIN_VRAM_PCT=0
      export PROBE_MIN_VRAM_PCT
      if ! probe_max_context "$SERVICE" "$PORT" "$moe" "$min_ctx"; then
        log "    FAIL probe (no ctx >= $min_ctx)"
        echo -e "${quant}\t${moe}\t-\t-\t-\t-\t-" >> "$DETAIL"
        continue
      fi
    fi
    run_ctx=$PROBE_CTX
    log "    probed ctx=$run_ctx VRAM ${PROBE_VRAM}"

    if ! reload_for_benchmark "$moe" "$run_ctx" "$mf"; then
      log "    FAIL reload/VRAM stable"
      echo -e "${quant}\t${moe}\t${run_ctx}\t-\t-\t-\t-" >> "$DETAIL"
      continue
    fi
    mem=$(gpu_memory_csv) || {
      log "    FAIL nvidia-smi unavailable"
      echo -e "${quant}\t${moe}\t${run_ctx}\t-\t-\t-\t-" >> "$DETAIL"
      continue
    }
    used=$(echo "$mem" | cut -d, -f1)
    total=$(echo "$mem" | cut -d, -f2)
    pct=$(vram_pct "$used" "$total")
    if ! stats=$(run_completion_tps); then
      log "    FAIL completion"
      echo -e "${quant}\t${moe}\t${run_ctx}\t${used}\t${total}\t${pct}\t-" >> "$DETAIL"
      continue
    fi
    read -r _ sec tps <<< "$stats"
    log "    ctx=$run_ctx VRAM ${used}/${total} (${pct}%) => ${tps} tok/s"
    echo -e "${quant}\t${moe}\t${run_ctx}\t${used}\t${total}\t${pct}\t${tps}" >> "$DETAIL"

    if [ "$used" -lt "$vram_min" ]; then vram_min=$used; fi
    if [ "$used" -gt "$vram_max" ]; then vram_max=$used; fi

    if python3 -c "import sys; sys.exit(0 if float('${pct}') > float('${best_vram_pct}') else 1)"; then
      best_vram_pct=$pct
      best_vram_moe=$moe
      best_vram_ctx=$run_ctx
      best_vram_tps=$tps
    fi
    if python3 -c "import sys; sys.exit(0 if float('${pct}') >= float('${TARGET_VRAM_PCT}') else 1)"; then
      if python3 -c "import sys; sys.exit(0 if float('${tps}') > float('${best_tps}') else 1)"; then
        best_tps=$tps
        best_moe=$moe
        best_ctx=$run_ctx
        best_pct=$pct
      fi
    fi
    if [[ -z "$fallback_moe" ]] || python3 -c "import sys; sys.exit(0 if float('${tps}') > float('${fallback_tps}') else 1)"; then
      fallback_moe=$moe
      fallback_ctx=$run_ctx
      fallback_tps=$tps
    fi
  done

  spread=$(( vram_max - vram_min ))
  if [ "$vram_max" -gt 0 ] && [ "$spread" -lt "$MIN_VRAM_SPREAD" ]; then
    log "  WARN $quant: VRAM spread only ${spread} MiB (${vram_min}-${vram_max}) — readings may be stale"
  fi

  pick_moe=$best_moe
  pick_ctx=$best_ctx
  pick_note=">=${TARGET_VRAM_PCT}% VRAM, best tok/s"
  pick_tps=$best_tps
  pick_pct=$best_pct
  if [[ -z "$pick_moe" ]]; then
    pick_moe=$best_vram_moe
    pick_ctx=$best_vram_ctx
    pick_tps=$best_vram_tps
    pick_pct=$best_vram_pct
    pick_note="fallback: highest GPU VRAM"
  fi
  if [[ -z "$pick_moe" ]]; then
    pick_moe=$fallback_moe
    pick_ctx=$fallback_ctx
    pick_tps=$fallback_tps
    pick_pct=$(awk -F'\t' -v q="$quant" -v m="$pick_moe" '$1==q && $2==m {print $6; exit}' "$DETAIL")
    pick_note="fallback: best tok/s (no ${TARGET_VRAM_PCT}% VRAM point)"
  fi
  if [[ -z "$pick_moe" ]]; then
    log ">>> $quant: no successful points"
    echo -e "${quant}\t${mf}\t-\t-\t-\t-\tall failed" >> "$SUMMARY"
    continue
  fi

  log ">>> $quant recommend: CPU_MOE_LAYERS=${pick_moe} CONTEXT_SIZE=${pick_ctx} (${pick_note}, ${pick_tps} tok/s, ${pick_pct}% VRAM)"
  echo -e "${quant}\t${mf}\t${pick_moe}\t${pick_ctx}\t${pick_pct}\t${pick_tps}\t${pick_note}" >> "$SUMMARY"
done

log "Restoring original .env: $ORIG_MODEL moe=$ORIG_MOE ctx=$ORIG_CTX"
restore_env_file "$ENV_SNAPSHOT" .env
trap - EXIT
compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1

{
  echo "# Generated by tune_quants.sh — copy into .env per workload"
  while IFS=$'\t' read -r quant mf moe ctx _ tps note; do
    [[ "$quant" == "quant" ]] && continue
    [[ "$moe" == "-" ]] && continue
    echo ""
    echo "# $quant ($note, ~${tps} tok/s)"
    echo "# MODEL_FILE=${mf}"
    echo "# CPU_MOE_LAYERS=${moe}"
    echo "# CONTEXT_SIZE=${ctx}"
  done < "$SUMMARY"
} > "${RESULTS}/recommendations.env"

update_memory_md "$SUMMARY" "$RESULTS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PER-QUANT RECOMMENDATIONS (per-MoE max ctx)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo ""
echo "Detail: $DETAIL"
echo "Env snippet: ${RESULTS}/recommendations.env"
echo "Log: ${RESULTS}/run.log"
echo "MEMORY.md updated (TUNE_QUANTS block)"
