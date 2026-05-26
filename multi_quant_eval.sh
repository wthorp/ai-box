#!/bin/bash
# multi_quant_eval.sh
# Downloads UD-Q5_K_XL / UD-Q6_K_XL / UD-Q8_K_XL, probes max context per quant,
# and runs the Aider polyglot benchmark on each. Downloads next quant while
# the current eval runs.
#
# Usage: ./multi_quant_eval.sh [--languages python,go,rust,javascript] [--threads 1]
#
# Prerequisites: docker compose up -d turboquant (done automatically per quant)

set -euo pipefail

MODELS_DIR=/data/ai/models
AI_BOX="$(cd "$(dirname "$0")" && pwd)"
HF_BASE="https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main"
RESULTS_DIR="${BENCH_RESULTS_DIR:-/data/ai/local/eval-results}/multi-quant-$(date +%Y%m%d-%H%M%S)"
ORIGINAL_MODEL="Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
ORIGINAL_CTX=8192
NPARTS=3

# Pass-through to aider benchmark (override via CLI args)
BENCH_LANGUAGES="${BENCH_LANGUAGES:-python,go,rust,javascript}"
BENCH_THREADS="${BENCH_THREADS:-1}"
BENCH_EDIT_FORMAT="${BENCH_EDIT_FORMAT:-whole}"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS_DIR}/run.log" >&2; }

# ── Download all parts of a multi-shard GGUF ─────────────────────────────────
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

# ── Wait for turboquant /health ───────────────────────────────────────────────
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

# ── Probe max context that loads without OOM ─────────────────────────────────
# Prints "ctx:vram_used_mib:vram_total_mib" on success
probe_context() {
  local model_file=$1
  for ctx in 65536 32768 16384 8192; do
    log "  Probing -c $ctx ..."
    sed -i \
      "s|^MODEL_FILE=.*|MODEL_FILE=${model_file}|; s|^CONTEXT_SIZE=.*|CONTEXT_SIZE=${ctx}|" \
      "${AI_BOX}/.env"
    (cd "$AI_BOX" && docker compose up -d turboquant) > /dev/null 2>&1
    if wait_healthy 300; then
      local used total
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
             | head -1 | tr -d ' ')
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
              | head -1 | tr -d ' ')
      log "  OK: -c $ctx | VRAM ${used}/${total} MiB"
      echo "${ctx}:${used}:${total}"
      return 0
    else
      log "  OOM or timeout at -c $ctx, backing off"
      (cd "$AI_BOX" && docker compose stop turboquant) > /dev/null 2>&1
      sleep 10
    fi
  done
  log "ERROR: no context size loaded for $model_file"
  return 1
}

# ── Run Aider polyglot benchmark against running turboquant ──────────────────
run_bench() {
  local quant=$1
  local ctx=$2
  local run_name
  run_name="$(date +%Y%m%d-%H%M)-${quant}"
  local out="${RESULTS_DIR}/${quant}.txt"

  log "Bench start: $quant (ctx=$ctx, langs=${BENCH_LANGUAGES})"
  {
    echo "=== $quant  ctx=$ctx  $(date) ==="
    (cd "$AI_BOX" && docker compose run --rm --no-deps bench \
      "$run_name" \
      --model "openai/${quant}" \
      --exercises-dir /bench/exercises \
      --languages "$BENCH_LANGUAGES" \
      --edit-format "$BENCH_EDIT_FORMAT" \
      --threads "$BENCH_THREADS" \
      --num-ctx "$ctx" \
      2>&1)
  } | tee "$out"
  log "Bench done: $quant → $out"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

QUANTS=("UD-Q5_K_XL" "UD-Q6_K_XL" "UD-Q8_K_XL")

log "=== Multi-quant Aider benchmark starting ==="
log "Results → $RESULTS_DIR"
log "Languages: $BENCH_LANGUAGES | threads: $BENCH_THREADS | edit-format: $BENCH_EDIT_FORMAT"

# Stop forge (not needed; bench talks directly to turboquant)
(cd "$AI_BOX" && docker compose stop forge) > /dev/null 2>&1 || true

# Kick off first download immediately
download_quant "${QUANTS[0]}" &
DL_PID=$!

for i in "${!QUANTS[@]}"; do
  quant="${QUANTS[$i]}"

  log "Waiting for $quant download..."
  wait $DL_PID

  # Start next download in background while we run this eval
  DL_PID=0
  if [ $((i + 1)) -lt ${#QUANTS[@]} ]; then
    next="${QUANTS[$((i + 1))]}"
    download_quant "$next" &
    DL_PID=$!
  fi

  log "Switching turboquant → $quant"
  (cd "$AI_BOX" && docker compose stop turboquant) > /dev/null 2>&1
  sleep 5

  gguf_first="Qwen3-Coder-Next-${quant}-00001-of-00003.gguf"
  ctx_info=$(probe_context "$gguf_first")
  ctx=$(echo "$ctx_info" | cut -d: -f1)

  run_bench "$quant" "$ctx"
done

[ "$DL_PID" -ne 0 ] && wait $DL_PID 2>/dev/null || true

# ── Restore original model ────────────────────────────────────────────────────
log "Restoring $ORIGINAL_MODEL (ctx=$ORIGINAL_CTX)"
(cd "$AI_BOX" && docker compose stop turboquant) > /dev/null 2>&1
sed -i \
  "s|^MODEL_FILE=.*|MODEL_FILE=${ORIGINAL_MODEL}|; s|^CONTEXT_SIZE=.*|CONTEXT_SIZE=${ORIGINAL_CTX}|" \
  "${AI_BOX}/.env"
(cd "$AI_BOX" && docker compose up -d) > /dev/null 2>&1

# ── Print summary ─────────────────────────────────────────────────────────────
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
