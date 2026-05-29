#!/bin/bash
# Exhaustive-ish quant/context/MoE sweep.
#
# Defaults are deliberately explicit:
#   quants:   Q4,Q5,Q6
#   contexts: 32768,65536,131072,262144
#   moe:      even CPU_MOE_LAYERS with at least 4 MoE layers left on GPU
#
# Logs free RAM, container RAM, VRAM, load status, and completion throughput for
# every attempted point. This script is for generating data, not for preserving
# old assumptions.

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/grid_sweep.sh "$@"
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
QUANT_LIST="Q4,Q5,Q6"
CTX_LIST="32768,65536,131072,262144"
MIN_GPU_MOE=4
MAX_TOKENS=64
LOAD_TIMEOUT=720
PROMPT="Explain quicksort in one concise paragraph."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quants) QUANT_LIST="$2"; shift 2 ;;
    --contexts) CTX_LIST="$2"; shift 2 ;;
    --min-gpu-moe) MIN_GPU_MOE="$2"; shift 2 ;;
    --tokens) MAX_TOKENS="$2"; shift 2 ;;
    --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "$SERVICE" == "rotorquant" && "$PORT" == "8080" ]] && PORT=8082

RESULTS="${EVAL_RESULTS_DIR:-/eval-results}/grid-sweep-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
OUT="${RESULTS}/results.tsv"
SUMMARY="${RESULTS}/best.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/run.log" >&2; }

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

to_mib() {
  python3 - "$1" <<'PY'
import re, sys
s = sys.argv[1].strip()
m = re.match(r"([0-9.]+)\s*([KMGT]?i?B)", s, re.I)
if not m:
    print("")
    raise SystemExit
n = float(m.group(1))
u = m.group(2).lower()
scale = {"b": 1/1024/1024, "kb": 1/1024, "kib": 1/1024, "mb": 1, "mib": 1,
         "gb": 1024, "gib": 1024, "tb": 1024*1024, "tib": 1024*1024}[u]
print(int(n * scale))
PY
}

container_mem_mib() {
  local usage current
  usage=$(docker stats --no-stream --format '{{.MemUsage}}' "ai-box-${SERVICE}-1" 2>/dev/null || true)
  current=${usage%% / *}
  [[ -n "$current" ]] || { echo ""; return 0; }
  to_mib "$current"
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

run_completion() {
  local body t0 t1
  t0=$(date +%s.%N)
  body=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
      '{model:"local",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,stream:false}')" \
    2>/dev/null) || return 1
  t1=$(date +%s.%N)
  python3 - "$body" "$t0" "$t1" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
t0, t1 = map(float, sys.argv[2:4])
usage = body.get("usage") or {}
timings = body.get("timings") or {}
comp = usage.get("completion_tokens") or 0
elapsed = max(t1 - t0, 1e-9)
print("\t".join([
    str(comp),
    f"{elapsed:.3f}",
    f"{comp / elapsed:.2f}",
    str(timings.get("prompt_per_second", "")),
    str(timings.get("predicted_per_second", "")),
]))
PY
}

record_row() {
  printf '%s\n' "$*" >> "$OUT"
}

ENV_SNAPSHOT=$(save_env_file .env)
cleanup() {
  restore_env_file "$ENV_SNAPSHOT" .env
}
trap cleanup EXIT

{
  echo -e "quant\tmodel\tctx\tcpu_moe\tgpu_moe\tstatus\tfree_ram_before_mib\tfree_ram_after_load_mib\tfree_ram_after_completion_mib\tcontainer_mem_mib\tvram_used_mib\tvram_total_mib\tvram_pct\tcompletion_tokens\tseconds\twall_tok_per_s\tprompt_tok_per_s\tdecode_tok_per_s\trestarts\toom\tnote"
} > "$OUT"

IFS=',' read -ra QUANTS <<< "$QUANT_LIST"
IFS=',' read -ra CONTEXTS <<< "$CTX_LIST"

max_cpu_moe=$((48 - MIN_GPU_MOE))
if (( max_cpu_moe < 0 || max_cpu_moe > 48 )); then
  echo "Invalid --min-gpu-moe: $MIN_GPU_MOE" >&2
  exit 1
fi
if (( max_cpu_moe % 2 == 1 )); then
  max_cpu_moe=$((max_cpu_moe - 1))
fi

MOES=()
for ((moe=0; moe<=max_cpu_moe; moe+=2)); do
  MOES+=("$moe")
done

log "=== grid sweep quants=${QUANTS[*]} contexts=${CONTEXTS[*]} cpu_moe=${MOES[*]} min_gpu_moe=$MIN_GPU_MOE ==="
log "Results: $OUT"

for quant in "${QUANTS[@]}"; do
  if ! model_exists "$quant"; then
    log "SKIP $quant: model missing"
    continue
  fi
  model=$(model_gguf "$quant")

  for ctx in "${CONTEXTS[@]}"; do
    for moe in "${MOES[@]}"; do
      gpu_moe=$((48 - moe))
      log "TRY quant=$quant ctx=$ctx cpu_moe=$moe gpu_moe=$gpu_moe"

      free_before=$(free_available_mib)
      set_env_value MODEL_FILE "$model" .env
      set_env_value CONTEXT_SIZE "$ctx" .env
      set_env_value CPU_MOE_LAYERS "$moe" .env

      compose stop "$SERVICE" > /dev/null 2>&1 || true
      compose rm -sf "$SERVICE" > /dev/null 2>&1 || true
      sleep 3
      compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1 || true
      docker update --restart=no "ai-box-${SERVICE}-1" > /dev/null 2>&1 || true

      status=ok
      note=""
      if ! wait_healthy_or_exit "$LOAD_TIMEOUT"; then
        status=load_failed
        note="health timeout, exit, or oom before ready"
      fi

      free_after_load=$(free_available_mib)
      mem=$(gpu_memory_csv 2>/dev/null || echo ",")
      vram_used=${mem%,*}
      vram_total=${mem#*,}
      vram_pct=""
      if [[ "$vram_used" =~ ^[0-9]+$ && "$vram_total" =~ ^[0-9]+$ && "$vram_total" -gt 0 ]]; then
        vram_pct=$(python3 -c "print(round(100*${vram_used}/${vram_total}, 1))")
      fi
      cmem=$(container_mem_mib)
      restarts=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.RestartCount}}' 2>/dev/null || echo "")
      oom=$(docker inspect "ai-box-${SERVICE}-1" --format '{{.State.OOMKilled}}' 2>/dev/null || echo "")

      comp="" seconds="" wall_tps="" prompt_tps="" decode_tps=""
      if [[ "$status" == "ok" ]]; then
        if stats=$(run_completion); then
          IFS=$'\t' read -r comp seconds wall_tps prompt_tps decode_tps <<< "$stats"
        else
          status=completion_failed
          note="load ok, completion failed"
        fi
      fi
      free_after_completion=$(free_available_mib)

      record_row "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$quant" "$model" "$ctx" "$moe" "$gpu_moe" "$status" \
        "$free_before" "$free_after_load" "$free_after_completion" "$cmem" \
        "$vram_used" "$vram_total" "$vram_pct" \
        "$comp" "$seconds" "$wall_tps" "$prompt_tps" "$decode_tps" \
        "$restarts" "$oom" "$note")"

      log "  $status mem=${cmem}MiB vram=${vram_used}/${vram_total} (${vram_pct}%) wall_tps=${wall_tps:-} decode_tps=${decode_tps:-} note=${note:-}"
    done
  done
done

python3 - "$OUT" "$SUMMARY" <<'PY'
import csv, sys
src, dst = sys.argv[1:3]
rows = []
with open(src, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["status"] != "ok":
            continue
        try:
            row["_tps"] = float(row["wall_tok_per_s"] or 0)
        except ValueError:
            row["_tps"] = 0.0
        rows.append(row)

best = {}
for row in rows:
    key = (row["quant"], row["ctx"])
    if key not in best or row["_tps"] > best[key]["_tps"]:
        best[key] = row

fields = ["quant", "ctx", "cpu_moe", "gpu_moe", "wall_tok_per_s", "decode_tok_per_s",
          "container_mem_mib", "vram_used_mib", "vram_total_mib", "vram_pct"]
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    for key in sorted(best):
        w.writerow({k: best[key].get(k, "") for k in fields})
PY

log "Best-by-quant-context: $SUMMARY"
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

log "Restoring original .env"
restore_env_file "$ENV_SNAPSHOT" .env
trap - EXIT
compose up -d --force-recreate "$SERVICE" > /dev/null 2>&1
