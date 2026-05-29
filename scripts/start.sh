#!/bin/bash
# Start turboquant (or rotorquant) with VRAM-optimal params.
#
# Usage: ./start.sh [service] [min_ctx]
#   service  — turboquant (default) or rotorquant
#   min_ctx  — minimum acceptable context window (default: 65536; fills ~24GB VRAM)
#
# Algorithm: walk from all-GPU toward all-CPU; stop at the first
# CPU_MOE_LAYERS value that still leaves enough VRAM for min_ctx tokens
# of KV cache. Remaining VRAM beyond that sets CONTEXT_SIZE.
#
# Constants below are empirical for Qwen3-Coder-Next-UD on RTX 3090:
#   MOE_MIB_PER_LAYER: measured from nvidia-smi (16092 MiB used at CPU_MOE=35,
#                       CONTEXT=65536) — (16092 - 3228 KV - 800 non-MoE) / 13
#   NON_MOE_MIB:       residual after MoE + KV — attention, embed, norms, CUDA
#   KV_PER_TOKEN_MIB:  3228 MiB / 65536 tokens (turbo4 K + turbo3 V, 48 layers)
#   If you change --cache-type-k/v, re-measure and update KV_PER_TOKEN_MIB.

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/start.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"
source "${AI_BOX}/scripts/lib/env.sh"

SERVICE=${1:-turboquant}
MIN_CTX=${2:-65536}
MODEL_FILE=$(env_value MODEL_FILE .env)
MODELS_DIR=$(env_value MODELS_DIR .env)
MEM_LIMIT=${INFERENCE_MEMORY_LIMIT:-$(env_value INFERENCE_MEMORY_LIMIT .env)}
MEM_LIMIT=${MEM_LIMIT:-88g}

# ── Model constants (Qwen3-Coder-Next, 48 layers, 512 experts) ─────────────
readonly N_LAYERS=48
readonly MOE_MIB_PER_LAYER=928
readonly NON_MOE_MIB=800
readonly KV_PER_TOKEN_MIB=0.04928
readonly SAFETY_MIB=1024
readonly MAX_CTX=262144
readonly SYS_OVERHEAD_MIB=300  # baseline CUDA + display (measured at fresh start)

# ── Usable VRAM ─────────────────────────────────────────────────────────────
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ ! "$TOTAL_VRAM" =~ ^[0-9]+$ ]]; then
  echo "ERROR: nvidia-smi did not return GPU memory; fix NVIDIA/runner visibility before tuning" >&2
  exit 1
fi
USABLE=$(( TOTAL_VRAM - SYS_OVERHEAD_MIB ))
echo "GPU: ${TOTAL_VRAM} MiB total, ${USABLE} MiB usable"

model_path="${MODELS_DIR:-/data/ai/models}/${MODEL_FILE}"
if [[ ! -f "$model_path" && -f "/models/${MODEL_FILE}" ]]; then
  model_path="/models/${MODEL_FILE}"
fi
if [[ -f "$model_path" ]]; then
  if [[ "$MODEL_FILE" =~ ^(.+)-00001-of-([0-9]+)\.gguf$ ]]; then
    model_dir=$(dirname "$model_path")
    model_glob="${model_dir}/${BASH_REMATCH[1]}-"*"of-${BASH_REMATCH[2]}.gguf"
    model_mib=$(du -cm $model_glob 2>/dev/null | awk '/total$/ {print $1}')
    model_mib=${model_mib:-$(du -m "$model_path" | awk '{print $1}')}
  else
    model_mib=$(du -m "$model_path" | awk '{print $1}')
  fi
  mem_limit_mib=$(python3 - "$MEM_LIMIT" <<'PY'
import re, sys
s = sys.argv[1].strip().lower()
m = re.fullmatch(r"([0-9]+)([kmgt]?)b?", s)
if not m:
    print(0)
    raise SystemExit
n = int(m.group(1))
unit = m.group(2)
scale = {"": 1 / 1024 / 1024, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}[unit]
print(int(n * scale))
PY
)
  safety_mib=8192
  echo "Host RAM budget: model file ${model_mib} MiB, container cap ${mem_limit_mib} MiB"
  if [[ "$mem_limit_mib" -gt 0 && $(( model_mib + safety_mib )) -gt "$mem_limit_mib" ]]; then
    echo "ERROR: ${MODEL_FILE} plus ${safety_mib} MiB safety exceeds INFERENCE_MEMORY_LIMIT=${MEM_LIMIT}" >&2
    echo "Use a smaller quant, raise the cap, or disable --no-mmap/--mlock for this experiment." >&2
    exit 1
  fi
else
  echo "WARN: model file not found for host RAM budget check: $model_path" >&2
fi

# ── Find optimal CPU_MOE_LAYERS ─────────────────────────────────────────────
CPU_MOE=$N_LAYERS
CONTEXT=0

echo "  moe  gpu  model(MiB)  avail(MiB)  ctx"
for moe in $(seq 0 $N_LAYERS); do
    gpu=$(( N_LAYERS - moe ))
    model_mib=$(( gpu * MOE_MIB_PER_LAYER + NON_MOE_MIB ))
    avail_mib=$(( USABLE - model_mib - SAFETY_MIB ))
    [ "$avail_mib" -le 0 ] && continue

    ctx=$(python3 -c "
import math
avail = $avail_mib
raw = int(avail / $KV_PER_TOKEN_MIB)
if raw < 1: print(0)
else: print(min(2**int(math.log2(raw)), $MAX_CTX))
")
    echo "  ${moe}   ${gpu}   ${model_mib}       ${avail_mib}          ${ctx}"
    if [ "$ctx" -ge "$MIN_CTX" ]; then
        CPU_MOE=$moe
        CONTEXT=$ctx
        break
    fi
done
echo ""

if [ "$CONTEXT" -lt "$MIN_CTX" ]; then
    echo "ERROR: cannot fit MIN_CTX=$MIN_CTX even with all MoE layers on CPU" >&2
    exit 1
fi

echo "CPU_MOE_LAYERS=${CPU_MOE}  ($(( N_LAYERS - CPU_MOE ))/${N_LAYERS} MoE layers on GPU)"
echo "CONTEXT_SIZE=${CONTEXT}"

# ── Start ────────────────────────────────────────────────────────────────────
CPU_MOE_LAYERS=$CPU_MOE CONTEXT_SIZE=$CONTEXT \
    docker compose up -d "$SERVICE"
