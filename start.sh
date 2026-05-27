#!/bin/bash
# Start turboquant (or rotorquant) with VRAM-optimal params.
#
# Usage: ./start.sh [service] [min_ctx]
#   service  — turboquant (default) or rotorquant
#   min_ctx  — minimum acceptable context window (default: 32768)
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

set -euo pipefail
cd "$(dirname "$0")"

SERVICE=${1:-turboquant}
MIN_CTX=${2:-65536}

# ── Model constants (Qwen3-Coder-Next, 48 layers, 512 experts) ─────────────
readonly N_LAYERS=48
readonly MOE_MIB_PER_LAYER=928
readonly NON_MOE_MIB=800
readonly KV_PER_TOKEN_MIB=0.04928
readonly SAFETY_MIB=1024
readonly MAX_CTX=262144
readonly SYS_OVERHEAD_MIB=300  # baseline CUDA + display (measured at fresh start)

# ── Usable VRAM ─────────────────────────────────────────────────────────────
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
USABLE=$(( TOTAL_VRAM - SYS_OVERHEAD_MIB ))
echo "GPU: ${TOTAL_VRAM} MiB total, ${USABLE} MiB usable"

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
