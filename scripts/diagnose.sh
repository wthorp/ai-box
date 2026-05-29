#!/bin/bash
# diagnose.sh — MoE checklist health snapshot (run inside runner container).
#
# Usage: ./diagnose.sh

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/diagnose.sh "$@"
fi

set -euo pipefail

section() { echo ""; echo "=== $* ==="; }

section "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi unavailable"

section "VRAM detail"
nvidia-smi 2>/dev/null | head -20 || true

section "Memory / swap"
free -h 2>/dev/null || true
swapon --show 2>/dev/null || echo "(no swap active)"

section "NUMA"
if command -v numactl &>/dev/null; then
  numactl --hardware 2>/dev/null | head -20
else
  echo "numactl not installed (optional for multi-socket)"
fi

section "Sysctl (inference)"
sysctl vm.swappiness vm.vfs_cache_pressure vm.dirty_ratio vm.dirty_background_ratio 2>/dev/null || true

section "Docker inference services"
docker compose ps turboquant rotorquant 2>/dev/null || true

section "turboquant health"
curl -sf http://localhost:8080/health 2>/dev/null && echo || echo "turboquant not healthy"

section "llama-server mlock (host PID)"
pid=$(pgrep -f 'llama-server.*--port 8080' 2>/dev/null | head -1 || true)
if [[ -n "$pid" ]]; then
  echo "PID $pid"
  grep -E '^(Name|VmRSS|Mlocked):' "/proc/${pid}/status" "/proc/${pid}/smaps_rollup" 2>/dev/null \
    | head -10 || true
else
  echo "no turboquant llama-server process on host"
fi

section "Paging (5s vmstat — si/so should stay ~0)"
vmstat 1 5 2>/dev/null || echo "vmstat not available"

echo ""
echo "Monitor live: nvtop | watch -n1 nvidia-smi | docker compose logs -f turboquant"
