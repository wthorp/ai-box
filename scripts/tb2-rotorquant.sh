#!/bin/bash
# Run TB2.0 for Q4 rotorquant comparison
# Run AFTER stopping turboquant and starting rotorquant
# rotorquant serves on port 8082

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/tb2-rotorquant.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

echo "Stopping turboquant, starting rotorquant..."
docker compose stop turboquant 2>/dev/null || true
"${AI_BOX}/scripts/start.sh" rotorquant

echo "Waiting for rotorquant to be ready..."
for _ in {1..60}; do
  if curl -sf http://localhost:8082/v1/models > /dev/null 2>&1; then
    echo "rotorquant is ready!"
    break
  fi
  sleep 5
done

export OPENAI_BASE_URL=http://localhost:8082/v1
export OPENAI_API_KEY=notneeded

harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --n-tasks 10 \
  --n-concurrent 1 \
  --verifier-timeout-multiplier 3 \
  --extra-docker-compose "${AI_BOX}/uv-python-cache-overlay.yaml" \
  --yes 2>&1 | tee "${AI_BOX}/tb2-rotorquant.log"

echo "rotorquant TB2.0 run complete!"

echo "Restoring turboquant..."
docker compose stop rotorquant 2>/dev/null || true
"${AI_BOX}/scripts/start.sh" turboquant
