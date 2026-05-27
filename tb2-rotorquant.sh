#!/bin/bash
# Run TB2.0 for Q4 rotorquant comparison
# Run AFTER stopping turboquant and starting rotorquant
# rotorquant serves on port 8082

set -euo pipefail

cd /data/ai/ai-box

# Ensure turboquant is down, rotorquant is up
echo "Stopping turboquant, starting rotorquant..."
docker compose stop turboquant 2>/dev/null || true
/data/ai/ai-box/start.sh rotorquant

# Wait for rotorquant to be healthy
echo "Waiting for rotorquant to be ready..."
for i in {1..60}; do
  if curl -sf http://localhost:8082/v1/models > /dev/null 2>&1; then
    echo "rotorquant is ready!"
    break
  fi
  sleep 5
done

export OPENAI_BASE_URL=http://localhost:8082/v1
export OPENAI_API_KEY=notneeded

/home/bill/.local/bin/harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --n-tasks 10 \
  --n-concurrent 1 \
  --verifier-timeout-multiplier 3 \
  --extra-docker-compose uv-python-cache-overlay.yaml \
  --yes 2>&1 | tee /data/ai/ai-box/tb2-rotorquant.log

echo "rotorquant TB2.0 run complete!"

# Restore turboquant
echo "Restoring turboquant..."
docker compose stop rotorquant 2>/dev/null || true
/data/ai/ai-box/start.sh turboquant
