#!/bin/bash
# Re-run TB2.0 tasks that failed due to EnvironmentStartTimeout or VerifierTimeout
# Optimizations:
#   - uv-python-cache-overlay.yaml: mounts Python 3.13 cache (avoids 711s download)
#   - apt-proxy.conf: routes apt through local apt-cacher-ng at host:3142 (caches Debian pkg lists)
#   - verifier-timeout-multiplier 3: gives first Debian task enough time for cold apt-get update
# Note: pytorch-model-cli excluded - requires ffmpeg (~200MB) + torch (~300MB) via apt/pip,
#       infeasible at ~45 kB/s container network speed regardless of timeout

set -euo pipefail

export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=notneeded

cd /data/ai/ai-box

/home/bill/.local/bin/harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --include-task-name gpt2-codegolf \
  --include-task-name llm-inference-batching-scheduler \
  --include-task-name break-filter-js-from-html \
  --include-task-name reshard-c4-data \
  --include-task-name write-compressor \
  --include-task-name merge-diff-arc-agi-task \
  --include-task-name log-summary-date-ranges \
  --n-concurrent 1 \
  --verifier-timeout-multiplier 3 \
  --extra-docker-compose uv-python-cache-overlay.yaml \
  --yes
