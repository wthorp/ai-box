#!/bin/bash
# Re-run TB2.0 tasks that failed due to EnvironmentStartTimeout or VerifierTimeout
# Optimizations:
#   - uv-python-cache-overlay.yaml: mounts Python 3.13 cache (avoids 711s download)
#   - verifier-timeout-multiplier 3: gives first Debian task enough time for cold apt-get update
# Note: pytorch-model-cli excluded - requires ffmpeg (~200MB) + torch (~300MB) via apt/pip,
#       infeasible at ~45 kB/s container network speed regardless of timeout

if [[ -z "${AI_BOX_RUNNER:-}" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT"
  exec docker compose run --rm -e AI_BOX_RUNNER=1 runner scripts/tb2-rerun.sh "$@"
fi

set -euo pipefail
AI_BOX="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AI_BOX"

export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=notneeded

harbor run \
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
  --extra-docker-compose "${AI_BOX}/uv-python-cache-overlay.yaml" \
  --yes
