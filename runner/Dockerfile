# runner/Dockerfile
# Orchestration image: docker compose, wget, nvidia-smi, DeepSWE/Pier.
# Mount /var/run/docker.sock to control rotorquant on the host.
#
# Usage (from repo root):
#   docker compose run --rm runner scripts/deepswe-run.sh
#   ./deepswe-run.sh   # thin wrapper → scripts/ → runner container

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV AI_BOX=/workspace
ENV AI_BOX_RUNNER=1
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl wget ca-certificates \
        python3 python3-venv \
        jq gawk grep sed coreutils git shellcheck \
        docker.io docker-compose-v2 \
    && rm -rf /var/lib/apt/lists/*

# DeepSWE harness
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "datacurve-pier" "sqlglot"

WORKDIR /workspace

ENTRYPOINT ["bash"]
