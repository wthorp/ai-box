# runner/Dockerfile
# Orchestration image: docker compose, harbor (Terminal-Bench), wget, nvidia-smi.
# Mount /var/run/docker.sock to control turboquant/rotorquant/bench on the host.
#
# Usage (from repo root):
#   docker compose run --rm runner scripts/smoke_bench.sh
#   ./smoke_bench.sh   # thin wrapper → scripts/ → runner container

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV AI_BOX=/workspace
ENV AI_BOX_RUNNER=1
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl wget ca-certificates \
        python3 python3-venv \
        jq gawk grep sed coreutils \
        docker.io docker-compose-v2 \
    && rm -rf /var/lib/apt/lists/*

# Terminal-Bench / DeepSWE harnesses (pinned Harbor, current Pier)
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "harbor==0.8.0" "datacurve-pier"

WORKDIR /workspace

ENTRYPOINT ["bash"]
