FROM node:24-bookworm-slim

ARG SVERKLO_VERSION=latest

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g "sverklo@${SVERKLO_VERSION}"

COPY scripts/sverklo_mcp_proxy.py /usr/local/bin/sverklo-mcp-proxy
RUN chmod +x /usr/local/bin/sverklo-mcp-proxy

WORKDIR /workspace

ENV SVERKLO_HTTP_HOST=0.0.0.0
ENV SVERKLO_HTTP_PORT=3007
ENV SVERKLO_PROJECT_PATH=/deep-swe/tasks

CMD ["python3", "/usr/local/bin/sverklo-mcp-proxy"]
