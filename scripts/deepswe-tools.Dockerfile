FROM golang:1.26-bookworm AS go-tools

RUN go install golang.org/x/tools/gopls@latest && \
    go install golang.org/x/tools/cmd/goimports@latest && \
    go install honnef.co/go/tools/cmd/staticcheck@latest

FROM node:24-bookworm-slim

ARG SVERKLO_VERSION=latest

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_TOOL_DIR=/opt/qsa-tools/uv-tools
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_PYTHON_INSTALL_DIR=/opt/qsa-tools/uv-python

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      bash ca-certificates curl git python3 python3-pip shellcheck shfmt xz-utils && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g "sverklo@${SVERKLO_VERSION}" pyright typescript @biomejs/biome @ast-grep/cli && \
    python3 -m pip install --break-system-packages PyYAML sqlglot && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    uv tool install -p 3.13 ruff && \
    uv tool install -p 3.13 mypy && \
    uv tool install -p 3.13 serena-agent && \
    sverklo --help >/dev/null && \
    serena --help >/dev/null

COPY --from=go-tools /go/bin/gopls /usr/local/bin/gopls
COPY --from=go-tools /go/bin/goimports /usr/local/bin/goimports
COPY --from=go-tools /go/bin/staticcheck /usr/local/bin/staticcheck
