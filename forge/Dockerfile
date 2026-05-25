# forge/Dockerfile
# Multi-stage Alpine build: clone forge from GitHub, install deps with uv,
# then copy into a slim runtime image. CMD (not ENTRYPOINT) so
# 'docker compose run' can override for running eval scenarios.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.13-alpine3.23 AS builder

RUN pip install --no-cache-dir --root-user-action=ignore uv

ENV UV_LINK_MODE=copy

RUN apk add --no-cache git
WORKDIR /app
RUN git clone --depth=1 https://github.com/antoinezambelli/forge.git . \
    && uv sync --extra anthropic --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.13-alpine3.23

ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache ca-certificates \
    && addgroup -g 1000 appuser \
    && adduser -D -u 1000 -G appuser appuser

COPY --from=builder --chown=appuser:appuser /app /app
WORKDIR /app
USER appuser
ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:8081/health || exit 1

EXPOSE 8081
CMD ["forge-proxy", "--host", "0.0.0.0", "--port", "8081", \
     "--backend-url", "http://localhost:8080"]
