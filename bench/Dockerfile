# bench/Dockerfile
# Aider polyglot benchmark runner.
# Installs Python, Go, Rust, Node.js runtimes + aider (from source with dev
# deps), then clones the polyglot-benchmark exercises at build time.
#
# Run via:
#   docker compose run --rm bench <run-name> [--languages python,go,rust,javascript]

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/go/bin:/root/.cargo/bin:${PATH}"
# benchmark/ is not an installed package — add it to PYTHONPATH so that
# `import prompts`, `import plots` etc. resolve from within the dir.
ENV PYTHONPATH="/bench/aider/benchmark"

# ── System packages, Python, Node.js ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        nodejs npm \
        git curl ca-certificates \
        build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Go ────────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://go.dev/dl/go1.23.4.linux-amd64.tar.gz \
    | tar -C /usr/local -xz

# ── Rust ─────────────────────────────────────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --no-modify-path

# ── Benchmark repos ───────────────────────────────────────────────────────────
WORKDIR /bench
RUN git clone --depth=1 https://github.com/Aider-AI/polyglot-benchmark exercises \
    && git clone --depth=1 https://github.com/Aider-AI/aider aider \
    && python3 -m venv /bench/venv

ENV PATH="/bench/venv/bin:${PATH}"

# ── Aider + benchmark deps (install from source so versions align) ────────────
WORKDIR /bench/aider
RUN pip install --no-cache-dir \
        -e . \
        -r requirements/requirements-dev.txt

# ── Git identity (aider commits inside exercise temp dirs) ────────────────────
RUN git config --global user.email "bench@localhost" \
    && git config --global user.name "bench"

# Results land in ./tmp/<run-name>/ — mount eval-results here
VOLUME ["/bench/aider/tmp"]

ENV OPENAI_API_KEY=notneeded
# Required: benchmark.py checks for this var and exits early if absent (safety guard)
ENV AIDER_DOCKER=1
# Store results under ./tmp/ so the volume mount on /bench/aider/tmp lands them on the host
ENV AIDER_BENCHMARK_DIR=tmp

# Pass run-name + flags after 'docker compose run bench'
ENTRYPOINT ["python3", "benchmark/benchmark.py"]
CMD ["--help"]
