# turboquant/Dockerfile
# Multi-stage build: compile on Ubuntu 24.04 (glibc 2.39) so the binary
# is portable to any newer Ubuntu, then copy into a slim CUDA runtime image.
#
# Source: https://github.com/TheTom/llama-cpp-turboquant
# Branch: feature/turboquant-kv-cache
# Adds turbo2/turbo3/turbo4 KV cache quantisation and --n-cpu-moe flag.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libssl-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth=1 \
    -b feature/turboquant-kv-cache \
    https://github.com/TheTom/llama-cpp-turboquant.git .

# Build for common CUDA compute capabilities:
#   75 = Turing  (RTX 20xx)
#   80 = Ampere  (A100, RTX 30xx base)
#   86 = Ampere  (RTX 3090 / 3080)
#   89 = Ada     (RTX 40xx)
#   90 = Hopper  (H100)
#
# BUILD_SHARED_LIBS=OFF: produces a self-contained static binary — no need
# to copy libggml-*.so files into the runtime image.
# GGML_CUDA_NO_VMM=ON: disables CUDA virtual memory management, eliminating
# the libcuda.so driver dependency at build time (no GPU in Docker builder).
RUN cmake -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA_NO_VMM=ON \
    && cmake --build build --target llama-server -j$(nproc)

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libzstd1 \
        libssl3 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/bin/ /usr/local/lib/turboquant/
RUN ln -s /usr/local/lib/turboquant/llama-server /usr/local/bin/llama-server \
    && echo "/usr/local/lib/turboquant" > /etc/ld.so.conf.d/turboquant.conf \
    && ldconfig

# Reduce glibc malloc arena count — limits fragmentation in long-running servers.
# DO NOT set LD_PRELOAD=jemalloc or OMP_WAIT_POLICY=passive here:
# both were measured to degrade token-generation speed ~10% on CUDA+CPU-MoE
# workloads due to CUDA allocator conflicts and OpenMP thread wake-up latency.
ENV MALLOC_ARENA_MAX=2

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
