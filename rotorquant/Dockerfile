# rotorquant/Dockerfile
# Multi-stage build: compile rotorquant (llama.cpp fork with planar/iso KV
# cache compression) then copy into a slim CUDA runtime image.
#
# Source: https://github.com/johndpope/llama-cpp-turboquant
# Branch: feature/planarquant-kv-cache
# Adds planar3/planar4 (2D Givens) and iso3/iso4 (quaternion) KV cache
# compression — faster decode and better perplexity than turbo3/turbo4.
# Inherits --n-cpu-moe for MoE expert CPU offload from turboquant base.

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
    -b feature/planarquant-kv-cache \
    https://github.com/johndpope/llama-cpp-turboquant.git .

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

COPY --from=builder /build/build/bin/ /usr/local/lib/rotorquant/
RUN ln -s /usr/local/lib/rotorquant/llama-server /usr/local/bin/llama-server \
    && echo "/usr/local/lib/rotorquant" > /etc/ld.so.conf.d/rotorquant.conf \
    && ldconfig

ENV MALLOC_ARENA_MAX=2

EXPOSE 8082
ENTRYPOINT ["/usr/local/bin/llama-server"]
