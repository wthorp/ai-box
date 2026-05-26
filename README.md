# ai-box

Reproducible AI inference server infrastructure for Ubuntu 26.04.
One bootstrap script sets up the host; Docker Compose runs the services.

## Services

| Service | Description |
|---------|-------------|
| `turboquant` | llama.cpp fork with turbo KV-cache quantisation (`turbo2/3/4`) and `--n-cpu-moe` for MoE CPU offload |
| `rotorquant` | llama.cpp fork with planar/iso KV-cache compression (28% faster decode, better perplexity than turbo) — opt-in via `--profile rotorquant` |
| `bench` | [Aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark) runner — measures code-editing accuracy across Python, Go, Rust, JavaScript (225 exercises from Exercism) |

## Hardware

Reference machine used for development and benchmarking:

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 9 7950X (16c/32t, up to 5.88 GHz) |
| **RAM** | 92 GB |
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM, compute 8.6) |
| **Storage** | Samsung SSD 990 PRO 4 TB NVMe |
| **OS** | Ubuntu 26.04 LTS, kernel 7.0.0-15, NVIDIA driver 595.71.05 |

The `--n-cpu-moe 35` default keeps 35 MoE expert layers on the Ryzen 9's 32 threads while the attention layers run on the RTX 3090. Adjust `CPU_MOE_LAYERS` to taste for different CPU/GPU combinations.

## Prerequisites

- Ubuntu 26.04 LTS (fresh install)
- NVIDIA GPU (Turing / Ampere / Ada / Hopper)
- NVIDIA driver 595 already installed, **or** let `setup.sh` install `nvidia-utils-595-server`
- Internet access during setup

## 1 — Host bootstrap

Run once on the server as root or with sudo:

```bash
sudo bash setup.sh
```

This installs and pins Docker, the NVIDIA Container Toolkit, CUDA toolkit 13.1,
configures the NVIDIA Docker runtime, generates CDI specs, and applies OS-level
performance tuning (`vm.swappiness`, dirty-ratio). Log out and back in (or run
`newgrp docker`) so your user can run Docker without sudo.

## 2 — Configure

```bash
cp .env.example .env
$EDITOR .env          # set MODELS_DIR, MODEL_FILE, etc.
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/data/ai/models` | Host directory containing GGUF files |
| `MODEL_FILE` | *(required)* | Filename of the model to load |
| `CPU_MOE_LAYERS` | `35` | MoE expert layers to keep on CPU |
| `CONTEXT_SIZE` | `8192` | Context window in tokens |

## 3 — Start

```bash
# First run builds the image from source (~10 min with GPU available)
docker compose up -d

# Check health
curl http://localhost:8080/health

# Logs
docker compose logs -f turboquant
```

## 4 — Aider polyglot benchmark

`bench` runs the [Aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark) — 225 Exercism exercises across Python, Go, Rust, and JavaScript. Each exercise gives the model failing tests and asks it to edit the code until they pass.

> **Note:** TypeScript is not in the polyglot benchmark; JavaScript covers that slot.

### Running the benchmark

```bash
# Quick check — Python only, first 10 exercises
docker compose run --rm bench smoke-test \
  --model openai/local \
  --exercises-dir /bench/exercises \
  --languages python \
  --threads 1

# Full 4-language run against the loaded model
docker compose run --rm bench q4-baseline \
  --model openai/local \
  --exercises-dir /bench/exercises \
  --languages python,go,rust,javascript \
  --edit-format whole \
  --threads 1

# Results land in EVAL_RESULTS_DIR/<run-name>/
```

The `--model openai/local` value is arbitrary — turboquant serves whatever model is loaded regardless of the name in the request. Use a descriptive name (e.g. `openai/q5-xl`) to keep results organised.

### Context window

Aider may default to a small context for unknown models. Pass `--openai-api-base` or set `OPENAI_API_BASE` (already in the compose environment) and optionally add `--set-context-window 32768` if you want to override the default.

### Smoke test

`smoke_bench.sh` runs 20 exercises across Python+Go at max context — completes in ~20 min. Use it to quickly validate a model or compare two servers:

```bash
# Quick check against turboquant (default)
./smoke_bench.sh

# Compare rotorquant
./smoke_bench.sh --server rotorquant

# Custom label
./smoke_bench.sh --model q5-xl --port 8080
```

### Multi-quant comparison

See `multi_quant_eval.sh` in the repo root for an automated pipeline that downloads Q5/Q6/Q8 UD variants, probes the max context per quant, and runs the benchmark on each.

## 5 — rotorquant

rotorquant ([scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)) is a llama.cpp fork built on turboquant that replaces WHT-based KV cache compression with block-diagonal rotations:

- `planar3/4` — 2D Givens rotations
- `iso3/4` — 4D quaternion rotations

Benchmarked at 28% faster decode and better perplexity than turbo equivalents. Runs on port 8082 and is opt-in via Docker Compose profiles.

```bash
# Build and start rotorquant alongside turboquant
docker compose --profile rotorquant up -d rotorquant

# Smoke test against rotorquant
./smoke_bench.sh --server rotorquant
```

---

## 6 — Adding future services

Add a new directory (e.g. `ollama/`, `vllm/`) with its own `Dockerfile`, then
add a service block to `docker-compose.yml`. Commit when stable.

## Performance notes

The compose file is tuned for throughput over security:

- `network_mode: host` — eliminates Docker NAT overhead
- `ipc: host` — enables CUDA IPC shared memory
- `cap_add: SYS_NICE` — lets CUDA set thread scheduling priority
- `security_opt: seccomp:unconfined` — removes syscall filter
- `ulimits.memlock: -1` — required for `--mlock` to pin weights in RAM
- `MALLOC_ARENA_MAX=2` — reduces glibc arena fragmentation

Two optimisations were tested and **deliberately excluded** because they
degraded token-generation speed ~10% on CUDA + CPU-MoE workloads:
`jemalloc LD_PRELOAD` (conflicts with CUDA's allocator) and
`OMP_WAIT_POLICY=passive` (adds OpenMP thread wake-up latency per token).
