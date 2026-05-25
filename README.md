# ai-box

Reproducible AI inference server infrastructure for Ubuntu 26.04.
One bootstrap script sets up the host; Docker Compose runs the services.

## Services

| Service | Description |
|---------|-------------|
| `turboquant` | llama.cpp fork with turbo KV-cache quantisation (`turbo2/3/4`) and `--n-cpu-moe` for MoE CPU offload |

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

## 4 — Adding future services

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
