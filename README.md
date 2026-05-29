# ai-box

Reproducible AI inference server infrastructure for Ubuntu 26.04.
One bootstrap script sets up the host; Docker Compose runs the services.

## Services

| Service | Description |
|---------|-------------|
| `turboquant` | llama.cpp fork with turbo KV-cache quantisation (`turbo2/3/4`) and `--n-cpu-moe` for MoE CPU offload |
| `rotorquant` | llama.cpp fork with planar/iso KV-cache compression (28% faster decode, better perplexity than turbo) — opt-in via `--profile rotorquant` |

## Hardware

Reference machine used for development and benchmarking:

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 9 7950X (16c/32t, up to 5.88 GHz) |
| **RAM** | 96 GB (2× 48 GB) |
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
| `INFERENCE_MEMORY_LIMIT` | `88g` | Per-server RAM cap; lower this if the host has less RAM than the reference box |
| `INFERENCE_MEMORY_SWAP_LIMIT` | `88g` | Keep equal to `INFERENCE_MEMORY_LIMIT` to prevent inference containers from using swap |
| `EVAL_RESULTS_DIR` | `./eval-results` | Benchmark and smoke-test output directory |
| `DEEPSWE_DIR` | `/data/ai/deep-swe` | Optional host checkout of `datacurve-ai/deep-swe` for Pier runs |
| `UV_PYTHON_CACHE_DIR` | `~/.local/share/uv/python` | Optional host uv cache for TB task containers (faster cold starts) |

## 3 — Start

```bash
# Auto-tune CPU_MOE_LAYERS and CONTEXT_SIZE from GPU VRAM, then start
./start.sh

# Or start with fixed .env values:
docker compose up -d

# Check health
curl http://localhost:8080/health

# Logs
docker compose logs -f turboquant
```

### Orchestration scripts (run in Docker)

All workflow scripts except `setup.sh` re-exec inside the `runner` container
(Docker socket, `nvidia-smi`, `wget`, Pier). Thin wrappers at the repo root
forward to `scripts/`:

`runner` is profile-gated tooling, so plain `docker compose up -d` starts only
the default inference stack. Invoke it explicitly with
`docker compose run --rm ...` or use the wrapper scripts.

| Script | Purpose |
|--------|---------|
| `start.sh` | VRAM-aware `CPU_MOE_LAYERS` / `CONTEXT_SIZE`, then `docker compose up` |
| `deepswe-run.sh` | DeepSWE/Pier run against the loaded local endpoint |
| `light_tune.sh` | Quick `--n-cpu-moe` sweep (tok/s, VRAM, mlock) |
| `tune_quants.sh` | Ideal `CPU_MOE_LAYERS` + `CONTEXT_SIZE` per Q4/Q5/Q6/Q8 |
| `diagnose.sh` | MoE checklist snapshot (GPU, swap, sysctl, paging) |

Rebuild orchestration image after Dockerfile changes: `docker compose build runner`

`setup.sh` is the only script that must run on the host (installs Docker and the
NVIDIA stack). Everything else can also be invoked explicitly:

## 4 — DeepSWE benchmark target

DeepSWE is the preferred benchmark lane for software-engineering quality after
quant/model settings are stable. It uses DeepSWE tasks and runs through Pier.

```bash
# On the host, once:
git clone https://github.com/datacurve-ai/deep-swe /data/ai/deep-swe
docker compose build runner

# Small deterministic subset against the currently loaded model endpoint:
./deepswe-run.sh --n-tasks 5 --sample-seed 0
```

Use `mini-swe-agent` for pure model/quant comparisons because it keeps the agent
layer stable:

```bash
./deepswe-run.sh --agent mini-swe-agent --model openai/local --n-tasks 10 --sample-seed 0
```

Use `codex` later for full agent experiments, including MCP/skills once the
sandbox config is explicit and reproducible:

```bash
./deepswe-run.sh --agent codex --model openai/local --n-tasks 1 --sample-seed 0
```

Keep those as separate result lanes:

- **Quant lane:** `mini-swe-agent`, no MCP/skills, same DeepSWE subset.
- **Agent lane:** `codex`, then codex plus MCP/skills, same subset and model.

## 5 — rotorquant

rotorquant ([scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)) is a llama.cpp fork built on turboquant that replaces WHT-based KV cache compression with block-diagonal rotations:

- `planar3/4` — 2D Givens rotations
- `iso3/4` — 4D quaternion rotations

Benchmarked at 28% faster decode and better perplexity than turbo equivalents. Runs on port 8082 and is opt-in via Docker Compose profiles.

```bash
# Build and start rotorquant alongside turboquant
docker compose --profile rotorquant up -d rotorquant
```

---

## 6 — Adding future services

Add a new directory (e.g. `ollama/`, `vllm/`) with its own `Dockerfile`, then
add a service block to `docker-compose.yml`. Commit when stable.

## MoE tuning (Docker-first)

`setup.sh` applies host-side checklist items: `vm.swappiness=1`, memlock limits,
`nvtop`/`iotop`, CPU performance governor, and optional NUMA visibility.

Inference containers already use the high-impact flags from the MoE checklist:

| Setting | Where |
|---------|--------|
| `--no-mmap` / `--mlock` | `docker-compose.yml` command |
| `--fit off` | `docker-compose.yml` command; prevents silent llama.cpp setting changes during tuning |
| `--n-cpu-moe` | `.env` / `./start.sh` (VRAM-aware) |
| `turbo4`/`turbo3` KV cache | turboquant (planar4/3 on rotorquant) |
| `IPC_LOCK` + `memlock: -1` | compose `cap_add` / `ulimits` |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | compose environment |

Quick local tuning without a full benchmark (tests at **max context that loads**,
not 8192 — small contexts are only probed for Q8 quants):

```bash
./diagnose.sh
./light_tune.sh --moe 34,35,36,37   # per-moe max ctx, ~24 GB VRAM target
./tune_quants.sh --quants Q4,Q5,Q6,Q8   # ~1h: full per-quant MoE + max ctx
./start.sh                          # production: max ctx + CPU_MOE_LAYERS from VRAM model
```

Per-quant presets from tuning live in `eval-results/tune-quants-*/recommendations.env`.
Ongoing tuning notes and validated Q4 numbers: **[MEMORY.md](MEMORY.md)**.

Swap is left enabled by default in `setup.sh` (safer for desktops). On a dedicated
inference box you can run `sudo swapoff -a` manually.

## Performance notes

The compose file is tuned for throughput over security:

- `network_mode: host` — eliminates Docker NAT overhead
- `ipc: host` — enables CUDA IPC shared memory
- `cap_add: SYS_NICE` — lets CUDA set thread scheduling priority
- `security_opt: seccomp:unconfined` — removes syscall filter
- `ulimits.memlock: -1` — required for `--mlock` to pin weights in RAM
- `mem_limit` / `memswap_limit` — bounds bad model loads and prevents swap storms
- `MALLOC_ARENA_MAX=2` — reduces glibc arena fragmentation

Inference services use `restart: unless-stopped` so they come back after host
reboots. Keep the memory and swap limits set so an invalid model/context choice
is killed inside the container rather than pushing the host into swap.

Two optimisations were tested and **deliberately excluded** because they
degraded token-generation speed ~10% on CUDA + CPU-MoE workloads:
`jemalloc LD_PRELOAD` (conflicts with CUDA's allocator) and
`OMP_WAIT_POLICY=passive` (adds OpenMP thread wake-up latency per token).
