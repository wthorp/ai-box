# ai-box

Reproducible AI inference server infrastructure for Ubuntu 26.04.
One bootstrap script sets up the host; Docker Compose runs the services.

## Services

| Service | Description |
|---------|-------------|
| `rotorquant` | llama.cpp fork with planar/iso KV-cache compression and `--n-cpu-moe` for MoE CPU offload |
| `tabbyapi` | Optional TabbyAPI OpenAI-compatible server for EXL3 directory models |

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
| `TABBY_MODEL_NAME` | `gemma-4-31b-dense-exl3` | Optional TabbyAPI model directory under `MODELS_DIR` |
| `TABBY_CONTEXT_SIZE` | `32768` | Optional TabbyAPI max sequence/cache size for EXL3 runs |
| `TABBY_GPU_SPLIT` | `23.0` | Manual single-GPU VRAM budget in GB for 24 GB cards |
| `TABBY_CACHE_MODE` | `Q8` | 8-bit KV cache alias for EXL3 runs |
| `TABBY_OVERRIDE_PRESET` | `qsa_coding` | Generated sampler fallback preset for clients that omit sampling params |

## 3 — Start

```bash
# Auto-tune CPU_MOE_LAYERS and CONTEXT_SIZE from GPU VRAM, then start
./start.sh

# Or start with fixed .env values:
docker compose up -d

# Check health
curl http://localhost:8080/health

# Logs
docker compose logs -f rotorquant
```

### TabbyAPI EXL3 backend

TabbyAPI is available as an alternative OpenAI-compatible backend for EXL3
directory models under `/data/ai/models`, such as `gemma-4-31b-dense-exl3` and
the two MoE quant directories `gemma-4-moe-exl3/gemma-4-moe-4.10bpw` and
`gemma-4-moe-exl3/gemma-4-moe-5.10bpw`, plus
`qwen3-coder-next-4.0bpw`.

```bash
# Start one EXL3 model on port 5000.
TABBY_MODEL_NAME=gemma-4-31b-dense-exl3 \
TABBY_CONTEXT_SIZE=32768 \
docker compose --profile tabby up -d tabbyapi

curl http://localhost:5000/v1/models

# Run the existing DeepSWE path against TabbyAPI instead of rotorquant/Qwen.
INFERENCE_SERVICE=tabbyapi \
./deepswe-run.sh --agent mini-swe-agent --model openai/local --n-tasks 1 --sample-seed 0

# Run the original mini-swe-agent benchmark lane with Sverklo skills available.
INFERENCE_SERVICE=tabbyapi \
./deepswe-run.sh \
  --agent mini-swe-agent \
  --mcp-profile sverklo \
  --model openai/local \
  --n-tasks 1 \
  --n-concurrent 1 \
  --quiet-yes
```

The compose service generates `/app/config.yml` at startup from `TABBY_*`
environment variables. For Gemma/Gemini-named models it defaults to
`tool_format=gemma4`; for Qwen-named models it defaults to
`tool_format=qwen3_coder`. TabbyAPI uses `backend=exllamav3`, a 23 GB manual
GPU split, Q8 KV cache, and a generated `qsa_coding` sampler fallback preset.
Flash attention is enabled automatically by ExLlamaV3 on supported NVIDIA GPUs
such as RTX 3090.
Set `OPENAI_BASE_URL=http://172.17.0.1:5000/v1` directly if you do not want to
use `INFERENCE_SERVICE=tabbyapi`.

The sweep aliases for the three local Gemma options are
`GEMMA4_DENSE_EXL3`, `GEMMA4_MOE_410_EXL3`, and `GEMMA4_MOE_510_EXL3`.
The Qwen EXL3 alias is `QWEN3_CODER_NEXT_EXL3`.

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
| `deepswe-sweep.sh` | Fail-fast DeepSWE sweep across Q4..Q8 and descending CPU MoE |
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

MCP profiles are injected into an eval-results task overlay, leaving the
upstream DeepSWE checkout untouched. The closest-to-baseline MCP lane keeps
Pier's `mini-swe-agent` path and exposes MCP-backed repository skills through a
local `skill` command in the task container. For Sverklo, use the stdio profile
so the MCP server runs inside the live DeepSWE task container and indexes that
trial's repo checkout:

```bash
python3 scripts/deepswe.py run \
  --agent mini-swe-agent \
  --mcp-profile sverklo \
  --model openai/local \
  --n-tasks 1 \
  --n-concurrent 1 \
  --quiet-yes
```

This lane does not replace the agent loop. The fixed instruction addition only
mentions that `skill --help` is available. Skill calls and post-run progress
signals are logged under each trial's `telemetry/` directory, and the harness
also writes `telemetry-summary.json` at the job directory for RL analysis.

`qwen-sverklo` remains available as an experimental targeted adapter, but it is
not the controlled benchmark lane for comparing model/parameter/MCP effects.

There is also an HTTP Sverklo sidecar for plumbing checks:

```bash
docker compose --profile sverklo up -d sverklo
python3 scripts/deepswe.py sverklo-preflight
./deepswe-run.sh --agent codex --mcp-profile sverklo-http --model openai/local --n-tasks 1 --sample-seed 0
```

The sidecar indexes `SVERKLO_PROJECT_PATH` outside the trial container, so it is
only for connectivity tests or fixed host-side checkouts, not primary benchmark
runs.

Keep those as separate result lanes:

- **Quant lane:** `mini-swe-agent`, no MCP/skills, same DeepSWE subset.
- **MCP skill lane:** `mini-swe-agent`, optional Sverklo stdio MCP in the trial
  container, same subset and same model parameters as baseline.
- **TabbyAPI lane:** `mini-swe-agent` pointed at the `tabbyapi` service with an
  EXL3 directory model.
- **Compatibility lane:** `codex` plus MCP/skills, only for comparing Pier's
  high-level agent integration.
- **Fail-fast lane:** `deepswe-sweep.sh`, append-only TSV, stops on the first
  DeepSWE failure or incomplete run.

## 5 — rotorquant

rotorquant ([scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)) is the inference backend for this project. It provides block-diagonal KV cache compression:

- `planar3/4` — 2D Givens rotations
- `iso3/4` — 4D quaternion rotations

It runs on port 8080 as the default inference service.

```bash
docker compose up -d rotorquant
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
| `planar4`/`planar3` KV cache | rotorquant |
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
