# ai-box memory — tuning & operations

Living notes for this repo and the reference RTX 3090 (24 GB) box. Update after
benchmarks or `./tune_quants.sh` runs. Machine-readable outputs also land under
`EVAL_RESULTS_DIR` (host default: `/data/ai/local/eval-results`).

## Hardware (reference)

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 7950X (16c/32t) |
| RAM | 96 GB |
| GPU | NVIDIA RTX 3090, 24 GB VRAM |
| Models dir | `/data/ai/models` |

## Model files (Qwen3-Coder-Next UD)

| Quant | `MODEL_FILE` | Notes |
|-------|----------------|-------|
| Q4 | `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` | Single file (~47 GB) |
| Q5 | `Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf` | 3-part GGUF; llama.cpp loads via part 1 |
| Q6 | `Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf` | same |
| Q8 | `Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf` | same; context probe may try 8192 |

Paths are defined in `scripts/lib/models.sh`.

## Inference defaults (turboquant)

- **Port** 8080, **`-ngl 999`**, **`--no-mmap`**, **`--mlock`**
- KV: `--cache-type-k turbo4`, `--cache-type-v turbo3`
- MoE offload: `--n-cpu-moe` ← `.env` `CPU_MOE_LAYERS`
- Context: `-c` ← `.env` `CONTEXT_SIZE`
- Compose: `IPC_LOCK`, `memlock: -1`, host network/ipc, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `GGML_CUDA_FORCE_MMQ=1`

## Tuning methodology

1. **Per-MoE max context** — `./tune_quants.sh` probes the largest ctx **for each** `CPU_MOE_LAYERS` value and evaluates the current grid at `32k/64k/128k/256k` context windows.
2. **MoE search space** — sweep all sensible even `CPU_MOE_LAYERS` values that still keep a minimum number of active experts on GPU.
3. **Selection rule** — prefer the best tok/s among points that meet the target VRAM band; otherwise fall back to highest VRAM, then best tok/s.
4. **Bad run signal** — if every MoE point reports the same VRAM footprint, treat it as stale or misloaded.

## Per-quant recommendations

Copy into `.env` for the quant you are serving. Regenerate with `./tune_quants.sh`.

<!-- TUNE_QUANTS:START -->

Updated: 2026-05-29 — full grid sweep and Q8 recheck

| Quant | CPU_MOE | CONTEXT | wall tok/s | decode tok/s | RAM | VRAM |
|-------|---------|---------|------------|--------------|-----|------|
| Q4 | **28** | **262144** | **44.60** | 52.64 | 26.6 GiB | 97.8% |
| Q5 | **32** | **65536** | **36.51** | 42.76 | 36.0 GiB | 90.2% |
| Q6 | **34** | **32768** | **30.43** | 35.57 | 47.0 GiB | 97.3% |
| Q8 | **38** | **131072** | **26.30** | 30.68 | 61.7 GiB | 90.9% |

<!-- TUNE_QUANTS:END -->

**Latest automated run:** 2026-05-29 grid sweep; detailed `eval-results/` outputs were cleared after consolidation.

## Orchestration

- Workflow scripts **re-exec in `runner`** (`AI_BOX_RUNNER=1`); DeepSWE/Pier is in the image, not host mount.
- DeepSWE/Pier is the preferred post-tuning benchmark lane. Use `mini-swe-agent`
  for pure model/quant comparisons, then a separate `codex` lane for MCP/skills
  experiments after the model settings are stable.
- **`setup.sh`** is host-only (Docker, NVIDIA toolkit, sysctl/memlock).
- **`./start.sh`** — analytical VRAM model for `CPU_MOE_LAYERS` + max `CONTEXT_SIZE` (use after quant change).
- Rebuild runner after Dockerfile changes: `docker compose build runner`.
- Current sweep results measure a single load + prompt/completion pass per point.
  They populate KV cache during inference, but they do not yet benchmark
  multi-turn KV reuse or prompt-cache warm state across repeated chat turns.

## Pitfalls (learned)

- Changing `.env` without **`docker compose up -d --force-recreate`** does not reload the model.
- Benchmark/tuning scripts must snapshot and restore `.env`; never restore to
  hardcoded Q4/8192 defaults.
- Probes that finish in seconds are usually hitting **stale health** from the previous container.
- `CPU_MOE_LAYERS=35` in `.env.example` is a generic default; **28** is better for Q4 @ ~65k ctx on 3090.
- Host sysctl may still be defaults until `sudo bash setup.sh` is run.
- Use compose project **`ai-box`** consistently if multiple checkouts exist.
- **`runner` sets `MODELS_DIR=/models`** — tuning scripts must re-export `MODELS_DIR` from `.env` before nested `docker compose` or turboquant mounts `/models` (empty) on the host.
- Stop **`workspace-turboquant`** if present; it competes for port 8080 / GPU with `ai-box-turboquant`.

## Related paths

| What | Where |
|------|--------|
| Scripts | `scripts/tune_quants.sh`, `scripts/light_tune.sh`, `scripts/deepswe-sweep.sh`, `scripts/lib/probe_context.sh` |

---

*Last updated: 2026-05-29 — full grid sweep copied in; stale eval-results cleared.*
