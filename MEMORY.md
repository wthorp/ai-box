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

1. **Per-MoE max context** — `./tune_quants.sh` probes largest ctx **for each** `CPU_MOE_LAYERS` value (not one global ctx for all MoE). Uses `docker compose rm -sf` + `--force-recreate`.
2. **Min context** — Q4: **65536** minimum; Q5: **32768**; Q6: **16384**; Q8: **8192**.
3. **VRAM** — `wait_vram_stable` after load; pick **≥85% GPU VRAM** with best tok/s; else highest VRAM; else best tok/s.
4. **Bad run signal** — If all MoE points show the same ~15 GB VRAM, the run is stale (old bug: single probe at 262144 without per-MoE).

Empirical Q4 sweep (fixed **ctx=65536**, force-recreate, 2026-05-28) — see also `eval-results/moe-tune-summary.md`:

| `CPU_MOE_LAYERS` | GPU VRAM | tok/s (32 tok) |
|------------------|----------|----------------|
| **28** | 22573 MiB (~92%) | **35.8** |
| 32 | 18869 MiB (~77%) | 34.1 |
| 36 | 15165 MiB (~62%) | 32.4 |

Higher `n-cpu-moe` pushes experts to **host RAM**; GPU fill drops by design.

## Per-quant recommendations

Copy into `.env` for the quant you are serving. Regenerate with `./tune_quants.sh`.

<!-- TUNE_QUANTS:START -->

Updated: 2026-05-29 — full grid sweep `grid-sweep-20260529-122108`; Q8 add-on `grid-sweep-20260529-150326`

| Quant | CPU_MOE | CONTEXT | Note |
|-------|---------|---------|------|
| Q4 | **28** | **262144** | Best Q4 grid point: **44.60 wall tok/s**, 52.64 decode tok/s, 26.6 GiB RAM, 97.8% VRAM. Same moe=28 won 32k/64k/128k too. |
| Q5 | **32** | **65536** | Best Q5 grid point by tok/s: **36.51 wall tok/s**, 42.76 decode tok/s, 36.0 GiB RAM, 90.2% VRAM. At 256k: moe=32, 36.49 tok/s, 96.1% VRAM. |
| Q6 | **34** | **32768** | Best Q6 32k point: **30.43 wall tok/s**, 35.57 decode tok/s, 47.0 GiB RAM, 97.3% VRAM. For 64k/128k/256k use moe=36. |
| Q8 | **38** | **131072** | Best Q8 grid point by tok/s: **26.30 wall tok/s**, 30.68 decode tok/s, 61.7 GiB RAM, 90.9% VRAM. At 256k: moe=38, 25.52 tok/s, 94.8% VRAM. |

<!-- TUNE_QUANTS:END -->

**Latest automated run:** newest `eval-results/tune-quants-*/recommendations.{tsv,env}` and `run.log`.

## Orchestration

- Workflow scripts **re-exec in `runner`** (`AI_BOX_RUNNER=1`); Harbor in image, not host mount.
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
| Q4 MoE write-up | `eval-results/moe-tune-summary.md` (on host) |
| Per-quant tune output | `eval-results/tune-quants-*/` |
| Scripts | `scripts/tune_quants.sh`, `scripts/light_tune.sh`, `scripts/lib/probe_context.sh` |

---

*Last updated: 2026-05-29 — Q6 recommendation filled; Q5 pending; Q8 not viable with current 88g/no-mmap/mlock setup.*
