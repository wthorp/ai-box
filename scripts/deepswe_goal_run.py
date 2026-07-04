#!/usr/bin/env python3
"""Staged DeepSWE runner for faster, guardrailed full-batch runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DEFAULTS = ROOT / "eval-results"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_cmd(
    cmd: list[str],
    env: dict[str, str] | None = None,
    capture_output: bool = True,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env=merged_env,
            text=True,
            shell=False,
            capture_output=capture_output,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            exc.cmd,
            124,
            stdout=exc.stdout or "",
            stderr=f"timeout_exceeded={timeout}",
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            [str(exc.filename)],
            127,
            stdout="",
            stderr=f"command_not_found={exc.filename}",
        )


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def vram_mib() -> tuple[int | None, int | None]:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return None, None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None, None
    used, total = lines[0].split(",", 1)
    if not (used.strip().isdigit() and total.strip().isdigit()):
        return None, None
    return int(used), int(total)


def free_mem_mib() -> int | None:
    result = run_cmd(["free", "-m"], capture_output=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 7:
                try:
                    return int(fields[6])
                except Exception:
                    return None
    return None


def parse_nul_summary(payload: str) -> dict[str, str]:
    fields = [
        "finished_at",
        "total_trials",
        "completed_trials",
        "errored_trials",
        "cancelled_trials",
        "passed_trials",
        "pass_rate_pct",
        "running_trials",
        "pending_trials",
        "exception_summary",
        "status",
        "note",
    ]
    parts = payload.split("\0")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    parsed: dict[str, str] = {field: "" for field in fields}
    for idx, field in enumerate(fields):
        if idx < len(parts):
            parsed[field] = parts[idx]
    return parsed


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")


def resolve_job_dir(base_dir: Path, expected_name: str) -> Path | None:
    expected = base_dir / expected_name
    if expected.exists():
        return expected

    candidates: list[tuple[float, Path]] = []
    if not base_dir.exists():
        return None

    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue
        if candidate.name == expected_name or candidate.name.startswith(expected_name):
            try:
                candidates.append((candidate.stat().st_mtime, candidate))
            except Exception:
                candidates.append((0.0, candidate))
        if expected_name in candidate.name:
            try:
                candidates.append((candidate.stat().st_mtime, candidate))
            except Exception:
                candidates.append((0.0, candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    for candidate in base_dir.rglob(expected_name + "*"):
        if candidate.is_dir():
            try:
                return candidate
            except Exception:
                return candidate
    return None


def merge_extra_env(items: list[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"extra env spec must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        merged[key] = value
    return merged


def should_advance(summary: dict[str, str], threshold_pct: float) -> tuple[bool, str]:
    status = summary.get("status", "")
    if status != "ok":
        return False, f"status={status or 'unknown'}"

    try:
        pass_rate = float(summary.get("pass_rate_pct", "0") or 0)
    except ValueError:
        pass_rate = 0.0

    if pass_rate < threshold_pct:
        return False, f"pass_rate={pass_rate:.1f}% < {threshold_pct:.1f}%"
    return True, "ok"


def with_dry_run_flag(args: list[str]) -> list[str]:
    output = list(args)
    if "--dry-run" in output:
        return output
    if "--" in output:
        idx = output.index("--")
        output.insert(idx, "--dry-run")
        return output
    output.append("--dry-run")
    return output


@dataclass
class PhaseConfig:
    name: str
    n_tasks: int
    pass_threshold_pct: float
    step_limit: int
    timeout_seconds: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="mini-swe-agent")
    parser.add_argument("--agent-import-path")
    parser.add_argument("--environment-import-path")
    parser.add_argument("--model", default="openai/local")
    parser.add_argument("--task-path", default=os.environ.get("DEEPSWE_DIR", "/deep-swe") + "/tasks")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--results-dir", default=str(EVAL_DEFAULTS))

    parser.add_argument(
        "--profiles",
        default="none,sverklo",
        help="Comma-separated MCP profiles to run (default: none,sverklo).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Exclude baseline lane (profile none).",
    )

    parser.add_argument("--phase1-tasks", type=int, default=5)
    parser.add_argument("--phase2-tasks", type=int, default=15)
    parser.add_argument("--phase3-tasks", type=int, default=30)
    parser.add_argument("--phase1-pass-threshold", type=float, default=80.0)
    parser.add_argument("--phase2-pass-threshold", type=float, default=85.0)
    parser.add_argument("--phase3-pass-threshold", type=float, default=85.0)
    parser.add_argument("--phase1-step-limit", type=int, default=40)
    parser.add_argument("--phase2-step-limit", type=int, default=120)
    parser.add_argument("--phase3-step-limit", type=int, default=180)
    parser.add_argument("--phase1-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--phase2-timeout-minutes", type=float, default=90.0)
    parser.add_argument("--phase3-timeout-minutes", type=float, default=180.0)
    parser.add_argument("--global-timeout-minutes", type=float, default=0.0)

    parser.add_argument("--qsa-early-stop", type=int, default=1)
    parser.add_argument("--qsa-fail-score-abort", type=float, default=0.70)
    parser.add_argument("--qsa-loop-abort-repeats", type=int, default=5)
    parser.add_argument("--qsa-no-edit-abort-step", type=int, default=35)
    parser.add_argument("--qsa-stale-abort-steps", type=int, default=12)
    parser.add_argument("--qsa-max-steps", type=int, default=180)
    parser.add_argument("--qsa-llm-timeout-sec", type=float, default=240.0)

    parser.add_argument("--preflight-tests", action="store_true", default=True)
    parser.add_argument("--skip-preflight-tests", action="store_true")
    parser.add_argument("--agent-kwarg", action="append", default=[], help="extra --agent-kwarg key=value")
    parser.add_argument("--extra-env", action="append", default=[], help="extra environment KEY=VALUE")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--continue-on-fail", action="store_true", help="run remaining phases after gate failures")
    parser.add_argument("--dry-run-only", action="store_true", help="compute commands and summaries only")
    return parser.parse_args(argv)


def normalize_profiles(parsed: argparse.Namespace) -> list[str]:
    seen: set[str] = set()
    profiles: list[str] = []
    for profile in (item.strip() for item in parsed.profiles.split(",") if item.strip()):
        if profile in seen:
            continue
        seen.add(profile)
        profiles.append(profile)
    if not profiles:
        return []
    if not parsed.no_baseline:
        if "none" not in profiles:
            profiles.insert(0, "none")
    else:
        profiles = [profile for profile in profiles if profile != "none"]
    return profiles


def build_deepswe_args(
    parsed: argparse.Namespace,
    profile: str,
    jobs_dir: Path,
    phase: PhaseConfig,
    run_id: str,
) -> list[str]:
    args = [
        "run",
        "--agent",
        parsed.agent,
        "--task-path",
        parsed.task_path,
        "--model",
        parsed.model,
        "--mcp-profile",
        profile,
        "--results-dir",
        parsed.results_dir,
        "--n-tasks",
        str(phase.n_tasks),
        "--sample-seed",
        str(parsed.sample_seed),
        "--n-concurrent",
        str(parsed.n_concurrent),
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        f"{run_id}-{profile}-{phase.name}",
        "--quiet-yes",
    ]
    if parsed.agent_import_path:
        args.extend(["--agent-import-path", parsed.agent_import_path])
    if parsed.environment_import_path:
        args.extend(["--environment-import-path", parsed.environment_import_path])

    args.append("--")

    step_kwarg = "step_limit"
    if parsed.agent == "qwen-sverklo":
        step_kwarg = "max_steps"

    args.extend(["--agent-kwarg", f"{step_kwarg}={phase.step_limit}"])
    for kwarg in parsed.agent_kwarg:
        args.extend(["--agent-kwarg", kwarg])
    args.append("--")
    return args


def run_phase(
    parsed: argparse.Namespace,
    run_root: Path,
    profile: str,
    phase: PhaseConfig,
    manifest_path: Path,
    common_env: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    jobs_dir = run_root / "jobs" / profile
    jobs_dir.mkdir(parents=True, exist_ok=True)

    start_utc = datetime.now(timezone.utc).isoformat()
    pre_free = free_mem_mib()
    pre_vram_used, pre_vram_total = vram_mib()

    deepswe_args = build_deepswe_args(parsed, profile, jobs_dir, phase, parsed.run_id)
    job_name = f"{parsed.run_id}-{profile}-{phase.name}"
    expected_job_dir = jobs_dir / job_name

    dry_start = time.perf_counter()
    dry_cmd = [str(ROOT / "scripts/deepswe.py"), *with_dry_run_flag(list(deepswe_args))]
    dry_run = run_cmd(dry_cmd, cwd=ROOT, capture_output=True)
    preprocessing_ms = (time.perf_counter() - dry_start) * 1000.0

    processing_ms = 0.0
    note = ""
    summary: dict[str, str] = {}
    pier_exit: int | None = None

    if dry_run.returncode != 0:
        status = "preflight_dry_run_failed"
        note = dry_run.stderr.strip()[:1200]
    elif parsed.dry_run_only:
        status = "dry_run_only"
        summary = {"status": status, "note": dry_run.stdout.strip()}
    else:
        run_cmd_args = ["python3", str(ROOT / "scripts/deepswe.py"), *deepswe_args]
        process_start = time.perf_counter()
        run_proc = run_cmd(run_cmd_args, env=common_env, capture_output=True, cwd=ROOT, timeout=None if phase.timeout_seconds <= 0 else phase.timeout_seconds)
        processing_ms = (time.perf_counter() - process_start) * 1000.0
        pier_exit = run_proc.returncode
        timed_out = run_proc.returncode == 124 and run_proc.stderr.startswith("timeout_exceeded=")

        resolved_job_dir = resolve_job_dir(jobs_dir, job_name)
        if resolved_job_dir and resolved_job_dir != expected_job_dir:
            note = f"discovered job_dir={resolved_job_dir} (expected {expected_job_dir})"

        if not resolved_job_dir:
            status = "missing_job_dir"
            if not note:
                note = f"expected {expected_job_dir} not found under {jobs_dir}"
            if run_proc.returncode != 0:
                note = (
                    f"{note} (pier_exit={run_proc.returncode}); "
                    f"stdout={run_proc.stdout.strip()[:400]} stderr={run_proc.stderr.strip()[:400]}"
                )
            elif run_proc.returncode == 0:
                note = f"{note}; command reported success but no output job dir found"
        elif timed_out:
            status = "timeout"
            note = run_proc.stderr.strip()[:1200]
        else:
            summarize = run_cmd(
                [
                    "python3",
                    str(ROOT / "scripts/deepswe_harness.py"),
                    "summarize-job",
                    "--job-dir",
                    str(resolved_job_dir),
                    "--pier-exit",
                    str(pier_exit),
                ],
                capture_output=True,
                cwd=ROOT,
            )
            if summarize.returncode != 0:
                status = "summarize_job_failed"
                note = summarize.stderr.strip()[:1200]
            else:
                summary = parse_nul_summary(summarize.stdout)
                status = summary.get("status", "missing_status")
                note = summary.get("note", "")

            if status == "timeout" and resolved_job_dir.exists():
                try:
                    summarize = run_cmd(
                        [
                            "python3",
                            str(ROOT / "scripts/deepswe_harness.py"),
                            "summarize-job",
                            "--job-dir",
                            str(resolved_job_dir),
                            "--pier-exit",
                            str(pier_exit),
                        ],
                        capture_output=True,
                        cwd=ROOT,
                    )
                    if summarize.returncode == 0:
                        summary = parse_nul_summary(summarize.stdout)
                except Exception:
                    summary = summary or {}
        if status == "timeout":
            summary["status"] = "timeout"
            summary.setdefault("note", "")
            summary["note"] = f"{summary['note']}; {note}" if summary["note"] else note
            summary.setdefault("pass_rate_pct", "0")
            summary.setdefault("passed_trials", "0")
            summary.setdefault("completed_trials", "0")
            summary.setdefault("errored_trials", str(phase.n_tasks))
            summary.setdefault("exception_summary", "")
            summary.setdefault("running_trials", "0")
            summary.setdefault("pending_trials", "0")
            summary.setdefault("cancelled_trials", "0")
        if summary.get("status", status) == "":
            summary["status"] = status
        if summary.get("note", "") == "" and note:
            summary["note"] = note

    post_free = free_mem_mib()
    post_vram_used, post_vram_total = vram_mib()

    active_job_dir = resolve_job_dir(jobs_dir, job_name)

    record = {
        "run_id": parsed.run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase.name,
        "lane": profile,
        "mcp_profile": profile,
        "agent": parsed.agent,
        "agent_import_path": parsed.agent_import_path,
        "environment_import_path": parsed.environment_import_path,
        "model": parsed.model,
        "task_path": parsed.task_path,
        "n_tasks": phase.n_tasks,
        "sample_seed": parsed.sample_seed,
        "n_concurrent": parsed.n_concurrent,
        "pass_threshold_pct": phase.pass_threshold_pct,
        "step_limit": phase.step_limit,
        "phase_timeout_seconds": phase.timeout_seconds,
        "start_utc": start_utc,
        "preprocessing_ms": round(preprocessing_ms, 2),
        "processing_ms": round(processing_ms, 2),
        "pier_exit": pier_exit,
        "status": summary.get("status", status),
        "free_ram_before_mib": pre_free,
        "free_ram_after_mib": post_free,
        "pre_vram_used_mib": pre_vram_used,
        "pre_vram_total_mib": pre_vram_total,
        "post_vram_used_mib": post_vram_used,
        "post_vram_total_mib": post_vram_total,
        "job_dir": str(active_job_dir) if active_job_dir else str(expected_job_dir),
        "command": " ".join([str(ROOT / "scripts/deepswe.py"), *deepswe_args]),
        **summary,
    }

    record.setdefault("pass_rate_pct", "0")
    record.setdefault("passed_trials", "0")
    record.setdefault("completed_trials", "0")
    record.setdefault("errored_trials", "0")
    record.setdefault("exception_summary", "")
    record.setdefault("running_trials", "0")
    record.setdefault("pending_trials", "0")
    record.setdefault("cancelled_trials", "0")

    append_manifest(manifest_path, record)

    if parsed.dry_run_only:
        return False, {"reason": "dry_run_only", "record": record}

    passed, reason = should_advance(record, phase.pass_threshold_pct)
    return passed, {"reason": reason, "record": record}


def run_preflight(parsed: argparse.Namespace, manifest_path: Path) -> bool:
    start = time.perf_counter()
    run = run_cmd(
        [
            "python3",
            "-m",
            "unittest",
            "discover",
            "-s",
            "scripts/tests",
            "-p",
            "test_deepswe_harness.py",
        ],
        cwd=ROOT,
        capture_output=True,
    )
    preflight_ms = (time.perf_counter() - start) * 1000.0

    append_manifest(
        manifest_path,
        {
            "run_id": parsed.run_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "phase0",
            "lane": "preflight",
            "mcp_profile": "none",
            "agent": parsed.agent,
            "agent_import_path": parsed.agent_import_path,
            "environment_import_path": parsed.environment_import_path,
            "model": parsed.model,
            "task_path": parsed.task_path,
            "n_tasks": 0,
            "sample_seed": parsed.sample_seed,
            "n_concurrent": parsed.n_concurrent,
            "pass_threshold_pct": 100.0,
            "step_limit": 0,
            "phase_timeout_seconds": 0,
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing_ms": round(preflight_ms, 2),
            "processing_ms": 0.0,
            "pier_exit": run.returncode,
            "free_ram_before_mib": free_mem_mib(),
            "free_ram_after_mib": free_mem_mib(),
            "status": "ok" if run.returncode == 0 else "failed",
            "note": (run.stderr.strip() or run.stdout.strip())[:1200],
            "pass_rate_pct": "100" if run.returncode == 0 else "0",
            "passed_trials": "0",
            "completed_trials": "0",
            "errored_trials": "0",
            "exception_summary": "",
            "running_trials": "0",
            "pending_trials": "0",
            "cancelled_trials": "0",
            "command": " ".join(
                [
                    "python3",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "scripts/tests",
                    "-p",
                    "test_deepswe_harness.py",
                ]
            ),
        },
    )
    return run.returncode == 0


def build_common_env(parsed: argparse.Namespace) -> dict[str, str]:
    env = merge_extra_env(parsed.extra_env)
    env.update(
        {
            "QSA_EARLY_STOP": str(parsed.qsa_early_stop),
            "QSA_FAIL_SCORE_ABORT": str(parsed.qsa_fail_score_abort),
            "QSA_LOOP_ABORT_REPEATS": str(parsed.qsa_loop_abort_repeats),
            "QSA_NO_EDIT_ABORT_STEP": str(parsed.qsa_no_edit_abort_step),
            "QSA_STALE_ABORT_STEPS": str(parsed.qsa_stale_abort_steps),
            "QSA_MAX_STEPS": str(parsed.qsa_max_steps),
            "QSA_LLM_TIMEOUT_SEC": str(parsed.qsa_llm_timeout_sec),
        }
    )
    return env


def validate_common_requirements(parsed: argparse.Namespace) -> None:
    if not Path(parsed.task_path).is_dir():
        raise FileNotFoundError(f"task path missing: {parsed.task_path}")
    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not found; vram metrics will be None", file=sys.stderr)


def build_phases(parsed: argparse.Namespace) -> list[PhaseConfig]:
    global_timeout_seconds = max(0, int(parsed.global_timeout_minutes * 60))

    def apply_global_timeout(minutes: float) -> int:
        requested = max(0.0, minutes) * 60
        seconds = int(requested)
        if global_timeout_seconds <= 0:
            return seconds
        if seconds <= 0:
            return global_timeout_seconds
        return min(seconds, global_timeout_seconds)

    return [
        PhaseConfig(
            "phase1",
            parsed.phase1_tasks,
            parsed.phase1_pass_threshold,
            parsed.phase1_step_limit,
            apply_global_timeout(parsed.phase1_timeout_minutes),
        ),
        PhaseConfig(
            "phase2",
            parsed.phase2_tasks,
            parsed.phase2_pass_threshold,
            parsed.phase2_step_limit,
            apply_global_timeout(parsed.phase2_timeout_minutes),
        ),
        PhaseConfig(
            "phase3",
            parsed.phase3_tasks,
            parsed.phase3_pass_threshold,
            parsed.phase3_step_limit,
            apply_global_timeout(parsed.phase3_timeout_minutes),
        ),
    ]


def main(argv: list[str]) -> int:
    parsed = parse_args(argv)
    parsed.run_id = parsed.run_id or timestamp()
    parsed.results_dir = str(Path(parsed.results_dir).resolve())

    validate_common_requirements(parsed)

    profiles = normalize_profiles(parsed)
    if not profiles:
        print("error: no MCP profiles selected", file=sys.stderr)
        return 1

    run_root = Path(parsed.results_dir) / "deepswe-goal-run" / parsed.run_id
    manifest = Path(parsed.manifest) if parsed.manifest else run_root / "manifest.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)
    common_env = build_common_env(parsed)
    phases = build_phases(parsed)

    if parsed.preflight_tests and not parsed.skip_preflight_tests:
        print("[phase0] unit preflight...")
        if not run_preflight(parsed, manifest):
            print("[phase0] preflight failed; aborting")
            return 1
        print(f"[phase0] ok -> {manifest}")

    any_fail = False
    for profile in profiles:
        print(f"[run] profile={profile}")
        for phase in phases:
            if phase.n_tasks <= 0:
                continue
            print(
                f"[run]   {profile}:{phase.name} n_tasks={phase.n_tasks} "
                f"pass_threshold={phase.pass_threshold_pct}% step_limit={phase.step_limit} "
                f"timeout_sec={phase.timeout_seconds}"
            )
            allowed, details = run_phase(parsed, run_root, profile, phase, manifest, common_env)
            reason = details["reason"]
            if not allowed:
                any_fail = True
                note = details["record"].get("note", "")
                if note:
                    print(f"[{profile}:{phase.name}] blocked: {reason}; note={note[:240]}")
                else:
                    print(f"[{profile}:{phase.name}] blocked: {reason}")
                if not parsed.continue_on_fail:
                    print("[stop] stopping progression after first gate failure.")
                    return 1
            else:
                print(f"[{profile}:{phase.name}] passed")

    if any_fail:
        print(f"[result] completed with gate failures. manifest={manifest}")
        return 1
    print(f"[result] all gates passed. manifest={manifest}")
    print(f"[result] results root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
