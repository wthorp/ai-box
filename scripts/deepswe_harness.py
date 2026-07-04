#!/usr/bin/env python3
"""Structured helpers for the DeepSWE Pier harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import transition_analysis


LOCAL_AGENT_IMPORT_PATH = "scripts.pier_agents.mini_swe_agent_run:MiniSweAgentRun"
LOCAL_ENVIRONMENT_IMPORT_PATH = (
    "scripts.pier_envs.docker_llm_proxy:DockerLlmProxyEnvironment"
)

RESULT_FIELDS = [
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
]

TSV_HEADER = [
    "quant",
    "context",
    "cpu_moe",
    "gpu_moe",
    "job_name",
    "job_dir",
    "status",
    "pier_exit",
    *RESULT_FIELDS,
    "free_ram_before_mib",
    "free_ram_after_load_mib",
    "free_ram_after_completion_mib",
    "vram_used_mib",
    "vram_total_mib",
    "vram_pct",
    "note",
]


def split_extra(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in args:
        return args, []
    index = args.index("--")
    return args[:index], args[index + 1 :]


def has_cost_limit_arg(extra_args: list[str]) -> bool:
    for index, arg in enumerate(extra_args):
        next_arg = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        if arg.startswith("--agent-kwarg=cost_limit=") or arg.startswith(
            "cost_limit="
        ):
            return True
        if arg == "--agent-kwarg" and next_arg.startswith("cost_limit="):
            return True
    return False


def should_default_cost_limit(agent: str, agent_import_path: str) -> bool:
    return agent == "mini-swe-agent" or agent_import_path == LOCAL_AGENT_IMPORT_PATH


def build_pier_args(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    pier_args = ["run", "-p", args.task_path, "--model", args.model]
    if args.job_name:
        pier_args.extend(["--job-name", args.job_name])
    if args.jobs_dir:
        pier_args.extend(["--jobs-dir", args.jobs_dir])
    if args.n_tasks is not None:
        pier_args.extend(["--n-tasks", str(args.n_tasks)])
    if args.sample_seed is not None:
        pier_args.extend(["--sample-seed", str(args.sample_seed)])
    if args.n_concurrent is not None:
        pier_args.extend(["--n-concurrent", str(args.n_concurrent)])
    if args.quiet_yes:
        pier_args.extend(["--quiet", "--yes"])
    if args.debug_harness:
        pier_args.extend(["--debug", "--no-delete"])

    if args.agent_import_path:
        pier_args.extend(["--agent-import-path", args.agent_import_path])
    else:
        pier_args.extend(["--agent", args.agent])

    if args.environment_import_path:
        pier_args.extend(["--environment-import-path", args.environment_import_path])

    if should_default_cost_limit(
        args.agent, args.agent_import_path
    ) and not has_cost_limit_arg(extra_args):
        pier_args.extend(["--agent-kwarg", "cost_limit=None"])

    pier_args.extend(extra_args)
    return pier_args


def parse_float_file(path: Path) -> float:
    try:
        return float(path.read_text().strip() or "0")
    except Exception:
        return 0.0


def summarize_exceptions(counter: Counter[str]) -> str:
    return ";".join(
        f"{name}={count}"
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def summarize_result_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {field: "" for field in RESULT_FIELDS}

    data = json.loads(path.read_text())
    stats = data.get("stats") or {}
    evals = stats.get("evals") or {}
    passed_trials = 0.0
    saw_reward_stats = False
    exception_counts: Counter[str] = Counter()

    for eval_data in evals.values():
        reward_stats = (eval_data.get("reward_stats") or {}).get("reward") or {}
        if reward_stats:
            saw_reward_stats = True
        for reward_text, trials in reward_stats.items():
            try:
                reward = float(reward_text)
            except ValueError:
                reward = 0.0
            passed_trials += reward * len(trials or [])
        for exc_name, trials in (eval_data.get("exception_stats") or {}).items():
            exception_counts[exc_name] += len(trials or [])

    if not saw_reward_stats:
        passed_trials = (
            int(stats.get("n_completed_trials") or 0)
            - int(stats.get("n_errored_trials") or 0)
            - int(stats.get("n_cancelled_trials") or 0)
        )
        passed_trials = max(passed_trials, 0)

    total_trials = int(data.get("n_total_trials") or stats.get("n_total_trials") or 0)
    pass_rate = str(round(100 * passed_trials / total_trials, 1)) if total_trials else ""
    passed_text = (
        str(round(passed_trials, 3)).rstrip("0").rstrip(".")
        if passed_trials
        else "0"
    )

    return {
        "finished_at": str(data.get("finished_at") or ""),
        "total_trials": str(total_trials or ""),
        "completed_trials": str(stats.get("n_completed_trials") or ""),
        "errored_trials": str(stats.get("n_errored_trials") or ""),
        "cancelled_trials": str(stats.get("n_cancelled_trials") or ""),
        "passed_trials": passed_text,
        "pass_rate_pct": pass_rate,
        "running_trials": str(stats.get("n_running_trials") or ""),
        "pending_trials": str(stats.get("n_pending_trials") or ""),
        "exception_summary": summarize_exceptions(exception_counts),
    }


def agent_exit_status(trial_dir: Path) -> str:
    diag_path = trial_dir / "agent" / "mini-swe-agent.diagnostics.txt"
    try:
        for line in diag_path.read_text().splitlines():
            if line.startswith("exit_code="):
                code = line.split("=", 1)[1].strip()
                if code and code != "0":
                    return f"agent_nonzero:{code}"
    except OSError:
        pass
    return ""


def classify_missing_reward(trial_dir: Path) -> str:
    trajectory_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
    verifier_stdout_path = trial_dir / "verifier" / "test-stdout.txt"

    if status := agent_exit_status(trial_dir):
        return status
    if not trajectory_path.exists():
        return "agent_missing_trajectory"
    if not verifier_stdout_path.exists():
        return "verifier_missing_stdout"

    try:
        trajectory = json.loads(trajectory_path.read_text())
        exit_events = [
            item for item in trajectory.get("messages", []) if item.get("role") == "exit"
        ]
        if exit_events:
            return (
                ((exit_events[-1].get("extra") or {}).get("exit_status"))
                or "verifier_missing_reward"
            )
    except Exception:
        pass
    return "verifier_missing_reward"


def summarize_trial_artifacts(job_dir: Path) -> dict[str, str]:
    if not job_dir.is_dir():
        return {field: "" for field in RESULT_FIELDS}

    trial_dirs = [path for path in job_dir.iterdir() if path.is_dir()]
    if not trial_dirs:
        return {field: "" for field in RESULT_FIELDS}

    completed = 0
    errored = 0
    passed = 0.0
    latest_mtime = 0.0
    exceptions: Counter[str] = Counter()

    for trial_dir in trial_dirs:
        for root, _, files in os.walk(trial_dir):
            for file_name in files:
                try:
                    latest_mtime = max(
                        latest_mtime, (Path(root) / file_name).stat().st_mtime
                    )
                except OSError:
                    pass

        reward_path = trial_dir / "verifier" / "reward.txt"
        if reward_path.exists():
            completed += 1
            reward = parse_float_file(reward_path)
            passed += reward
            if reward <= 0:
                exceptions["verifier_reward_0"] += 1
            continue

        errored += 1
        exceptions[classify_missing_reward(trial_dir)] += 1

    total = len(trial_dirs)
    return {
        "finished_at": dt.datetime.fromtimestamp(latest_mtime).isoformat()
        if latest_mtime
        else "",
        "total_trials": str(total),
        "completed_trials": str(completed),
        "errored_trials": str(errored),
        "cancelled_trials": "0",
        "passed_trials": str(round(passed, 3)).rstrip("0").rstrip(".")
        if passed
        else "0",
        "pass_rate_pct": str(round(100 * passed / total, 1)) if total else "",
        "running_trials": "0",
        "pending_trials": "0",
        "exception_summary": summarize_exceptions(exceptions),
    }


def write_job_telemetry_summary(job_dir: Path) -> None:
    if not job_dir.is_dir():
        return

    trials: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        telemetry_path = trial_dir / "telemetry" / "summary.json"
        telemetry: dict[str, object] = {}
        if telemetry_path.exists():
            try:
                parsed = json.loads(telemetry_path.read_text())
                if isinstance(parsed, dict):
                    telemetry = parsed
            except Exception:
                telemetry = {}

        reward_path = trial_dir / "verifier" / "reward.txt"
        reward: float | None = None
        if reward_path.exists():
            reward = parse_float_file(reward_path)
            failure_class = "passed" if reward > 0 else "verifier_reward_0"
        else:
            failure_class = classify_missing_reward(trial_dir)

        mcp_count = int(telemetry.get("mcp_call_count") or 0)
        changed_count = int(telemetry.get("changed_file_count") or 0)
        transition_summary = transition_analysis.analyze_trial(trial_dir)
        totals["mcp_call_count"] += mcp_count
        totals["changed_file_count"] += changed_count
        totals["trial_count"] += 1
        if reward is not None:
            totals["completed_trial_count"] += 1
            totals["passed_trial_count"] += int(reward > 0)
        if mcp_count:
            totals["trials_with_mcp_calls"] += 1
        if changed_count:
            totals["trials_with_changes"] += 1
        for signal, count in (
            transition_summary.get("failure_signals") or {}
        ).items():
            totals[f"failure_signal:{signal}"] += int(count)
        fail_score = transition_summary.get("fail_score")
        if isinstance(fail_score, int | float):
            totals["fail_score_sum_milli"] += int(round(float(fail_score) * 1000))

        trials.append(
            {
                "trial": trial_dir.name,
                "reward": reward,
                "failure_class": failure_class,
                "telemetry": telemetry,
                "transition_analysis": transition_summary,
            }
        )

    if not trials:
        return

    if totals.get("trial_count"):
        totals["fail_score_avg_milli"] = int(
            round(totals.get("fail_score_sum_milli", 0) / totals["trial_count"])
        )
    output = {
        "job_dir": str(job_dir),
        "totals": dict(totals),
        "trials": trials,
    }
    (job_dir / "telemetry-summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def determine_status(
    result: dict[str, str], result_json_exists: bool, pier_exit: int
) -> tuple[str, str]:
    total_trials = result.get("total_trials", "")
    errored_trials = result.get("errored_trials", "0") or "0"
    passed_trials = result.get("passed_trials", "0") or "0"
    pass_rate = result.get("pass_rate_pct", "")
    exception_summary = result.get("exception_summary", "")

    status = "ok"
    note = ""
    if pier_exit != 0:
        status = "pier_failed"
        note = "pier exited nonzero"
        if exception_summary:
            note += f"; pier_exception:{exception_summary.split('=', 1)[0].split(';', 1)[0]}"
    elif not result_json_exists and not total_trials:
        status = "missing_result"
        note = "result.json and trial artifacts missing"
    elif errored_trials != "0":
        status = "deep_swe_failed"
        note = "DeepSWE reported errored trials"
    elif passed_trials != (total_trials or "0"):
        status = "deep_swe_failed"
        note = "DeepSWE reward below passing"
    elif not result.get("finished_at", ""):
        status = "incomplete"
        note = "job did not finish"

    if pass_rate:
        if note:
            note += "; "
        note += f"passed {passed_trials}/{total_trials or '0'} ({pass_rate}%)"
        if exception_summary:
            note += f"; {exception_summary}"

    return status, note


def deterministic_critic_note(job_dir: Path) -> str:
    summary_path = job_dir / "telemetry-summary.json"
    if not summary_path.exists():
        return ""
    try:
        parsed = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    totals = parsed.get("totals") or {}
    if not isinstance(totals, dict):
        return ""
    notes: list[str] = []
    prefix = "failure_signal:deterministic_critic_failure:"
    for key, value in sorted(totals.items()):
        if not key.startswith(prefix):
            continue
        try:
            count = int(value)
        except Exception:
            continue
        if count > 0:
            notes.append(f"critic:{key.removeprefix(prefix)}={count}")
    return "; ".join(notes)


def emit_nul(values: list[str]) -> None:
    sys.stdout.buffer.write("\0".join(values).encode())
    sys.stdout.buffer.write(b"\0")


def cmd_pier_args(argv: list[str], exec_pier: bool = False) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent", default="mini-swe-agent")
    parser.add_argument("--agent-import-path", default=LOCAL_AGENT_IMPORT_PATH)
    parser.add_argument(
        "--environment-import-path", default=LOCAL_ENVIRONMENT_IMPORT_PATH
    )
    parser.add_argument("--debug-harness", action="store_true")
    parser.add_argument("--job-name")
    parser.add_argument("--jobs-dir")
    parser.add_argument("--n-tasks")
    parser.add_argument("--sample-seed")
    parser.add_argument("--n-concurrent")
    parser.add_argument("--quiet-yes", action="store_true")

    known, extra = split_extra(argv)
    args = parser.parse_args(known)
    pier_args = build_pier_args(args, extra)
    if exec_pier:
        os.execvp("pier", ["pier", *pier_args])
    emit_nul(pier_args)
    return 0


def cmd_summarize_job(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--pier-exit", required=True, type=int)
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir)
    result_json = job_dir / "result.json"
    if result_json.exists():
        result = summarize_result_json(result_json)
    else:
        result = summarize_trial_artifacts(job_dir)

    write_job_telemetry_summary(job_dir)
    status, note = determine_status(result, result_json.exists(), args.pier_exit)
    critic_note = deterministic_critic_note(job_dir)
    if critic_note:
        note = f"{note}; {critic_note}" if note else critic_note
    emit_nul([*(result[field] for field in RESULT_FIELDS), status, note])
    return 0


def cmd_debug_snapshot(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args(argv)

    debug_dir = Path(args.job_dir) / "harness-debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{args.phase}.txt").write_text(
        "\n".join(
            [
                f"phase={args.phase}",
                f"timestamp={dt.datetime.now(dt.timezone.utc).isoformat()}",
                f"compose_project_name={os.environ.get('COMPOSE_PROJECT_NAME', 'ai-box')}",
                f"service={args.service}",
                f"job_dir={args.job_dir}",
                "",
            ]
        )
    )
    with (debug_dir / f"{args.phase}.compose-projects.json").open("w") as out:
        subprocess.run(
            ["docker", "compose", "ls", "--format", "json"],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    with (debug_dir / f"{args.phase}.containers.txt").open("w") as out:
        subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Labels}}",
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return 0


def cmd_analyze_trial(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", required=True)
    args = parser.parse_args(argv)
    summary = transition_analysis.analyze_trial(Path(args.trial_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: deepswe_harness.py "
            "<pier-args|run-pier|summarize-job|analyze-trial|debug-snapshot|tsv-header>",
            file=sys.stderr,
        )
        return 2
    command, rest = argv[0], argv[1:]
    if command == "pier-args":
        return cmd_pier_args(rest)
    if command == "run-pier":
        return cmd_pier_args(rest, exec_pier=True)
    if command == "summarize-job":
        return cmd_summarize_job(rest)
    if command == "analyze-trial":
        return cmd_analyze_trial(rest)
    if command == "debug-snapshot":
        return cmd_debug_snapshot(rest)
    if command == "tsv-header":
        print("\t".join(TSV_HEADER))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
