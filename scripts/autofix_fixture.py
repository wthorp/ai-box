#!/usr/bin/env python3
"""Create a deterministic autofix/critic fixture job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deepswe_harness


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def init_case_repo(root: Path) -> None:
    run(["git", "init"], root)
    run(["git", "config", "user.email", "test@example.com"], root)
    run(["git", "config", "user.name", "Test"], root)
    (root / "query.sql").write_text("select 1 as value\n", encoding="utf-8")
    run(["git", "add", "query.sql"], root)
    run(["git", "commit", "-m", "base"], root)
    (root / "query.sql").write_text("select from\n", encoding="utf-8")


def broker_payload(repo: Path) -> dict[str, Any]:
    script = Path(__file__).resolve().with_name("autofix_broker.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--cwd", str(repo), "--json", "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        payload["return_code"] = proc.returncode
        payload["stderr"] = proc.stderr
        return payload
    return {"ok": False, "return_code": proc.returncode, "stderr": proc.stderr}


def write_fixture_job(job_dir: Path, payload: dict[str, Any]) -> None:
    trial = job_dir / "trial-autofix-fixture"
    (trial / "agent").mkdir(parents=True, exist_ok=True)
    (trial / "verifier").mkdir(parents=True, exist_ok=True)
    (trial / "verifier" / "reward.txt").write_text("0\n", encoding="utf-8")
    events = [
        {
            "event": "tool_call",
            "name": "edit_file",
            "arguments": {"path": "query.sql"},
        },
        {
            "event": "tool_call",
            "name": "run_shell",
            "arguments": {"command": "git diff -- query.sql"},
        },
        {"event": "autofix_broker", "result": payload},
    ]
    (trial / "agent" / "qwen-sverklo.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in events) + "\n",
        encoding="utf-8",
    )
    deepswe_harness.write_job_telemetry_summary(job_dir)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo = Path(repo_tmp)
        init_case_repo(repo)
        payload = broker_payload(repo)

    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        cleanup_job = False
    else:
        job_dir = Path(tempfile.mkdtemp(prefix="qsa-autofix-fixture-"))
        cleanup_job = True
    write_fixture_job(job_dir, payload)
    note = deepswe_harness.deterministic_critic_note(job_dir)
    print(
        json.dumps(
            {
                "job_dir": str(job_dir),
                "cleanup_job": cleanup_job,
                "broker_ok": payload.get("ok"),
                "critic_note": note,
                "telemetry_summary": str(job_dir / "telemetry-summary.json"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if note else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
