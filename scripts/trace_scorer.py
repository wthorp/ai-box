#!/usr/bin/env python3
"""Classify Qwen/Sverklo trace failures into actionable buckets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def tool_key(event: dict[str, Any]) -> str:
    return json.dumps(
        {"name": event.get("name"), "arguments": event.get("arguments") or {}},
        sort_keys=True,
        ensure_ascii=True,
    )


def score_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [event for event in events if event.get("event") == "tool_call"]
    denials = [event for event in events if event.get("event") == "tool_denied"]
    policy_aborts = [event for event in events if event.get("event") == "policy_abort"]
    llm_errors = [event for event in events if event.get("event") == "llm_error"]
    edit_calls = [
        event for event in tool_calls
        if event.get("name") == "edit_file"
        or event.get("name", "").startswith("serena_insert_")
        or event.get("name", "").startswith("serena_replace_")
    ]
    shell_edit_calls = [
        event for event in tool_calls
        if event.get("name") == "run_shell"
        and any(marker in str((event.get("arguments") or {}).get("command", "")).lower() for marker in (">", "sed -i", "apply_patch", "tee ", "python -", "python3 -"))
    ]
    test_calls = [
        event for event in tool_calls
        if event.get("name") == "run_shell"
        and any(marker in str((event.get("arguments") or {}).get("command", "")).lower() for marker in ("pytest", "go test", "cargo test", "npm test", "yarn test", "pnpm test", "jest", "vitest"))
    ]

    max_same_run = 0
    current_key = None
    current_run = 0
    for event in tool_calls:
        key = tool_key(event)
        if key == current_key:
            current_run += 1
        else:
            current_key = key
            current_run = 1
        max_same_run = max(max_same_run, current_run)

    signals: list[str] = []
    if llm_errors:
        signals.append("llm_transport_error")
    if policy_aborts:
        signals.append("policy_abort")
    if denials:
        reasons = Counter(str(event.get("reason") or "") for event in denials)
        reason, _ = reasons.most_common(1)[0]
        if "first-edit deadline" in reason:
            signals.append("post_deadline_inspection")
        elif "test patch" in reason:
            signals.append("verifier_patch_access")
        else:
            signals.append("tool_denial")
    if not edit_calls and not shell_edit_calls:
        signals.append("no_edit")
    if max_same_run >= 3:
        signals.append("repeated_tool_loop")
    if (edit_calls or shell_edit_calls) and not test_calls:
        signals.append("edited_without_validation")

    score = 0.0
    weights = {
        "llm_transport_error": 0.50,
        "policy_abort": 0.35,
        "post_deadline_inspection": 0.45,
        "verifier_patch_access": 0.35,
        "tool_denial": 0.20,
        "no_edit": 0.35,
        "repeated_tool_loop": 0.25,
        "edited_without_validation": 0.20,
    }
    for signal in signals:
        score += weights.get(signal, 0.0)
    score = min(1.0, score)
    return {
        "tool_call_count": len(tool_calls),
        "edit_call_count": len(edit_calls) + len(shell_edit_calls),
        "test_call_count": len(test_calls),
        "denial_count": len(denials),
        "max_same_action_run": max_same_run,
        "failure_signals": signals,
        "fail_score": round(score, 3),
    }


def score_path(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "agent" / "qwen-sverklo.jsonl"
    return score_events(load_jsonl(path))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    output = {str(path): score_path(Path(path)) for path in args.paths}
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
