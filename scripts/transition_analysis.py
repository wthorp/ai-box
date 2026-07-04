#!/usr/bin/env python3
"""Transition-level failure signals for DeepSWE agent trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EDIT_MARKERS = (
    ">",
    ">>",
    "sed -i",
    "apply_patch",
    "python3 -",
    "python -",
    "perl -",
    "tee ",
    "cp ",
    "mv ",
    "touch ",
    "npm pkg",
)

TEST_MARKERS = (
    "test",
    "pytest",
    "unittest",
    "go test",
    "cargo test",
    "npm test",
    "pnpm test",
    "yarn test",
    "vitest",
    "jest",
    "rspec",
    "mvn test",
    "gradle test",
    "tsc",
    "typecheck",
)

LARGE_REWRITE_MARKERS = (
    "cat >",
    "cat <<",
    "tee ",
)

PATH_RE = re.compile(
    r"\b[\w./-]+\.(?:ts|tsx|js|jsx|py|rs|go|java|json|toml|yaml|yml|md|c|cpp|h|hpp)\b"
)


def _one_line(value: object, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit]


def _stable_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def _is_edit(name: str, arguments: dict[str, Any]) -> bool:
    if name == "finish":
        return False
    if name != "run_shell":
        return False
    command = str(arguments.get("command") or "").lower()
    return any(marker in command for marker in EDIT_MARKERS)


def _is_validation(name: str, arguments: dict[str, Any]) -> bool:
    if name != "run_shell":
        return False
    command = str(arguments.get("command") or "").lower()
    return any(marker in command for marker in TEST_MARKERS)


def _is_large_rewrite(name: str, arguments: dict[str, Any]) -> bool:
    if name != "run_shell":
        return False
    command = str(arguments.get("command") or "").lower()
    if not any(marker in command for marker in LARGE_REWRITE_MARKERS):
        return False
    return len(command) > 800 or " << " in command or "<<'" in command or '<<"' in command


def _paths_from_action(name: str, arguments: dict[str, Any], text: str = "") -> set[str]:
    paths: set[str] = set()
    if name == "read_file" and arguments.get("path"):
        paths.add(str(arguments["path"]))
    command = str(arguments.get("command") or "")
    paths.update(PATH_RE.findall(command))
    paths.update(PATH_RE.findall(text))
    return paths


def _failure_signals(
    transitions: list[dict[str, Any]], max_prompt_tokens: int
) -> tuple[dict[str, int], float, list[str]]:
    signals: Counter[str] = Counter()
    reasons: list[str] = []

    first_edit_step = next(
        (int(t["step"]) for t in transitions if t.get("is_edit")), None
    )
    first_validation_step = next(
        (int(t["step"]) for t in transitions if t.get("is_validation")), None
    )
    max_step = max((int(t["step"]) for t in transitions), default=0)
    changed_count = sum(1 for t in transitions if t.get("is_edit"))
    validation_count = sum(1 for t in transitions if t.get("is_validation"))
    max_same = max((int(t.get("same_action_run") or 0) for t in transitions), default=0)

    if max_same >= 3:
        signals["identical_action_loop"] += 1
        reasons.append(f"same normalized action repeated {max_same} times")
    if first_edit_step is None and max_step >= 40:
        signals["no_edit_by_step_40"] += 1
        reasons.append("no edit by step 40")
    if first_edit_step is not None and first_edit_step > 40:
        signals["late_first_edit"] += 1
        reasons.append(f"first edit at step {first_edit_step}")
    if changed_count and not validation_count:
        signals["edit_without_validation"] += 1
        reasons.append("edited but never ran a validation command")
    if first_edit_step is not None and first_validation_step is not None:
        if first_validation_step - first_edit_step > 10:
            signals["late_validation_after_edit"] += 1
            reasons.append(
                f"first validation {first_validation_step - first_edit_step} steps after first edit"
            )
    if any(t.get("is_large_rewrite") for t in transitions):
        signals["large_shell_rewrite"] += 1
        reasons.append("large shell rewrite used")
    if max_prompt_tokens >= 80000 and not validation_count:
        signals["context_bloat_without_validation"] += 1
        reasons.append(f"prompt exceeded {max_prompt_tokens} tokens without validation")

    stale_windows = 0
    for index in range(7, len(transitions)):
        window = transitions[index - 7 : index + 1]
        if (
            not any(t.get("new_paths") for t in window)
            and not any(t.get("is_edit") for t in window)
            and not any(t.get("is_validation") for t in window)
        ):
            stale_windows += 1
    if stale_windows:
        signals["stale_evidence_window"] += 1
        reasons.append("8-step window without new file, edit, or validation")

    weights = {
        "identical_action_loop": 0.35,
        "no_edit_by_step_40": 0.25,
        "late_first_edit": 0.15,
        "edit_without_validation": 0.20,
        "late_validation_after_edit": 0.10,
        "large_shell_rewrite": 0.12,
        "context_bloat_without_validation": 0.15,
        "stale_evidence_window": 0.10,
    }
    fail_score = min(1.0, sum(weights[name] for name in signals))
    return dict(signals), round(fail_score, 3), reasons


def analyze_qwen_jsonl(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

    seen_paths: set[str] = set()
    transitions: list[dict[str, Any]] = []
    deterministic_critic_failures: Counter[str] = Counter()
    deterministic_critic_findings: list[dict[str, Any]] = []
    previous_hash = ""
    same_run = 0
    max_prompt_tokens = 0
    last_assistant = ""

    for event in events:
        if event.get("event") == "llm_response":
            usage = event.get("usage") or {}
            max_prompt_tokens = max(max_prompt_tokens, int(usage.get("prompt_tokens") or 0))
        elif event.get("event") == "assistant_content":
            last_assistant = str(event.get("content") or "")
        elif event.get("event") == "tool_call":
            name = str(event.get("name") or "")
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            action_hash = _stable_hash({"name": name, "arguments": arguments})
            same_run = same_run + 1 if action_hash == previous_hash else 1
            previous_hash = action_hash
            paths = _paths_from_action(name, arguments, last_assistant)
            new_paths = sorted(paths - seen_paths)
            seen_paths.update(paths)
            transitions.append(
                {
                    "step": len(transitions) + 1,
                    "action_type": name,
                    "action_hash": action_hash,
                    "same_action_run": same_run,
                    "paths": sorted(paths),
                    "new_paths": new_paths,
                    "is_edit": _is_edit(name, arguments),
                    "is_validation": _is_validation(name, arguments),
                    "is_large_rewrite": _is_large_rewrite(name, arguments),
                    "summary": _one_line({"name": name, "arguments": arguments}),
                }
            )
        elif event.get("event") == "autofix_broker":
            result = event.get("result") or {}
            if not isinstance(result, dict):
                continue
            critics = result.get("critics") or {}
            if not isinstance(critics, dict):
                continue
            for critic_name, critic_result in critics.items():
                if not isinstance(critic_result, dict):
                    continue
                if critic_result.get("ok", True):
                    continue
                signal = f"deterministic_critic_failure:{critic_name}"
                deterministic_critic_failures[signal] += 1
                deterministic_critic_findings.append(
                    {
                        "critic": str(critic_name),
                        "available": critic_result.get("available"),
                        "error": critic_result.get("error"),
                        "analyzed_count": critic_result.get("analyzed_count"),
                        "findings": critic_result.get("findings", [])[:5]
                        if isinstance(critic_result.get("findings"), list)
                        else [],
                    }
                )

    signals, fail_score, reasons = _failure_signals(transitions, max_prompt_tokens)
    if deterministic_critic_failures:
        signals.update(deterministic_critic_failures)
        fail_score = min(1.0, round(fail_score + 0.30, 3))
        reasons.extend(
            f"{signal}={count}"
            for signal, count in sorted(deterministic_critic_failures.items())
        )
    return {
        "format": "qwen-sverklo-jsonl",
        "source": str(path),
        "step_count": len(transitions),
        "tool_call_count": len(transitions),
        "unique_action_count": len({t["action_hash"] for t in transitions}),
        "max_same_action_run": max(
            (int(t["same_action_run"]) for t in transitions), default=0
        ),
        "first_edit_step": next(
            (t["step"] for t in transitions if t["is_edit"]), None
        ),
        "first_validation_step": next(
            (t["step"] for t in transitions if t["is_validation"]), None
        ),
        "changed_file_count_proxy": sum(1 for t in transitions if t["is_edit"]),
        "validation_command_count": sum(1 for t in transitions if t["is_validation"]),
        "new_path_count": len(seen_paths),
        "max_prompt_tokens": max_prompt_tokens,
        "failure_signals": signals,
        "fail_score": fail_score,
        "value_proxy": round(1.0 - fail_score, 3),
        "negative_reasons": reasons,
        "deterministic_critic_findings": deterministic_critic_findings,
        "recent_transitions": transitions[-12:],
    }


def analyze_mini_trajectory(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    messages = data.get("messages") if isinstance(data, dict) else []
    if not isinstance(messages, list):
        messages = []

    assistant_hashes: list[str] = []
    max_same = 0
    same_run = 0
    previous_hash = ""
    all_text: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content") or ""
        if isinstance(content, list):
            text = "\n".join(json.dumps(item, ensure_ascii=True) for item in content)
        else:
            text = str(content)
        all_text.append(text)
        if message.get("role") == "assistant" and text.strip():
            digest = _stable_hash(_one_line(text.lower(), 800))
            assistant_hashes.append(digest)
            same_run = same_run + 1 if digest == previous_hash else 1
            previous_hash = digest
            max_same = max(max_same, same_run)

    joined = "\n".join(all_text)
    signals: Counter[str] = Counter()
    reasons: list[str] = []
    if max_same >= 3:
        signals["repeated_assistant_content"] += 1
        reasons.append(f"assistant content repeated {max_same} times")
    if "tool_call" not in joined and len(messages) > 6:
        signals["no_detected_tool_calls"] += 1
        reasons.append("trajectory has messages but no detected tool calls")
    if "test" not in joined.lower() and len(messages) > 10:
        signals["no_validation_mention"] += 1
        reasons.append("no validation/test mention in trajectory text")

    fail_score = min(
        1.0,
        0.25 * signals["repeated_assistant_content"]
        + 0.20 * signals["no_detected_tool_calls"]
        + 0.15 * signals["no_validation_mention"],
    )
    return {
        "format": "mini-swe-agent-trajectory",
        "source": str(path),
        "message_count": len(messages),
        "assistant_turn_count": len(assistant_hashes),
        "max_same_action_run": max_same,
        "failure_signals": dict(signals),
        "fail_score": round(fail_score, 3),
        "value_proxy": round(1.0 - fail_score, 3),
        "negative_reasons": reasons,
    }


def analyze_trial(trial_dir: Path) -> dict[str, Any]:
    qwen_path = trial_dir / "agent" / "qwen-sverklo.jsonl"
    if qwen_path.exists():
        return analyze_qwen_jsonl(qwen_path)

    mini_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
    if mini_path.exists():
        try:
            return analyze_mini_trajectory(mini_path)
        except Exception as exc:
            return {"format": "mini-swe-agent-trajectory", "error": str(exc)}

    return {"format": "none", "failure_signals": {"missing_agent_trajectory": 1}}
