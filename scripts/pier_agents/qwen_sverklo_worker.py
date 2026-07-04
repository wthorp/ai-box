#!/usr/bin/env python3
"""Small Qwen3-Coder-Next tool loop for DeepSWE task containers."""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LOG_DIR = Path(os.environ.get("QSA_LOG_DIR", "/logs/agent"))
LOG_JSONL = LOG_DIR / "qwen-sverklo.jsonl"
LOG_TEXT = LOG_DIR / "qwen-sverklo.txt"
SUMMARY = LOG_DIR / "qwen-sverklo-summary.json"

DEFAULT_SVERKLO_TOOLS = [
    "status",
    "context",
    "search",
    "overview",
    "lookup",
    "refs",
    "deps",
    "impact",
    "grep_results",
    "head_results",
    "ctx_slice",
    "ctx_grep",
    "ctx_stats",
]

DEFAULT_SERENA_TOOLS = [
    "initial_instructions",
    "get_symbols_overview",
    "find_symbol",
    "find_referencing_symbols",
    "find_implementations",
    "find_declaration",
    "get_diagnostics_for_file",
    "replace_symbol_body",
    "insert_after_symbol",
    "insert_before_symbol",
    "rename_symbol",
]


def log(event: str, **fields: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "event": event, **fields}
    with LOG_JSONL.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=True) + "\n")
    with LOG_TEXT.open("a", encoding="utf-8") as out:
        out.write(f"[{event}] {json.dumps(fields, ensure_ascii=True)[:8000]}\n")


def truncate(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def result_limit(name: str, default: int) -> int:
    return max(500, env_int(name, default))


def one_line(text: str, limit: int = 240) -> str:
    return truncate(" ".join(text.split()), limit)


def strip_shell_comments(command: str) -> str:
    lines = []
    for line in command.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


class McpStdioClient:
    def __init__(self, command: list[str], cwd: Path, timeout_sec: float = 180.0) -> None:
        self.command = command
        self.cwd = cwd
        self.timeout_sec = timeout_sec
        self._next_id = 1
        self._lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log("mcp_stdout_non_json", line=one_line(line, 1000))
                continue
            msg_id = message.get("id")
            if msg_id is None:
                log("mcp_notification", message=message)
                continue
            pending = self._pending.get(int(msg_id))
            if pending is not None:
                pending.put(message)
            else:
                log("mcp_unexpected_response", message=message)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            log("mcp_stderr", line=line.rstrip())

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"MCP server exited: {self.process.returncode}")
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        assert self.process.stdin is not None
        with self._lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"MCP server exited: {self.process.returncode}")
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            message: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                message["params"] = params
            responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = responses
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        try:
            response = responses.get(timeout=self.timeout_sec)
        finally:
            self._pending.pop(request_id, None)
        if "error" in response:
            raise RuntimeError(json.dumps(response["error"], ensure_ascii=True))
        return response.get("result") or {}

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "qwen-sverklo-worker", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized")

    def list_tools(self) -> list[dict[str, Any]]:
        return list((self.request("tools/list") or {}).get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any], server_name: str = "mcp") -> str:
        result = self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        parts: list[str] = []
        for item in result.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" or "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        if not parts:
            parts.append(json.dumps(result, ensure_ascii=True))
        return truncate(
            "\n".join(parts),
            result_limit(
                f"QSA_{env_key(server_name)}_RESULT_LIMIT",
                env_int("QSA_TOOL_RESULT_LIMIT", 8000),
            ),
        )


def env_key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.upper())


def schema_for_file_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command in the repository. Use for inspection, edits, and tests. Prefer short commands and include timeouts for test commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run."},
                        "timeout_sec": {
                            "type": "number",
                            "description": "Timeout in seconds. Default 60, max 600.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the repository with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository-relative file path."},
                        "start_line": {"type": "number", "description": "1-based start line. Default 1."},
                        "line_count": {"type": "number", "description": "Maximum lines. Default 200, max 1000."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a targeted exact text replacement in one repository file. Prefer this over shell for small edits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository-relative file path."},
                        "old_text": {"type": "string", "description": "Exact existing text to replace."},
                        "new_text": {"type": "string", "description": "Replacement text."},
                        "occurrence": {
                            "type": "number",
                            "description": "1-based occurrence to replace. Default 1.",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the task after editing and testing. Include a concise summary and test result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "tests": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        },
    ]


def to_qwen_tool(server_name: str, tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or "")
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    prefix = server_name.replace("-", "_")
    return {
        "type": "function",
        "function": {
            "name": f"{prefix}_{name}",
            "description": truncate(
                str(tool.get("description") or f"{server_name} {name}"),
                1000,
            ),
            "parameters": schema,
        },
    }


def sampler_options() -> dict[str, int | float]:
    """llama.cpp/rotorquant OpenAI-compatible sampler options."""
    return {
        "max_tokens": env_int("QSA_MAX_TOKENS", 2048),
        "temperature": env_float("QSA_TEMPERATURE", 0.05),
        "top_p": env_float("QSA_TOP_P", 0.8),
        "repeat_penalty": env_float("QSA_REPEAT_PENALTY", 1.12),
        "repeat_last_n": env_int("QSA_REPEAT_LAST_N", 4096),
        # DRY targets exact repeated sequences. Defaults are intentionally
        # firmer after observed DeepSWE analysis loops.
        "dry_multiplier": env_float("QSA_DRY_MULTIPLIER", 1.0),
        "dry_base": env_float("QSA_DRY_BASE", 1.75),
        "dry_allowed_length": env_int("QSA_DRY_ALLOWED_LENGTH", 3),
        "dry_penalty_last_n": env_int("QSA_DRY_PENALTY_LAST_N", 4096),
        # Mirostat encouraged long continuations in the observed runs.
        "mirostat": env_int("QSA_MIROSTAT", 0),
        "mirostat_tau": env_float("QSA_MIROSTAT_TAU", 4.5),
        "mirostat_eta": env_float("QSA_MIROSTAT_ETA", 0.1),
    }


def call_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    base_url = os.environ.get("QSA_BASE_URL", "http://172.17.0.1:8080/v1").rstrip("/")
    payload = {
        "model": os.environ.get("QSA_MODEL", "local"),
        "messages": messages,
        "tools": tools,
        "tool_choice": os.environ.get("QSA_TOOL_CHOICE", "required"),
        **sampler_options(),
    }
    extra_body = os.environ.get("QSA_EXTRA_BODY_JSON")
    if extra_body:
        extra = json.loads(extra_body)
        if not isinstance(extra, dict):
            raise RuntimeError("QSA_EXTRA_BODY_JSON must decode to an object")
        payload.update(extra)
    log(
        "llm_request",
        sampler={key: payload.get(key) for key in sampler_options()},
        message_count=len(messages),
        tool_count=len(tools),
    )
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.environ.get("QSA_LLM_TIMEOUT_SEC", "900"))) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body[:4000]}") from exc
    parsed = json.loads(body)
    log("llm_response", usage=parsed.get("usage"), finish_reason=(parsed.get("choices") or [{}])[0].get("finish_reason"))
    return parsed


def call_chat_with_retries(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    retries = env_int("QSA_LLM_RETRIES", 2)
    delay = env_float("QSA_LLM_RETRY_DELAY_SEC", 8.0)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return call_chat(messages, tools)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            retryable = "LLM HTTP 503" in message or "timed out" in message.lower()
            if attempt >= retries or not retryable:
                raise
            log(
                "llm_retry",
                attempt=attempt + 1,
                retries=retries,
                delay_sec=delay,
                error=one_line(message, 1000),
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def parse_args_json(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_result_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_shell(arguments: dict[str, Any], cwd: Path) -> str:
    command = str(arguments.get("command") or "")
    if not command.strip():
        return "error: missing command"
    timeout = min(int(arguments.get("timeout_sec") or 60), 600)
    started = time.time()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        text=True,
        input="",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        executable="/bin/bash",
    )
    output = {
        "return_code": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout": truncate(proc.stdout, 12000),
        "stderr": truncate(proc.stderr, 12000),
    }
    return json.dumps(output, ensure_ascii=True)


def run_autofix_broker(cwd: Path) -> str:
    script = Path(__file__).resolve().parents[1] / "autofix_broker.py"
    if not script.exists():
        return json.dumps({"ok": False, "error": f"missing {script}"}, ensure_ascii=True)
    timeout = min(env_int("QSA_AUTOFIX_TIMEOUT_SEC", 120), 600)
    proc = subprocess.run(
        [sys.executable, str(script), "--cwd", str(cwd), "--json"],
        cwd=str(cwd),
        text=True,
        input="",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    payload.update(
        {
            "return_code": proc.returncode,
            "stderr": truncate(proc.stderr, env_int("QSA_AUTOFIX_OUTPUT_LIMIT", 3000)),
        }
    )
    return json.dumps(payload, ensure_ascii=True)


def should_run_autofix_after_shell(policy: Any, command_text: str) -> bool:
    compact_command = " ".join(strip_shell_comments(command_text.lower()).split())
    return (
        env_int("QSA_AUTOFIX_AFTER_DIFF", 0) != 0
        and bool(getattr(policy, "edit_seen", False))
        and compact_command.startswith("git diff")
        and int(getattr(policy, "autofix_runs", 0)) < env_int("QSA_AUTOFIX_MAX_RUNS", 1)
    )


def git_diff_signature(cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(cwd),
            text=True,
            input="",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


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
    "mkdir ",
    "install -d",
    "touch ",
    "npm pkg",
)

TEST_MARKERS = (
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
)

INTERACTIVE_EDITOR_COMMANDS = (
    "vi",
    "vim",
    "nvim",
    "nano",
    "emacs",
)

TEST_COMMAND_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|[;&|()\s])go\s+test(\s|$)",
        r"(^|[;&|()\s])cargo\s+test(\s|$)",
        r"(^|[;&|()\s])npm\s+(run\s+)?test(:\S+)?(\s|$)",
        r"(^|[;&|()\s])pnpm\s+(run\s+)?test(:\S+)?(\s|$)",
        r"(^|[;&|()\s])yarn\s+(run\s+)?test(:\S+)?(\s|$)",
        r"(^|[;&|()\s])python3?\s+-m\s+(pytest|unittest)(\s|$)",
        r"(^|[;&|()\s])pytest(\s|$)",
        r"(^|[;&|()\s])unittest(\s|$)",
        r"(^|[;&|()\s])vitest(\s|$)",
        r"(^|[;&|()\s])jest(\s|$)",
        r"(^|[;&|()\s])rspec(\s|$)",
        r"(^|[;&|()\s])mvn\s+test(\s|$)",
        r"(^|[;&|()\s])(./)?gradlew?\s+test(\s|$)",
        r"(^|[;&|()\s])make\s+test(\s|$)",
    )
)

INSTALL_MARKERS = (
    "npm install",
    "pnpm install",
    "yarn install",
    "pip install",
    "uv pip install",
    "poetry install",
    "bundle install",
    "go get ",
    "cargo add ",
)

VERIFIER_PATCH_MARKERS = (
    "/tests/test.patch",
    "tests/test.patch",
    "/test.patch",
)


def is_test_command(command: str) -> bool:
    compact = " ".join(strip_shell_comments(command).lower().split())
    return any(pattern.search(compact) for pattern in TEST_COMMAND_PATTERNS)

BROAD_SVERKLO_TOOLS = {
    "sverklo_context",
    "sverklo_search",
    "sverklo_overview",
    "sverklo_deps",
    "sverklo_impact",
    "serena_activate_project",
    "serena_get_symbols_overview",
    "serena_find_symbol",
    "serena_find_referencing_symbols",
    "serena_search_for_pattern",
}

EDIT_TOOL_NAMES = {
    "edit_file",
    "serena_replace_symbol_body",
    "serena_insert_after_symbol",
    "serena_insert_before_symbol",
}


class AgentPolicy:
    """Small deterministic guardrail layer for Qwen's tool loop."""

    def __init__(self) -> None:
        self.edit_seen = False
        self.test_seen = False
        self.diff_seen = False
        self.broad_sverklo_calls = 0
        self.step = 0
        self.last_tool_hash = ""
        self.repeated_tool_calls = 0
        self.max_repeated_tool_calls = 0
        self.first_edit_step = 0
        self.first_test_step = 0
        self.last_new_evidence_step = 0
        self.seen_paths: set[str] = set()
        self.large_rewrite_seen = False
        self.last_assistant_norm = ""
        self.repeated_assistant_turns = 0
        self.denied_counts: dict[str, int] = {}
        self.verifier_patch_denials = 0
        self.post_deadline_denials = 0
        self.abort_reason = ""
        self.autofix_runs = 0
        self.summary_lines: list[str] = []
        self.max_broad_before_edit = env_int("QSA_MAX_BROAD_SVERKLO_BEFORE_EDIT", 6)
        self.first_edit_deadline = env_int("QSA_FIRST_EDIT_STEP", 8)
        self.no_broad_after_step = env_int("QSA_NO_BROAD_AFTER_STEP", 30)
        self.recent_message_count = env_int("QSA_RECENT_MESSAGE_COUNT", 18)
        self.max_repeated_denials = env_int("QSA_MAX_REPEATED_DENIALS", 4)
        self.max_post_deadline_denials = env_int("QSA_MAX_POST_DEADLINE_DENIALS", 2)
        self.early_stop_enabled = env_int("QSA_EARLY_STOP", 1) != 0
        self.loop_abort_repeats = env_int("QSA_LOOP_ABORT_REPEATS", 5)
        self.no_edit_abort_step = env_int("QSA_NO_EDIT_ABORT_STEP", 18)
        self.validation_grace_steps = env_int("QSA_VALIDATION_GRACE_STEPS", 12)
        self.stale_abort_steps = env_int("QSA_STALE_ABORT_STEPS", 10)

    def phase(self, step: int) -> str:
        if not self.edit_seen:
            return "inspect" if step < self.first_edit_deadline else "edit-now"
        if not self.test_seen:
            return "test"
        return "fix-or-finish"

    def reminder(self, step: int) -> str:
        loop_warning = ""
        if self.repeated_assistant_turns >= 1:
            loop_warning = (
                " Previous assistant response repeated the same analysis; call a concrete edit/test tool now."
            )
        phase = self.phase(step)
        if phase == "inspect":
            remaining = max(0, self.first_edit_deadline - step)
            return (
                f"CONTROL: Inspection budget is limited. First edit is due within {remaining} steps. "
                "Use targeted reads/searches only; do not restate the task unless making a decision."
                + loop_warning
            )
        if phase == "edit-now":
            return (
                "CONTROL: Stop broad exploration now. Call edit_file or run_shell with a concrete edit command in this response. "
                "Use a small exact replacement, patch, file creation, or exact line edit; then run git diff and a targeted test."
                + loop_warning
            )
        if phase == "test":
            return (
                "CONTROL: An edit has been made. Run git diff first, then the narrowest existing test or static check. "
                "Do not search for hidden verifier tests; they were only embedded as context."
                + loop_warning
            )
        return (
            "CONTROL: Alternate test output -> focused fix. If tests pass or no useful progress remains, call finish."
            + loop_warning
        )

    def allow_tool(self, step: int, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        if name == "finish":
            if not self.edit_seen:
                return False, "finish denied: no implementation edit has been made"
            if not self.diff_seen:
                return False, "finish denied: inspect git diff before finishing"
            if not self.test_seen:
                return False, "finish denied: run a relevant validation command before finishing"
        if name == "run_shell":
            command = " ".join(strip_shell_comments(str(arguments.get("command") or "")).lower().split())
            first_word = command.split(maxsplit=1)[0] if command else ""
            if first_word in INTERACTIVE_EDITOR_COMMANDS:
                return False, (
                    "interactive editors are unavailable in this tool loop; use edit_file, apply a small patch, or a noninteractive shell edit"
                )
            if command.startswith("ls -r"):
                return False, (
                    "broad recursive listing wastes context; use rg --files with a narrow pattern or read_file"
                )
            if not self.edit_seen and any(marker in command for marker in INSTALL_MARKERS):
                return False, (
                    "do not install dependencies before editing; inspect source and make the smallest implementation change first"
                )
            if not self.edit_seen and is_test_command(command):
                return False, (
                    "do not run tests before the first implementation edit; hidden task tests are only embedded as context, so edit source first"
                )
            if command.startswith("find .") and "-maxdepth" not in command:
                return False, (
                    "broad find is too expensive; use rg --files or add -maxdepth and a narrow path"
                )
            if command.startswith("find /"):
                return False, (
                    "global filesystem search is too expensive; search the repository with rg --files or a narrow path"
                )
            if any(marker in command for marker in VERIFIER_PATCH_MARKERS):
                return False, (
                    "verifier test patches were already provided as task context and are not part of the repository runtime; use that context to edit implementation files"
                )
            if self.edit_seen and not self.diff_seen and not command.startswith("git diff"):
                return False, (
                    "after editing, inspect git diff before running tests or searching more files"
                )
        if name == "read_file" and self.is_verifier_patch_access(name, arguments):
            return False, (
                "verifier test patches were already provided as task context; do not read them from the repository"
            )

        if not self.edit_seen and step >= self.first_edit_deadline:
            if name == "run_shell":
                command = str(arguments.get("command") or "").lower()
                if not any(marker in command for marker in EDIT_MARKERS):
                    return False, (
                        "first-edit deadline reached; issue a concrete edit command now"
                    )
            elif name not in EDIT_TOOL_NAMES and name != "finish":
                return False, (
                    "first-edit deadline reached; no more inspection before a concrete edit"
                )

        if name in BROAD_SVERKLO_TOOLS:
            if not self.edit_seen and self.broad_sverklo_calls >= self.max_broad_before_edit:
                return False, (
                    "exploration budget exhausted; use read_file on known files or make the first edit"
                )
            if self.edit_seen and step >= self.no_broad_after_step:
                return False, "broad Sverklo exploration is closed after editing; use targeted reads/tests"
        if name == "read_file":
            requested = int(arguments.get("line_count") or 120)
            max_lines = env_int("QSA_READ_FILE_MAX_LINES", 160)
            if requested > max_lines:
                arguments["line_count"] = max_lines
        return True, ""

    def is_verifier_patch_access(self, name: str, arguments: dict[str, Any]) -> bool:
        if name == "run_shell":
            text = strip_shell_comments(str(arguments.get("command") or "")).lower()
        elif name == "read_file":
            text = str(arguments.get("path") or "").lower()
        else:
            return False
        return any(marker in text for marker in VERIFIER_PATCH_MARKERS)

    def fail_score(self) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if self.max_repeated_tool_calls >= self.loop_abort_repeats:
            score += 0.45
            reasons.append(
                f"same tool action repeated {self.max_repeated_tool_calls} times"
            )
        if not self.edit_seen and self.step >= self.no_edit_abort_step:
            score += 0.35
            reasons.append(f"no edit by step {self.step}")
        if (
            self.edit_seen
            and not self.test_seen
            and self.step - self.first_edit_step >= self.validation_grace_steps
        ):
            score += 0.25
            reasons.append(
                f"no validation within {self.validation_grace_steps} steps after edit"
            )
        if (
            self.step >= self.stale_abort_steps
            and self.step - self.last_new_evidence_step >= self.stale_abort_steps
        ):
            score += 0.20
            reasons.append(
                f"no new file/edit/test evidence for {self.stale_abort_steps} steps"
            )
        if self.large_rewrite_seen:
            score += 0.10
            reasons.append("large shell rewrite seen")
        return min(1.0, score), reasons

    def maybe_abort(self) -> None:
        if not self.early_stop_enabled or self.abort_reason:
            return
        score, reasons = self.fail_score()
        threshold = env_float("QSA_FAIL_SCORE_ABORT", 0.70)
        if score >= threshold:
            self.abort_reason = (
                f"early-stop fail_score={score:.2f} value_proxy={1.0 - score:.2f}; "
                + "; ".join(reasons)
            )

    def note_assistant(self, content: str) -> None:
        norm = one_line(content.lower(), 500)
        if norm and norm == self.last_assistant_norm:
            self.repeated_assistant_turns += 1
        else:
            self.repeated_assistant_turns = 0
        self.last_assistant_norm = norm

    def note_tool(
        self,
        step: int,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        shell_edit_changed: bool = False,
    ) -> None:
        self.step = step
        action_hash = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        if action_hash == self.last_tool_hash:
            self.repeated_tool_calls += 1
        else:
            self.repeated_tool_calls = 1
        self.last_tool_hash = action_hash
        self.max_repeated_tool_calls = max(
            self.max_repeated_tool_calls, self.repeated_tool_calls
        )

        new_evidence = False
        if name in BROAD_SVERKLO_TOOLS:
            self.broad_sverklo_calls += 1
        edit_file_ok = name == "edit_file" and parse_result_json(result).get("ok") is True
        if name in EDIT_TOOL_NAMES and name != "edit_file":
            self.edit_seen = True
            if not self.first_edit_step:
                self.first_edit_step = step
            new_evidence = True
        if edit_file_ok:
            self.edit_seen = True
            if not self.first_edit_step:
                self.first_edit_step = step
            new_evidence = True
        if name == "run_shell":
            command = str(arguments.get("command") or "")
            lowered = command.lower()
            compact_command = " ".join(strip_shell_comments(command).lower().split())
            if compact_command.startswith("git diff"):
                self.diff_seen = True
                new_evidence = True
            if any(marker in lowered for marker in EDIT_MARKERS) and shell_edit_changed:
                self.edit_seen = True
                if not self.first_edit_step:
                    self.first_edit_step = step
                new_evidence = True
            if is_test_command(command):
                self.test_seen = True
                if not self.first_test_step:
                    self.first_test_step = step
                new_evidence = True
            if (
                ("cat >" in lowered or "cat <<" in lowered or "tee " in lowered)
                and len(command) > 800
            ):
                self.large_rewrite_seen = True
            for token in command.replace("'", " ").replace('"', " ").split():
                if any(token.endswith(suffix) for suffix in (".ts", ".tsx", ".js", ".py", ".rs", ".go")):
                    if token not in self.seen_paths:
                        self.seen_paths.add(token)
                        new_evidence = True
        if name == "read_file":
            path = str(arguments.get("path") or "")
            if path and path not in self.seen_paths:
                self.seen_paths.add(path)
                new_evidence = True
        if name == "edit_file":
            path = str(arguments.get("path") or "")
            if path and path not in self.seen_paths:
                self.seen_paths.add(path)
                new_evidence = True
        if new_evidence:
            self.last_new_evidence_step = step
        if name == "finish":
            return
        detail = one_line(result, 400)
        if name == "run_shell":
            command = one_line(str(arguments.get("command") or ""), 180)
            line = f"{name}: {command} -> {detail}"
        elif name == "read_file":
            path = str(arguments.get("path") or "")
            start = arguments.get("start_line") or 1
            count = arguments.get("line_count") or 120
            excerpt = one_line(result, env_int("QSA_READ_SUMMARY_CHARS", 1400))
            line = f"{name}: {path}:{start}+{count} => {excerpt}"
        elif name == "edit_file":
            path = str(arguments.get("path") or "")
            line = f"{name}: {path} -> {detail}"
        else:
            line = f"{name}: {detail}"
        self.summary_lines.append(line)
        max_lines = env_int("QSA_SUMMARY_LINES", 36)
        if len(self.summary_lines) > max_lines:
                self.summary_lines = self.summary_lines[-max_lines:]
        self.maybe_abort()

    def note_denial(self, name: str, arguments: dict[str, Any], reason: str) -> None:
        if self.is_verifier_patch_access(name, arguments):
            self.verifier_patch_denials += 1
            if self.verifier_patch_denials >= self.max_post_deadline_denials:
                self.abort_reason = (
                    f"repeated verifier patch access denial for {name}: {reason}"
                )
                return
        if not self.edit_seen and reason.startswith("first-edit deadline reached"):
            self.post_deadline_denials += 1
            if self.post_deadline_denials >= self.max_post_deadline_denials:
                self.abort_reason = (
                    f"repeated post-deadline inspection denial for {name}: {reason}"
                )
                return
        key = json.dumps(
            {"name": name, "arguments": arguments, "reason": reason},
            sort_keys=True,
            ensure_ascii=True,
        )
        self.denied_counts[key] = self.denied_counts.get(key, 0) + 1
        if self.denied_counts[key] >= self.max_repeated_denials:
            self.abort_reason = (
                f"repeated policy denial for {name}: {reason}; arguments={arguments}"
            )

    def compact_messages(self, messages: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
        if len(messages) <= self.recent_message_count + 3:
            compacted = list(messages)
        else:
            compacted = [messages[0], messages[1]]
            if self.summary_lines:
                compacted.append(
                    {
                        "role": "system",
                        "content": "Run progress summary:\n" + "\n".join(self.summary_lines),
                    }
                )
            recent = messages[-self.recent_message_count :]
            while recent and recent[0].get("role") == "tool":
                recent = recent[1:]
            compacted.extend(recent)
        compacted.append({"role": "system", "content": self.reminder(step)})
        return compacted


def read_file(arguments: dict[str, Any], cwd: Path) -> str:
    rel = str(arguments.get("path") or "")
    if not rel:
        return "error: missing path"
    path = (cwd / rel).resolve()
    if not str(path).startswith(str(cwd.resolve())):
        return "error: path escapes repository"
    start = max(1, int(arguments.get("start_line") or 1))
    count = min(
        max(1, int(arguments.get("line_count") or 120)),
        env_int("QSA_READ_FILE_MAX_LINES", 160),
    )
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"error: {exc}"
    selected = lines[start - 1 : start - 1 + count]
    width = len(str(start + len(selected)))
    return "\n".join(f"{index:>{width}}: {line}" for index, line in enumerate(selected, start=start))


def edit_file(arguments: dict[str, Any], cwd: Path) -> str:
    rel = str(arguments.get("path") or "")
    old_text = str(arguments.get("old_text") or "")
    new_text = str(arguments.get("new_text") or "")
    occurrence = max(1, int(arguments.get("occurrence") or 1))
    if not rel:
        return "error: missing path"
    if not old_text:
        return "error: missing old_text"
    path = (cwd / rel).resolve()
    if not str(path).startswith(str(cwd.resolve())):
        return "error: path escapes repository"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"error: {exc}"
    matches = []
    start = 0
    while True:
        index = text.find(old_text, start)
        if index < 0:
            break
        matches.append(index)
        start = index + max(1, len(old_text))
    if len(matches) < occurrence:
        return json.dumps(
            {
                "ok": False,
                "error": "old_text occurrence not found",
                "matches": len(matches),
                "requested_occurrence": occurrence,
            },
            ensure_ascii=True,
        )
    index = matches[occurrence - 1]
    updated = text[:index] + new_text + text[index + len(old_text) :]
    path.write_text(updated, encoding="utf-8")
    return json.dumps(
        {
            "ok": True,
            "path": rel,
            "occurrence": occurrence,
            "matches_before": len(matches),
            "old_bytes": len(old_text.encode("utf-8")),
            "new_bytes": len(new_text.encode("utf-8")),
        },
        ensure_ascii=True,
    )


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def default_allowed_tools(server_name: str) -> list[str]:
    if server_name == "sverklo":
        return DEFAULT_SVERKLO_TOOLS
    if server_name == "serena":
        return DEFAULT_SERENA_TOOLS
    return []


def selected_mcp_tools(server_name: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default = ",".join(default_allowed_tools(server_name))
    raw = os.environ.get(f"QSA_{env_key(server_name)}_TOOLS", default)
    allowed = {name.strip() for name in raw.split(",") if name.strip()}
    selected = [tool for tool in tools if str(tool.get("name")) in allowed]
    if selected:
        return selected
    fallback_limit = env_int(f"QSA_{env_key(server_name)}_FALLBACK_TOOL_LIMIT", 10)
    return tools[:fallback_limit]


def load_mcp_commands(cwd: Path) -> list[dict[str, Any]]:
    raw = os.environ.get("QSA_MCP_COMMANDS_JSON")
    if raw:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                {
                    "name": str(item.get("name") or f"mcp{index}"),
                    "command": list(item.get("command") or []),
                }
                for index, item in enumerate(parsed)
                if isinstance(item, dict) and item.get("command")
            ]
    command = json.loads(os.environ.get("QSA_SVERKLO_COMMAND_JSON", "[]") or "[]")
    if not command:
        return []
    return [{"name": "sverklo", "command": command}]


def startup_guidance() -> str:
    guidance = [
        "STARTUP CONTRACT:",
        "First assistant response must contain a tool call, not prose. Start with sverklo_context, sverklo_search, serena_initial_instructions, serena_get_symbols_overview, read_file, or run_shell. Keep any private analysis short enough that the tool call is emitted.",
        "DeepSWE task tests are the executable spec. If task context includes tests/test.patch or tests/test.sh, treat that embedded text as already inspected and target only files implicated by it. Never read, cat, grep, or apply /tests/test.patch or tests/test.patch through tools.",
        "Do not run dependency installation commands such as npm install, pip install, go get, cargo add, bundle install, poetry install, or yarn install before the first implementation edit. These usually waste the run budget and mutate lockfiles.",
        "When task context already contains new tests, do not try to run those new tests before editing; they are not present in the runtime repository until the verifier applies them. Use embedded tests to infer behavior, edit source, inspect git diff, then run an existing narrow build/typecheck/test command.",
        "If no MCP tools are listed, use read_file and short run_shell commands. Prefer `rg --files`, `rg PATTERN path`, and specific file reads; do not use `ls -R`.",
        "Use MCP before broad shell search when MCP is available: Sverklo for repository-wide text/dependency orientation; Serena for symbols, references, declarations, diagnostics, and symbol edits.",
        "If Serena is available, call serena_initial_instructions once before depending on Serena symbol tools. If a Serena symbol tool fails because a language server is unavailable, fall back immediately to read_file/run_shell and do not retry the same failed Serena call.",
        "Do not stop after inspection. By the first-edit deadline you should have a concrete edit in place. If context is incomplete, make the smallest reversible edit to the most likely file, then inspect/test that result.",
        "For small edits, prefer edit_file with exact old_text/new_text. When editing through shell, use small patches and preserve existing formatting. Scratch files, deleted files, or no-op commands do not count as an implementation edit. After edits, run git diff before validation.",
        "Validation must be narrow first: an existing related package test, or a syntax/build check inferred from project files. Do not search for hidden verifier test files after they fail to run.",
        "FINISH ONLY after a patch plus validation result, or after proving from code that no patch is needed.",
        "TACTICS:",
        "Patch discipline: do not finish until you have either changed repository files or can explicitly explain why no change is needed. After every change, inspect git diff and run the narrowest relevant validation command.",
        "Caveman mode: make one small concrete move at a time. Find file. Read exact code. Patch exact code. Test exact thing. Do not narrate broad theories when a tool call can answer the next question.",
        "Negative predictor avoidance: avoid ending after only search/read steps; avoid repeated identical tool calls; avoid reading many unrelated files; avoid large whole-file rewrites unless the file is tiny.",
        "Sverklo skill: use Sverklo for repo orientation, dependency/blast-radius context, and targeted symbol or text search across the whole problem repository.",
        "Serena skill: use Serena for symbol overview, symbol lookup, reference lookup, and precise symbol-level edits when the target language server supports it.",
        "Validation skill: infer the project's normal test/build command from package files, Makefiles, pyproject, go.mod, Cargo.toml, or README. Run a narrow command first; if unavailable, run a syntax/type/build check.",
        "Finish discipline: finish only after summarizing the patch and the validation command/result. If validation fails, either make one focused fix or report the exact remaining failure.",
    ]
    extra = os.environ.get("QSA_EXTRA_STARTUP_GUIDANCE", "").strip()
    if extra:
        guidance.extend(["TASK-SPECIFIC GUIDANCE:", extra])
    return "\n".join(guidance)


def initial_context() -> str:
    path = os.environ.get("QSA_INITIAL_CONTEXT_FILE")
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = truncate(text.strip(), env_int("QSA_INITIAL_CONTEXT_LIMIT", 24000))
    if not text:
        return ""
    return (
        "PRELOADED TASK CONTEXT:\n"
        "Use this before searching. Prefer editing files implicated by this context.\n"
        f"{text}\n"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-b64", required=True)
    args = parser.parse_args(argv)

    instruction = base64.b64decode(args.instruction_b64).decode("utf-8")
    cwd = Path.cwd().resolve()
    mcp_commands = load_mcp_commands(cwd)
    log("start", cwd=str(cwd), commands=mcp_commands, model=os.environ.get("QSA_MODEL"))

    clients: dict[str, McpStdioClient] = {}
    finished = False
    tool_call_count = 0
    mcp_call_counts: dict[str, int] = {}
    try:
        name_map: dict[str, tuple[str, str]] = {}
        qwen_tools = [*schema_for_file_tools()]
        for item in mcp_commands:
            server_name = str(item["name"]).replace("-", "_")
            client = McpStdioClient(list(item["command"]), cwd)
            clients[server_name] = client
            mcp_call_counts[server_name] = 0
            client.initialize()
            tools = client.list_tools()
            selected = selected_mcp_tools(server_name, tools)
            for tool in selected:
                qwen_name = f"{server_name}_{tool['name']}"
                name_map[qwen_name] = (server_name, str(tool["name"]))
                qwen_tools.append(to_qwen_tool(server_name, tool))
            log(
                "tools_ready",
                server=server_name,
                tools=[tool.get("name") for tool in selected],
                total_tools=len(qwen_tools),
            )

        system = (
            "You are a local coding agent running inside a DeepSWE task container at /app. "
            "Solve the user's software engineering task by inspecting and editing this repository. "
            "Use local MCP skills only for targeted repository orientation, symbol search, references, and precise edits. "
            "Use read_file and run_shell for exact file inspection, edits, git diff, and tests. "
            "Keep exploration short: identify likely files, edit early, then alternate tests and fixes. "
            "Do not repeat analysis. If you have named the likely fix, implement it instead of continuing to inspect. "
            "When you are done, call finish with a concise summary and tests. "
            "Do not claim success without checking the modified files or relevant tests.\n\n"
            + initial_context()
            + startup_guidance()
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        max_steps = int(os.environ.get("QSA_MAX_STEPS", "80"))
        policy = AgentPolicy()
        no_tool_turns = 0
        for step in range(1, max_steps + 1):
            log("step", step=step)
            try:
                response = call_chat_with_retries(
                    policy.compact_messages(messages, step), qwen_tools
                )
            except Exception as exc:
                policy.abort_reason = (
                    f"llm_error: {type(exc).__name__}: {one_line(str(exc), 1000)}"
                )
                log("llm_error", error=policy.abort_reason)
                break
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            messages.append(normalize_message(message))
            tool_calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            if content:
                log("assistant_content", content=truncate(content, 4000))
                policy.note_assistant(content)
            if not tool_calls:
                log("no_tool_calls", finish_reason=choice.get("finish_reason"))
                no_tool_turns += 1
                if (
                    choice.get("finish_reason") == "tool_calls"
                    and no_tool_turns <= env_int("QSA_MAX_EMPTY_TOOL_RETRIES", 2)
                ):
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous response indicated tool_calls but contained no usable tool call. "
                                "Call exactly one tool now. If an edit is needed, use edit_file with exact old_text/new_text. "
                                "If you already edited, call run_shell for git diff or a targeted test."
                            ),
                        }
                    )
                    continue
                if not policy.edit_seen and step >= policy.first_edit_deadline:
                    policy.abort_reason = (
                        "no tool call after first-edit deadline; model continued analysis instead of editing"
                    )
                    break
                if (
                    choice.get("finish_reason") == "length"
                    and no_tool_turns <= env_int("QSA_MAX_NO_TOOL_RETRIES", 3)
                ):
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your previous response ran out before a tool call. "
                                "Do not continue analysis. Call one concrete tool now; "
                                "if enough context is known, make the smallest edit."
                            ),
                        }
                    )
                    continue
                break
            no_tool_turns = 0
            for index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = parse_args_json(function.get("arguments"))
                call_id = tool_call.get("id") or f"call_{step}_{index}"
                log("tool_call", name=name, arguments=arguments, call_id=call_id)
                tool_call_count += 1
                shell_edit_changed = False
                try:
                    allowed_tool, denial = policy.allow_tool(step, name, arguments)
                    if not allowed_tool:
                        result = f"policy: {denial}"
                        policy.note_denial(name, arguments, denial)
                        log("tool_denied", name=name, reason=denial)
                    elif name == "run_shell":
                        command_text = str(arguments.get("command") or "").lower()
                        tracks_edit = any(marker in command_text for marker in EDIT_MARKERS)
                        diff_before = git_diff_signature(cwd) if tracks_edit else ""
                        result = run_shell(arguments, cwd)
                        diff_after = git_diff_signature(cwd) if tracks_edit else ""
                        shell_edit_changed = tracks_edit and diff_after != diff_before
                        if should_run_autofix_after_shell(policy, command_text):
                            policy.autofix_runs += 1
                            autofix_result = run_autofix_broker(cwd)
                            log("autofix_broker", result=parse_result_json(autofix_result))
                            result = (
                                str(result)
                                + "\n\n[AUTOFIX_BROKER]\n"
                                + autofix_result
                                + "\n\n[UPDATED_GIT_DIFF]\n"
                                + run_shell({"command": "git diff", "timeout_sec": 30}, cwd)
                            )
                    elif name == "read_file":
                        result = read_file(arguments, cwd)
                    elif name == "edit_file":
                        result = edit_file(arguments, cwd)
                    elif name == "finish":
                        finished = True
                        result = json.dumps({"ok": True, "finished": True}, ensure_ascii=True)
                        log("finish", arguments=arguments)
                    elif name in name_map:
                        server_name, remote_name = name_map[name]
                        mcp_call_counts[server_name] = mcp_call_counts.get(server_name, 0) + 1
                        result = clients[server_name].call_tool(
                            remote_name, arguments, server_name=server_name
                        )
                    else:
                        result = f"error: unknown tool {name}"
                except Exception as exc:
                    result = f"error: {type(exc).__name__}: {exc}"
                    log("tool_error", name=name, error=result)
                policy.note_tool(
                    step,
                    name,
                    arguments,
                    str(result),
                    shell_edit_changed=shell_edit_changed,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": truncate(
                            str(result), result_limit("QSA_TOOL_MESSAGE_LIMIT", 10000)
                        ),
                    }
                )
                if finished:
                    break
                if policy.abort_reason:
                    log("policy_abort", reason=policy.abort_reason)
                    break
            if finished:
                break
            if policy.abort_reason:
                break
    finally:
        for client in clients.values():
            client.close()

    SUMMARY.write_text(
        json.dumps(
            {
                "finished": finished,
                "tool_call_count": tool_call_count,
                "mcp_call_counts": mcp_call_counts,
                "sverklo_call_count": mcp_call_counts.get("sverklo", 0),
                "serena_call_count": mcp_call_counts.get("serena", 0),
                "abort_reason": policy.abort_reason,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("end", finished=finished, tool_call_count=tool_call_count, mcp_call_counts=mcp_call_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
