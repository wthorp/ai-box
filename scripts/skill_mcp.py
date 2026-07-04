#!/usr/bin/env python3
"""Small local skill CLI that bridges shell commands to configured MCP servers."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any


LOG_DIR = Path(os.environ.get("SKILL_TELEMETRY_DIR", "/logs/telemetry"))
EVENTS = LOG_DIR / "events.jsonl"
ALLOWED_CAVEMAN_FILE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
}


def emit(event: str, **fields: Any) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "event": event, **fields}
        with EVENTS.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        return


def truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _workspace_root() -> Path:
    return Path(os.environ.get("SKILL_CAVEMAN_ROOT", Path.cwd()))


def _caveman_status() -> str:
    root = _workspace_root()
    if not root.exists():
        return json.dumps({"status": "missing_root", "root": str(root)}, sort_keys=True)

    file_count = 0
    supported_file_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        file_count += 1
        if path.suffix.lower() in ALLOWED_CAVEMAN_FILE_EXTENSIONS:
            supported_file_count += 1

    return json.dumps(
        {
            "status": "ok",
            "root": str(root),
            "file_count": file_count,
            "supported_file_count": supported_file_count,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _caveman_search(query: str, max_lines: int = 120) -> str:
    query = query.strip()
    if not query:
        raise RuntimeError("search requires a non-empty query")

    root = _workspace_root()
    command = ["rg", "--line-number", "--hidden", "--glob", "!.git", query, str(root)]
    if shutil.which("rg") is None:
        command = ["grep", "-R", "-n", "--exclude-dir", ".git", query, str(root)]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=12,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return f"no matches for {query!r} in {root}"
    output = lines[:max_lines]
    if len(lines) > max_lines:
        output.append(f"... truncated; showed {max_lines}/{len(lines)} lines")
    return "\n".join(output)


def _caveman_read(path: str) -> str:
    root = _workspace_root()
    target = (root / path).resolve()
    try:
        if not target.exists():
            return f"not found: {path}"
        if not target.is_file():
            return f"not a file: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"read failed: {exc}"
    lines = text.splitlines()
    return "\n".join(f"{idx + 1:04d}: {line}" for idx, line in enumerate(lines[:220]))


def run_caveman_command(parts: list[str]) -> str:
    if not parts:
        return (
            "Usage:\n"
            "  skill caveman status\n"
            "  skill caveman search QUERY...\n"
            "  skill caveman read FILE\n"
        )

    subcommand, args = parts[0], parts[1:]
    emit("caveman_call", subcommand=subcommand, args=args)

    if subcommand == "status":
        return _caveman_status()
    if subcommand == "search":
        return _caveman_search(" ".join(args))
    if subcommand == "read":
        if not args:
            raise RuntimeError("read requires a file argument")
        return _caveman_read(args[0])
    raise RuntimeError(f"unknown caveman subcommand: {subcommand}")


def servers() -> list[dict[str, Any]]:
    raw = os.environ.get("SKILL_MCP_SERVERS_JSON", "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
    return parsed if isinstance(parsed, list) else []


class StdioClient:
    def __init__(self, command: list[str], cwd: Path, timeout: float) -> None:
        self.timeout = timeout
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
                emit("mcp_stdout_non_json", line=truncate(line, 1000))
                continue
            msg_id = message.get("id")
            if msg_id is None:
                continue
            pending = self._pending.get(int(msg_id))
            if pending is not None:
                pending.put(message)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            emit("mcp_stderr", line=truncate(line.rstrip(), 1000))

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
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
            response = responses.get(timeout=self.timeout)
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
                "clientInfo": {"name": "deepswe-skill", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized")


def rpc_http(url: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=float(os.environ.get("SKILL_TIMEOUT_SEC", "120"))) as response:
        parsed = json.loads(response.read())
    if "error" in parsed:
        raise RuntimeError(json.dumps(parsed["error"], ensure_ascii=True))
    return parsed.get("result") or {}


def server_by_name(name: str | None) -> dict[str, Any]:
    available = servers()
    if not available:
        raise RuntimeError("no MCP servers configured")
    if name:
        for server in available:
            if server.get("name") == name:
                return server
        raise RuntimeError(f"unknown MCP server: {name}")
    return available[0]


def call_mcp(server: dict[str, Any], tool: str, arguments: dict[str, Any]) -> str:
    started = time.time()
    ok = False
    size = 0
    try:
        if server.get("transport") == "stdio":
            command = [str(server.get("command") or ""), *(server.get("args") or [])]
            client = StdioClient(command, Path.cwd(), float(os.environ.get("SKILL_TIMEOUT_SEC", "120")))
            try:
                client.initialize()
                result = client.request("tools/call", {"name": tool, "arguments": arguments})
            finally:
                client.close()
        else:
            result = rpc_http(str(server.get("url")), "tools/call", {"name": tool, "arguments": arguments})
        parts: list[str] = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and ("text" in item or item.get("type") == "text"):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        text = "\n".join(parts) if parts else json.dumps(result, ensure_ascii=True)
        text = truncate(text, int(os.environ.get("SKILL_RESULT_LIMIT", "12000")))
        size = len(text)
        ok = True
        return text
    finally:
        emit(
            "mcp_call",
            server=server.get("name"),
            tool=tool,
            ok=ok,
            duration_sec=round(time.time() - started, 3),
            result_chars=size,
        )


def list_tools(server: dict[str, Any]) -> dict[str, Any]:
    if server.get("transport") == "stdio":
        command = [str(server.get("command") or ""), *(server.get("args") or [])]
        client = StdioClient(command, Path.cwd(), float(os.environ.get("SKILL_TIMEOUT_SEC", "120")))
        try:
            client.initialize()
            return client.request("tools/list")
        finally:
            client.close()
    return rpc_http(str(server.get("url")), "tools/list")


def shortcut_arguments(command: str, rest: list[str]) -> tuple[str, dict[str, Any]]:
    if command in {"status", "overview"}:
        return command, {}
    if command == "search":
        return "search", {"query": " ".join(rest)}
    if command == "context":
        return "context", {"target": " ".join(rest)} if rest else {}
    if command in {"lookup", "refs", "deps", "impact"}:
        return command, {"symbol": " ".join(rest)}
    raise RuntimeError(f"unknown shortcut: {command}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Call local MCP-backed repository skills from the shell."
    )
    parser.add_argument("--server", help="MCP server name. Defaults to first configured server.")
    parser.add_argument("command", nargs="?", default="help")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "help":
        print(
            "Usage:\n"
            "  skill servers\n"
            "  skill --server NAME list\n"
            "  skill --server NAME call TOOL '{\"arg\":\"value\"}'\n"
            "  skill status|overview|context|search|lookup|refs|deps|impact ...\n"
            "  skill caveman status|search QUERY...|read FILE\n"
        )
        return 0
    if args.command == "servers":
        print(json.dumps(servers(), indent=2))
        return 0
    if args.command == "caveman":
        try:
            print(run_caveman_command(args.args))
            return 0
        except Exception as exc:
            print(f"caveman failed: {exc}", file=sys.stderr)
            return 1

    server = server_by_name(args.server)
    if args.command == "list":
        print(json.dumps(list_tools(server), indent=2))
        return 0
    if args.command == "call":
        if not args.args:
            raise RuntimeError("skill call requires a tool name")
        tool = args.args[0]
        raw = args.args[1] if len(args.args) > 1 else "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("tool arguments must be a JSON object")
        print(call_mcp(server, tool, parsed))
        return 0

    tool, tool_args = shortcut_arguments(args.command, args.args)
    print(call_mcp(server, tool, tool_args))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        emit("skill_error", error=f"{type(exc).__name__}: {exc}")
        print(f"skill: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
