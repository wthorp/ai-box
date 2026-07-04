#!/usr/bin/env python3
"""HTTP MCP bridge for Sverklo's stdio MCP server.

Sverklo currently exposes MCP over stdio. Pier's staged DeepSWE harness uses
HTTP MCP URLs, so this bridge keeps one Sverklo subprocess alive and relays
JSON-RPC requests between HTTP POSTs and the subprocess' line-delimited stdio.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class SverkloProcess:
    def __init__(self, command: list[str], cwd: Path, timeout_sec: float) -> None:
        self.command = command
        self.cwd = cwd
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._next_id = 1
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def alive(self) -> bool:
        return self.process.poll() is None

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(f"[sverklo-proxy] non-json stdout: {line}", file=sys.stderr, flush=True)
                continue
            msg_id = message.get("id")
            if msg_id is None:
                print(
                    f"[sverklo-proxy] notification from server: {message}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pending = self._pending.get(msg_id)
            if pending is None:
                print(
                    f"[sverklo-proxy] unexpected response id={msg_id}: {message}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pending.put(message)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            print(f"[sverklo] {line.rstrip()}", file=sys.stderr, flush=True)

    def notify(self, message: dict[str, Any]) -> None:
        if not self.alive():
            raise RuntimeError(f"Sverklo exited with code {self.process.returncode}")
        assert self.process.stdin is not None
        with self._lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if not self.alive():
            raise RuntimeError(f"Sverklo exited with code {self.process.returncode}")
        assert self.process.stdin is not None
        original_id = message.get("id", 1)
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            outbound = dict(message)
            outbound["id"] = request_id
            responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = responses
            self.process.stdin.write(json.dumps(outbound) + "\n")
            self.process.stdin.flush()
        try:
            response = responses.get(timeout=self.timeout_sec)
        finally:
            self._pending.pop(request_id, None)
        response["id"] = original_id
        return response


class Handler(http.server.BaseHTTPRequestHandler):
    bridge: SverkloProcess

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[sverklo-proxy] {fmt % args}", file=sys.stderr, flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            status = 200 if self.bridge.alive() else 503
            self._json(status, {"ok": self.bridge.alive()})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("content-length", "0"))
        try:
            message = json.loads(self.rfile.read(length))
            if "id" not in message:
                self.bridge.notify(message)
                self.send_response(202)
                self.end_headers()
                return
            response = self.bridge.request(message)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(exc)},
            }
            self._json(500, response)
            return
        self._json(200, response)


def build_command(project_path: Path) -> list[str]:
    if args := os.environ.get("SVERKLO_ARGS"):
        return shlex.split(os.environ.get("SVERKLO_BIN", "sverklo")) + shlex.split(args)
    return shlex.split(os.environ.get("SVERKLO_BIN", "sverklo")) + [str(project_path)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SVERKLO_HTTP_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", default=os.environ.get("SVERKLO_HTTP_PORT", "3007"), type=int
    )
    parser.add_argument(
        "--project-path", default=os.environ.get("SVERKLO_PROJECT_PATH", "/workspace")
    )
    parser.add_argument(
        "--timeout-sec",
        default=float(os.environ.get("SVERKLO_PROXY_TIMEOUT_SEC", "120")),
        type=float,
    )
    args = parser.parse_args(argv)

    project_path = Path(args.project_path).resolve()
    if not project_path.is_dir():
        print(f"SVERKLO_PROJECT_PATH is not a directory: {project_path}", file=sys.stderr)
        return 2

    command = build_command(project_path)
    print(
        f"sverklo proxy listening on {args.host}:{args.port}; command={command}",
        file=sys.stderr,
        flush=True,
    )
    Handler.bridge = SverkloProcess(command, project_path, args.timeout_sec)
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
