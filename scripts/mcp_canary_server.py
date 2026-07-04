#!/usr/bin/env python3
"""Tiny Streamable HTTP MCP canary server.

It implements just enough JSON-RPC MCP for Pier/Codex smoke tests:
initialize, tools/list, and tools/call for a nonce-returning canary tool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class CanaryState:
    def __init__(self, nonce: str, log_path: Path) -> None:
        self.nonce = nonce
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        record = {"ts": utc_now(), **payload}
        with self.log_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")


class CanaryHandler(BaseHTTPRequestHandler):
    server_version = "mcp-canary/0.1"

    @property
    def state(self) -> CanaryState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._send_json({"ok": True, "nonce": self.state.nonce})
            return
        if self.path == "/logs":
            body = ""
            if self.state.log_path.exists():
                body = self.state.log_path.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/jsonl")
            self.send_header("content-length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path not in {"/", "/mcp"}:
            self.send_error(404)
            return
        length = int(self.headers.get("content-length") or "0")
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(self._error(None, -32700, f"parse error: {exc}"), 400)
            return

        response = self._handle_rpc(request)
        if response is None:
            self.send_response(202)
            self.end_headers()
            return
        self._send_json(response)

    def _handle_rpc(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        self.state.log({"event": "rpc", "method": method, "id": request_id})

        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mcp-canary", "version": "0.1.0"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._result(
                request_id,
                {
                    "tools": [
                        {
                            "name": "canary_nonce",
                            "description": "Return the configured canary nonce.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "job_id": {
                                        "type": "string",
                                        "description": "DeepSWE/Pier job identifier.",
                                    }
                                },
                                "additionalProperties": True,
                            },
                        }
                    ]
                },
            )
        if method == "tools/call":
            params = request.get("params") or {}
            if params.get("name") != "canary_nonce":
                return self._error(request_id, -32602, "unknown tool")
            arguments = params.get("arguments") or {}
            self.state.log(
                {
                    "event": "tool_call",
                    "tool": "canary_nonce",
                    "job_id": arguments.get("job_id"),
                    "arguments": arguments,
                }
            )
            return self._result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "nonce": self.state.nonce,
                                    "job_id": arguments.get("job_id"),
                                },
                                sort_keys=True,
                            ),
                        }
                    ],
                    "isError": False,
                },
            )
        return self._error(request_id, -32601, f"unknown method: {method}")

    def _result(self, request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(
        self, request_id: Any, code: int, message: str, status: int | None = None
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message, "status": status},
        }

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        self.state.log({"event": "access", "message": fmt % args})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default=os.environ.get("MCP_CANARY_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", default=os.environ.get("MCP_CANARY_PORT", "3005"), type=int
    )
    parser.add_argument(
        "--nonce", default=os.environ.get("MCP_CANARY_NONCE", "mcp-canary")
    )
    parser.add_argument(
        "--log-path",
        default=os.environ.get("MCP_CANARY_LOG", "/eval-results/mcp-canary.jsonl"),
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CanaryHandler)
    server.state = CanaryState(  # type: ignore[attr-defined]
        args.nonce, Path(args.log_path)
    )
    print(f"mcp-canary listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
