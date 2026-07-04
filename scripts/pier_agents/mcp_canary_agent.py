"""Deterministic Pier agent for validating MCP tool delivery."""

from __future__ import annotations

import json
import shlex
import urllib.request
from pathlib import Path
from typing import Any

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext


class McpCanaryAgent(BaseAgent):
    """Call the first configured MCP canary server and record the response."""

    @staticmethod
    def name() -> str:
        return "mcp-canary-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _rpc(
        self, url: str, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            payload["params"] = params
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.mcp_servers:
            raise RuntimeError("Pier did not provide any MCP servers to the agent")

        server = self.mcp_servers[0]
        if not server.url:
            raise RuntimeError(f"MCP server {server.name!r} has no URL")

        self._rpc(server.url, "initialize", {"clientInfo": {"name": self.name()}})
        self._rpc(server.url, "tools/list")
        call = self._rpc(
            server.url,
            "tools/call",
            {
                "name": "canary_nonce",
                "arguments": {"job_id": "pier-canary-validation"},
            },
        )

        text = call["result"]["content"][0]["text"]
        output = Path(self.logs_dir) / "mcp-canary-agent-result.json"
        output.write_text(text + "\n", encoding="utf-8")
        await environment.exec(
            command=(
                "printf '%s\n' "
                f"{shlex.quote(text)} "
                "> /workspace/canary-result.txt"
            )
        )
