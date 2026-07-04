"""Codex Pier adapter used for MCP-enabled DeepSWE runs.

The installed Pier Codex agent owns the real Codex invocation and MCP
registration. This subclass exists as a stable import path for harness runs and
adds a small diagnostic breadcrumb in trial logs when Pier exposes a log helper.
"""

from __future__ import annotations

import json
import os
import shlex

try:
    from pier.agents.installed.codex import Codex
    from pier.models.agent.install import InstallStep
except Exception as exc:  # pragma: no cover - Pier is only installed in runner.
    Codex = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if Codex is None:  # pragma: no cover

    class CodexMcpRun:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"Pier Codex agent is unavailable: {_IMPORT_ERROR}")

else:

    class CodexMcpRun(Codex):  # type: ignore[misc,no-redef]
        """Stable shim for selecting Codex with task-provided MCP servers."""

        def _uses_sverklo_stdio(self) -> bool:
            for server in getattr(self, "mcp_servers", None) or []:
                if getattr(server, "name", "") == "sverklo":
                    return True
                command = getattr(server, "command", "") or ""
                args = getattr(server, "args", []) or []
                if "sverklo" in " ".join([command, *args]):
                    return True
            return False

        def install_spec(self):  # noqa: ANN201
            spec = super().install_spec()
            if not self._uses_sverklo_stdio():
                return spec

            version = os.environ.get("SVERKLO_VERSION", "latest")
            install_sverklo = (
                "set -euo pipefail; "
                'export NVM_DIR="$HOME/.nvm"; '
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
                "command -v nvm >/dev/null || "
                "{ echo 'nvm is required for Sverklo install' >&2; exit 1; }; "
                "nvm install 24; "
                'NODE24="$(find "$NVM_DIR/versions/node" -maxdepth 1 -type d '
                "-name 'v24*' | sort -V | tail -1)\"; "
                'PATH="$NODE24/bin:$PATH"; '
                f"npm install -g sverklo@{shlex.quote(version)}; "
                "sverklo setup; "
                "sverklo --help >/dev/null"
            )
            spec.steps.append(InstallStep(user="agent", run=install_sverklo))
            return spec

        def _build_register_mcp_servers_command(self) -> str | None:
            """Write Codex MCP config, including stdio args arrays."""
            if not self.mcp_servers:
                return None
            lines: list[str] = []
            for server in self.mcp_servers:
                lines.append(f"[mcp_servers.{server.name}]")
                if server.transport == "stdio":
                    if not server.command:
                        continue
                    lines.append(f"command = {json.dumps(server.command)}")
                    if server.args:
                        lines.append(f"args = {json.dumps(server.args)}")
                else:
                    lines.append(f"url = {json.dumps(server.url)}")
                if server.name == "sverklo":
                    raw_tools = os.environ.get("SVERKLO_ENABLED_TOOLS")
                    if raw_tools is not None:
                        enabled_tools = [
                            tool.strip()
                            for tool in raw_tools.split(",")
                            if tool.strip()
                        ]
                        if enabled_tools:
                            lines.append(f"enabled_tools = {json.dumps(enabled_tools)}")
                    lines.append(
                        "startup_timeout_sec = "
                        f"{float(os.environ.get('SVERKLO_STARTUP_TIMEOUT_SEC', '60'))}"
                    )
                    lines.append(
                        "tool_timeout_sec = "
                        f"{float(os.environ.get('SVERKLO_TOOL_TIMEOUT_SEC', '120'))}"
                    )
                lines.append("")
            escaped_config = shlex.quote("\n".join(lines))
            return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'

        async def run(self, instruction, environment, context):  # noqa: ANN001
            servers = getattr(self, "mcp_servers", None) or []
            if servers:
                try:
                    names = ",".join(getattr(server, "name", "") for server in servers)
                    await self.exec_as_agent(
                        environment,
                        command=(
                            "mkdir -p /logs/agent && "
                            f"printf '%s\\n' 'mcp_servers={names}' "
                            ">> /logs/agent/codex-mcp.diagnostics.txt"
                        ),
                    )
                except Exception:
                    pass
            return await super().run(instruction, environment, context)
