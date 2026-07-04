"""Minimal Pier agent used for harness and verifier smoke tests."""

from __future__ import annotations

from pier.agents.installed.base import BaseInstalledAgent
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep


class NoopRun(BaseInstalledAgent):
    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "noop"

    def populate_context_post_run(self, context: AgentContext) -> None:
        return None

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[InstallStep(user="agent", run="true")],
        )

    async def run(self, instruction, environment, context):  # noqa: ANN001
        await self.exec_as_agent(
            environment,
            command=(
                "mkdir -p /logs/agent; "
                "printf 'noop agent made no repository changes\\n' "
                "> /logs/agent/noop.txt"
            ),
            cwd="/app",
            timeout_sec=60,
        )
