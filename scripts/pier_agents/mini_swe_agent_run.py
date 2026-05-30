import shlex
import uuid

from pier.agents.installed.base import with_prompt_template
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.agents.utils import get_api_key_var_names_from_model_name
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.trial.paths import EnvironmentPaths


class MiniSweAgentRun(MiniSweAgent):
    """Pier mini-swe-agent adapter without the upstream --exit-immediately flag."""

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        augmented_instruction = instruction
        if self.mcp_servers:
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    args_str = " ".join(server.args)
                    mcp_info += (
                        f"- {server.name}: stdio transport, "
                        f"command: {server.command} {args_str}\n"
                    )
                else:
                    mcp_info += f"- {server.name}: {server.transport} transport, url: {server.url}\n"
            augmented_instruction = instruction + mcp_info

        run_model_name = self._run_model_name
        if not run_model_name or "/" not in run_model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        env = self.build_process_env(
            {
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
            }
        )

        if self._get_env("MSWEA_API_KEY"):
            env["MSWEA_API_KEY"] = self._get_env("MSWEA_API_KEY") or ""
        else:
            try:
                for api_key_var in get_api_key_var_names_from_model_name(self.model_name):
                    if self._get_env(api_key_var):
                        env[api_key_var] = self._get_env(api_key_var) or ""
                    else:
                        raise ValueError(
                            f"Unset API variable for model {self.model_name}. "
                            f"Please set {api_key_var} or MSWEA_API_KEY environment variable"
                        )
            except ValueError as exc:
                raise ValueError(
                    f"Unable to determine API key for model {self.model_name}: {exc}. "
                    "Please set MSWEA_API_KEY environment variable as fallback"
                ) from exc

        if self._get_env("OPENAI_API_BASE"):
            env["OPENAI_API_BASE"] = self._get_env("OPENAI_API_BASE") or ""
        if self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = self._get_env("OPENAI_BASE_URL") or ""

        custom_config_path = None
        if self._config_yaml:
            custom_config_path = "/tmp/mswea-config/custom.yaml"
            heredoc_marker = f"MSWEA_CONFIG_EOF_{uuid.uuid4().hex[:8]}"
            write_config_cmd = (
                "mkdir -p /tmp/mswea-config\n"
                f"cat > '{custom_config_path}' << '{heredoc_marker}'\n"
                f"{self._config_yaml}\n"
                f"{heredoc_marker}\n"
            )
            await self.exec_as_agent(environment, command=write_config_cmd, env=env)

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        config_flags = self._build_config_flags(custom_config_path=custom_config_path)
        task = shlex.quote(augmented_instruction)

        await self.exec_as_agent(
            environment,
            command=(
                '. "$HOME/.local/bin/env"; '
                f"mini-swe-agent --yolo --model={run_model_name} --task={task} "
                f"--output={EnvironmentPaths.agent_dir / 'mini-swe-agent.trajectory.json'} "
                f"{extra_flags}{config_flags}"
                "2>&1 </dev/null | tee /logs/agent/mini-swe-agent.txt"
            ),
            env=env,
        )
