import base64
import json
import os
import shlex
import uuid
from pathlib import Path

from pier.agents.installed.base import with_prompt_template
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.agents.utils import get_api_key_var_names_from_model_name
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import InstallStep
from pier.models.trial.paths import EnvironmentPaths


class MiniSweAgentRun(MiniSweAgent):
    """Pier mini-swe-agent adapter with explicit local endpoint propagation."""

    def __init__(self, step_limit: int | str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_limit = int(step_limit) if step_limit not in (None, "") else None

    def _build_config_flags(self, *, custom_config_path: str | None = None) -> str:
        config_flags = super()._build_config_flags(
            custom_config_path=custom_config_path
        )
        if self._step_limit is not None:
            config_flags += f"-c agent.step_limit={self._step_limit} "
        return config_flags

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
        root_run = (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "--no-install-recommends ca-certificates curl git bash xz-utils; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache ca-certificates curl git bash xz; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y ca-certificates curl git bash xz; "
            "fi"
        )
        install_sverklo = (
            "set -euo pipefail; "
            'export NVM_DIR="$HOME/.nvm"; '
            'if [ ! -s "$NVM_DIR/nvm.sh" ]; then '
            "curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash; "
            "fi; "
            '. "$NVM_DIR/nvm.sh"; '
            "nvm install 24; "
            'NODE24="$(find "$NVM_DIR/versions/node" -maxdepth 1 -type d '
            "-name 'v24*' | sort -V | tail -1)\"; "
            'PATH="$NODE24/bin:$PATH"; '
            f"npm install -g sverklo@{shlex.quote(version)}; "
            "sverklo setup; "
            "sverklo --help >/dev/null"
        )
        spec.steps.append(InstallStep(user="root", run=root_run))
        spec.steps.append(InstallStep(user="agent", run=install_sverklo))
        return spec

    def _mcp_servers_json(self) -> str:
        servers: list[dict[str, object]] = []
        for server in getattr(self, "mcp_servers", None) or []:
            data: dict[str, object] = {
                "name": getattr(server, "name", ""),
                "transport": getattr(server, "transport", ""),
            }
            for key in ("url", "command", "args"):
                value = getattr(server, key, None)
                if value:
                    data[key] = value
            servers.append(data)
        return json.dumps(servers)

    async def _install_skill_cli(self, environment: BaseEnvironment) -> None:
        skill_path = Path(__file__).resolve().parents[1] / "skill_mcp.py"
        skill_b64 = base64.b64encode(skill_path.read_bytes()).decode("ascii")
        command = (
            "set -euo pipefail\n"
            "mkdir -p \"$HOME/.local/bin\" /logs/telemetry\n"
            "python3 - <<'PY'\n"
            "import base64, pathlib\n"
            "path = pathlib.Path.home() / '.local/bin/skill'\n"
            f"path.write_bytes(base64.b64decode({skill_b64!r}))\n"
            "path.chmod(0o755)\n"
            "PY\n"
            "python3 - <<'PY'\n"
            "import json, os, pathlib, time\n"
            "path = pathlib.Path('/logs/telemetry/events.jsonl')\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "event = {\n"
            "  'ts': time.time(),\n"
            "  'event': 'trial_start',\n"
            f"  'mcp_servers': json.loads({self._mcp_servers_json()!r}),\n"
            "  'agent': 'mini-swe-agent',\n"
            f"  'model': {json.dumps(self.model_name or '')},\n"
            "}\n"
            "with path.open('a', encoding='utf-8') as out:\n"
            "    out.write(json.dumps(event, ensure_ascii=True) + '\\n')\n"
            "PY"
        )
        await self.exec_as_agent(environment, command=command)

    async def _finalize_telemetry(
        self, environment: BaseEnvironment, trajectory: str
    ) -> None:
        script = f"""python3 - <<'PY'
import json, os, pathlib, subprocess, time

telemetry = pathlib.Path('/logs/telemetry')
telemetry.mkdir(parents=True, exist_ok=True)
events_path = telemetry / 'events.jsonl'
summary_path = telemetry / 'summary.json'

def emit(event, **fields):
    with events_path.open('a', encoding='utf-8') as out:
        out.write(json.dumps({{'ts': time.time(), 'event': event, **fields}}, ensure_ascii=True) + '\\n')

def run_git(args):
    try:
        return subprocess.run(
            ['git', *args],
            cwd='/app',
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        ).stdout
    except Exception:
        return ''

changed = [line.strip() for line in run_git(['diff', '--name-only']).splitlines() if line.strip()]
numstat = run_git(['diff', '--numstat']).splitlines()
added = deleted = 0
for line in numstat:
    parts = line.split('\\t')
    if len(parts) >= 2:
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            deleted += int(parts[1])

events = []
if events_path.exists():
    for raw in events_path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            events.append(json.loads(raw))
        except Exception:
            pass

mcp_calls = [event for event in events if event.get('event') == 'mcp_call']
errors = [event for event in events if event.get('event') in {{'skill_error', 'mcp_stderr'}}]
start_ts = next((event.get('ts') for event in events if event.get('event') == 'trial_start'), None)
first_mcp = next((event.get('ts') for event in mcp_calls), None)
now = time.time()

trajectory_path = pathlib.Path({trajectory!r})
trajectory_size = trajectory_path.stat().st_size if trajectory_path.exists() else 0
agent_exit_code = None
diag_path = pathlib.Path('/logs/agent/mini-swe-agent.diagnostics.txt')
if diag_path.exists():
    for line in diag_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('exit_code='):
            try:
                agent_exit_code = int(line.split('=', 1)[1].strip())
            except Exception:
                agent_exit_code = line.split('=', 1)[1].strip()
summary = {{
    'agent': 'mini-swe-agent',
    'agent_exit_code': agent_exit_code,
    'elapsed_sec': round(now - start_ts, 3) if isinstance(start_ts, (int, float)) else None,
    'mcp_call_count': len(mcp_calls),
    'mcp_error_count': len([event for event in mcp_calls if not event.get('ok')]),
    'skill_error_count': len(errors),
    'first_mcp_sec': round(first_mcp - start_ts, 3) if isinstance(start_ts, (int, float)) and isinstance(first_mcp, (int, float)) else None,
    'changed_file_count': len(changed),
    'changed_files': changed[:200],
    'diff_added_lines': added,
    'diff_deleted_lines': deleted,
    'trajectory_exists': trajectory_path.exists(),
    'trajectory_size_bytes': trajectory_size,
}}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + '\\n', encoding='utf-8')
emit('agent_end', **summary)
PY"""
        await self.exec_as_agent(environment, command=script, timeout_sec=60)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self._install_skill_cli(environment)

        repair_discipline = """

Repair discipline:
- Make the smallest targeted source edits needed for the tests. Do not rewrite whole source files.
- Do not use `cat > existing_file` or here-doc rewrites for existing source files longer than 100 lines.
- Prefer inspecting the failing tests first, then use focused search and line-range reads.
- Keep source text ASCII-only unless the file already uses non-ASCII for the exact feature being changed.
- After edits, run the relevant formatter or syntax checker for the language, then run `git diff --check`.
- Before finishing, inspect `git diff --stat` and the changed hunks to make sure only intended files changed.
"""

        augmented_instruction = instruction + repair_discipline
        if self.mcp_servers:
            mcp_info = "\n\nAdditional local repository skills are available via `skill --help`.\n"
            mcp_info += "Use them only when they help inspect this repository faster.\n"
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    args_str = " ".join(server.args)
                    mcp_info += (
                        f"- {server.name}: available through the `skill` command "
                        f"(backed by: {server.command} {args_str})\n"
                    )
                else:
                    mcp_info += f"- {server.name}: available through the `skill` command\n"
            augmented_instruction = instruction + mcp_info

        run_model_name = self._run_model_name
        if not run_model_name or "/" not in run_model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        env = self.build_process_env(
            {
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
                "SKILL_MCP_SERVERS_JSON": self._mcp_servers_json(),
                "SKILL_TELEMETRY_DIR": "/logs/telemetry",
                "SVERKLO_PROJECT_PATH": "/app",
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
        output_path = str(EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json")
        output_path_quoted = shlex.quote(output_path)
        command = (
            f"mini-swe-agent --yolo --model={shlex.quote(run_model_name)} --task={task} "
            f"--output={output_path_quoted} "
            f"{extra_flags}{config_flags}"
            "--exit-immediately"
        )
        command_b64 = base64.b64encode(command.encode()).decode()
        diagnostic_script = f"""set -euo pipefail
. "$HOME/.local/bin/env"
diagnostics=/logs/agent/mini-swe-agent.diagnostics.txt
agent_log=/logs/agent/mini-swe-agent.txt
trajectory={output_path_quoted}
{{
  echo "start_at=$(date -Is)"
  echo "cwd=$(pwd)"
  echo "user=$(id)"
  echo "logs_agent_ls=$(ls -ld /logs /logs/agent 2>&1)"
  echo "env_snapshot_begin"
  env | sort | awk -F= '/^(AI_BOX|HOME|LITELLM|MSWEA|OPENAI|PATH|PIER|PWD|USER)=/ {{
    if ($1 ~ /(KEY|TOKEN|SECRET|PASSWORD)/) {{
      print $1 "=<redacted>"
    }} else {{
      print
    }}
  }}'
  echo "env_snapshot_end"
  echo "command_b64={command_b64}"
}} >> "$diagnostics"
set +e
export PATH="$HOME/.local/bin:$PATH"
{command} 2>&1 </dev/null | tee "$agent_log"
status=${{PIPESTATUS[0]}}
set -e
{{
  echo "end_at=$(date -Is)"
  echo "exit_code=$status"
  if [ -f "$trajectory" ]; then
    echo "trajectory_exists=1"
    ls -l "$trajectory"
  else
    echo "trajectory_exists=0"
  fi
}} >> "$diagnostics"
exit "$status"
"""

        try:
            await self.exec_as_agent(
                environment,
                command=f"bash -lc {shlex.quote(diagnostic_script)}",
                env=env,
            )
        finally:
            await self._finalize_telemetry(
                environment, trajectory=str(EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json")
            )
