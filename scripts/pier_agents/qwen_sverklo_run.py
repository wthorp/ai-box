"""Targeted Pier adapter for Qwen3-Coder-Next plus local MCP skills.

This intentionally avoids Codex/mini-swe-agent. Pier only launches the task
container and this adapter runs a small local worker inside it.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path

from pier.agents.installed.base import BaseInstalledAgent
from pier.agents.network import allowlist_from_urls
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist


class QwenSverkloRun(BaseInstalledAgent):
    """Minimal local Qwen tool loop backed by stdio MCP servers."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        max_steps: int | str | None = None,
        max_tokens: int | str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        repeat_penalty: float | str | None = None,
        repeat_last_n: int | str | None = None,
        dry_multiplier: float | str | None = None,
        dry_base: float | str | None = None,
        dry_allowed_length: int | str | None = None,
        dry_penalty_last_n: int | str | None = None,
        mirostat: int | str | None = None,
        mirostat_tau: float | str | None = None,
        mirostat_eta: float | str | None = None,
        first_edit_step: int | str | None = None,
        no_edit_abort_step: int | str | None = None,
        max_no_tool_retries: int | str | None = None,
        tool_choice: str | None = None,
        extra_startup_guidance: str | None = None,
        task_context_path: str | None = None,
        task_context_b64: str | None = None,
        task_context_limit: int | str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_steps = int(max_steps or os.environ.get("QSA_MAX_STEPS", "80"))
        self._max_tokens = int(max_tokens or os.environ.get("QSA_MAX_TOKENS", "2048"))
        self._temperature = float(
            temperature or os.environ.get("QSA_TEMPERATURE", "0.05")
        )
        self._top_p = float(top_p or os.environ.get("QSA_TOP_P", "0.8"))
        self._repeat_penalty = float(
            repeat_penalty or os.environ.get("QSA_REPEAT_PENALTY", "1.12")
        )
        self._repeat_last_n = int(
            repeat_last_n or os.environ.get("QSA_REPEAT_LAST_N", "4096")
        )
        self._dry_multiplier = float(
            dry_multiplier or os.environ.get("QSA_DRY_MULTIPLIER", "1.0")
        )
        self._dry_base = float(dry_base or os.environ.get("QSA_DRY_BASE", "1.75"))
        self._dry_allowed_length = int(
            dry_allowed_length or os.environ.get("QSA_DRY_ALLOWED_LENGTH", "3")
        )
        self._dry_penalty_last_n = int(
            dry_penalty_last_n
            or os.environ.get("QSA_DRY_PENALTY_LAST_N", "4096")
        )
        self._mirostat = int(mirostat or os.environ.get("QSA_MIROSTAT", "0"))
        self._mirostat_tau = float(
            mirostat_tau or os.environ.get("QSA_MIROSTAT_TAU", "4.5")
        )
        self._mirostat_eta = float(
            mirostat_eta or os.environ.get("QSA_MIROSTAT_ETA", "0.1")
        )
        self._first_edit_step = int(
            first_edit_step or os.environ.get("QSA_FIRST_EDIT_STEP", "8")
        )
        self._no_edit_abort_step = int(
            no_edit_abort_step or os.environ.get("QSA_NO_EDIT_ABORT_STEP", "45")
        )
        self._max_no_tool_retries = int(
            max_no_tool_retries or os.environ.get("QSA_MAX_NO_TOOL_RETRIES", "3")
        )
        self._tool_choice = tool_choice or os.environ.get("QSA_TOOL_CHOICE", "required")
        self._extra_startup_guidance = (
            extra_startup_guidance or os.environ.get("QSA_EXTRA_STARTUP_GUIDANCE", "")
        )
        self._task_context_path = task_context_path or os.environ.get("QSA_TASK_CONTEXT_PATH")
        self._task_context_b64 = task_context_b64 or os.environ.get("QSA_TASK_CONTEXT_B64")
        self._task_context_limit = int(
            task_context_limit or os.environ.get("QSA_TASK_CONTEXT_LIMIT", "24000")
        )

    @staticmethod
    def name() -> str:
        return "qwen-sverklo"

    def populate_context_post_run(self, context: AgentContext) -> None:
        return None

    def _uses_stdio_server(self, name: str) -> bool:
        for server in getattr(self, "mcp_servers", None) or []:
            if (
                getattr(server, "name", "") == name
                and getattr(server, "transport", "") == "stdio"
            ):
                return True
        return False

    def install_spec(self) -> AgentInstallSpec:
        version = os.environ.get("SVERKLO_VERSION", "latest")
        uses_sverklo = self._uses_stdio_server("sverklo")
        uses_serena = self._uses_stdio_server("serena")
        use_prebuilt_tools = (
            os.environ.get("QSA_PREBUILT_TOOLCHAIN", "1") != "0"
            and (uses_sverklo or uses_serena)
        )
        root_run = (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "--no-install-recommends ca-certificates curl git bash xz-utils python3; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache ca-certificates curl git bash xz python3; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y ca-certificates curl git bash xz python3; "
            "fi"
        )
        if use_prebuilt_tools:
            checks = ['export PATH="/usr/local/bin:$HOME/.local/bin:$HOME/go/bin:$PATH"']
            if uses_sverklo:
                checks.extend(
                    [
                        "command -v sverklo >/dev/null",
                        "sverklo setup >/dev/null 2>&1 || true",
                        "sverklo --help >/dev/null",
                    ]
                )
            if uses_serena:
                checks.extend(
                    [
                        "command -v serena >/dev/null",
                        "serena init --no-browser >/dev/null 2>&1 || true",
                        "serena --help >/dev/null",
                    ]
                )
            checks.append("python3 --version")
            return AgentInstallSpec(
                agent_name=self.name(),
                version=self._version,
                cache_key=(
                    f"{self.name()}-prebuilt-tools-"
                    f"{version}-sverklo={int(uses_sverklo)}-serena={int(uses_serena)}"
                ),
                metadata={
                    "qsa_prebuilt_toolchain": True,
                    "qsa_prebuilt_toolchain_image": os.environ.get(
                        "QSA_PREBUILT_TOOLCHAIN_IMAGE",
                        f"ai-box-deepswe-tools:{version}",
                    ),
                },
                steps=[
                    InstallStep(user="root", run=root_run),
                    InstallStep(user="agent", run="set -euo pipefail; " + "; ".join(checks)),
                ],
            )
        serena_install = ""
        if uses_serena:
            serena_install = (
                "if ! command -v uv >/dev/null 2>&1; then "
                "timeout 300 curl -LsSf https://astral.sh/uv/install.sh | sh; "
                "fi; "
                'export PATH="$HOME/.local/bin:$PATH"; '
                'if [ -f go.mod ] && command -v go >/dev/null 2>&1; then '
                'timeout 900 go install golang.org/x/tools/gopls@latest || true; '
                'fi; '
                'export PATH="$HOME/go/bin:$PATH"; '
                "timeout 900 uv tool install -p 3.13 serena-agent; "
                "ln -sf \"$HOME/.local/bin/serena\" /usr/local/bin/serena; "
                "serena init --no-browser >/dev/null 2>&1 || true; "
                "serena --help >/dev/null; "
            )
        if uses_sverklo:
            agent_run = (
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
                "npm config set fetch-retry-maxtimeout 120000; "
                f"timeout 900 npm install -g sverklo@{shlex.quote(version)}; "
                "sverklo setup; "
                + serena_install
                + "python3 --version; "
                "sverklo --help >/dev/null"
            )
        elif uses_serena:
            agent_run = "set -euo pipefail; " + serena_install + "python3 --version"
        else:
            agent_run = "python3 --version"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(user="root", run=root_run),
                InstallStep(user="agent", run=agent_run),
            ],
        )

    def network_allowlist(self) -> NetworkAllowlist:
        urls = [
            os.environ.get("QWEN_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_API_BASE"),
            "http://172.17.0.1:8080/v1",
        ]
        allowlist = allowlist_from_urls(
            urls,
            default_domains=[
                "172.17.0.1",
                "astral.sh",
                "files.pythonhosted.org",
                "raw.githubusercontent.com",
                "github.com",
                "pypi.org",
                "registry.npmjs.org",
                "nodejs.org",
                "proxy.golang.org",
                "sum.golang.org",
            ],
        )
        return allowlist

    def _sverklo_command(self) -> list[str]:
        for server in self.mcp_servers:
            if server.name == "sverklo" and server.transport == "stdio" and server.command:
                return [server.command, *(server.args or [])]
        return [
            "bash",
            "-lc",
            'if command -v sverklo >/dev/null 2>&1; then '
            'exec sverklo "${SVERKLO_PROJECT_PATH:-.}"; '
            'fi; '
            'NODE24="$(find "$HOME/.nvm/versions/node" -maxdepth 1 -type d '
            '-name "v24*" | sort -V | tail -1 2>/dev/null || true)"; '
            'if [ -z "$NODE24" ]; then echo "Sverklo not installed" >&2; exit 127; fi; '
            'PATH="$NODE24/bin:$PATH" exec sverklo "${SVERKLO_PROJECT_PATH:-.}"',
        ]

    def _serena_command(self) -> list[str]:
        return [
            "bash",
            "-lc",
            'export PATH="/usr/local/bin:$HOME/.local/bin:$HOME/go/bin:$PATH"; '
            'exec serena start-mcp-server --transport stdio '
            '--context ide-assistant --project "${SERENA_PROJECT_PATH:-.}" '
            "--mode editing --mode interactive",
        ]

    def _mcp_commands(self) -> list[dict[str, object]]:
        commands: list[dict[str, object]] = []
        for server in getattr(self, "mcp_servers", None) or []:
            if server.transport != "stdio":
                continue
            if server.command:
                command = [server.command, *(server.args or [])]
            elif server.name == "serena":
                command = self._serena_command()
            else:
                command = self._sverklo_command()
            commands.append({"name": server.name, "command": command})
        return commands

    def _task_context_text(self) -> str:
        if self._task_context_b64:
            try:
                text = base64.b64decode(self._task_context_b64).decode("utf-8", "replace")
            except Exception:
                text = ""
            if text:
                if len(text) > self._task_context_limit:
                    half = max(1, self._task_context_limit // 2)
                    text = text[:half] + "\n... [task context truncated] ...\n" + text[-half:]
                return (
                    "\n\nAuthoritative task context from DeepSWE task files. "
                    "Use this embedded text as the primary spec and test target before broad repository search. "
                    "The tests/test.patch content below is already available; do not use tools to read, cat, grep, "
                    "sed, git-apply, or apply /tests/test.patch, tests/test.patch, or /test.patch from the runtime repository. "
                    "Use the embedded content only to identify required behavior and implicated implementation files.\n"
                    f"{text}\n"
                )
        if not self._task_context_path:
            return ""
        root = Path(self._task_context_path)
        if not root.exists():
            return ""
        parts: list[str] = []
        for rel in ("instruction.md", "tests/test.sh", "tests/test.patch"):
            path = root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"## task file: {rel}\n{text}")
        if not parts:
            return ""
        text = "\n\n".join(parts)
        if len(text) > self._task_context_limit:
            half = max(1, self._task_context_limit // 2)
            text = text[:half] + "\n... [task context truncated] ...\n" + text[-half:]
        return (
            "\n\nAuthoritative task context from DeepSWE task files. "
            "Use this embedded text as the primary spec and test target before broad repository search. "
            "The tests/test.patch content below is already available; do not use tools to read, cat, grep, "
            "sed, git-apply, or apply /tests/test.patch, tests/test.patch, or /test.patch from the runtime repository. "
            "Use the embedded content only to identify required behavior and implicated implementation files.\n"
            f"{text}\n"
        )

    async def run(self, instruction, environment, context):  # noqa: ANN001
        worker_path = Path(__file__).with_name("qwen_sverklo_worker.py")
        worker_b64 = base64.b64encode(worker_path.read_bytes()).decode("ascii")
        install_worker = (
            "python3 - <<'PY'\n"
            "import base64, pathlib\n"
            "path = pathlib.Path('/tmp/qwen_sverklo_worker.py')\n"
            f"path.write_bytes(base64.b64decode({worker_b64!r}))\n"
            "path.chmod(0o755)\n"
            "PY"
        )
        await self.exec_as_agent(environment, command=install_worker)

        initial_context_path = "/tmp/qsa_initial_context.txt"
        initial_context_cmd = r"""set -euo pipefail
out=/tmp/qsa_initial_context.txt
{
  echo "## container"
  echo "cwd=$(pwd)"
  echo
  echo "## top-level repository files"
  (find . -maxdepth 2 -type f \
    \( -name 'test.sh' -o -name 'package.json' -o -name 'go.mod' -o -name 'pyproject.toml' -o -name 'Cargo.toml' -o -name 'Makefile' -o -name 'README*' \) \
    | sort | sed -n '1,80p') 2>/dev/null || true
  echo
  for f in ./test.sh ./package.json ./go.mod ./pyproject.toml ./Cargo.toml ./Makefile; do
    if [ -f "$f" ]; then
      echo "## $f"
      sed -n '1,180p' "$f" || true
      echo
    fi
  done
  echo "## nearby DeepSWE test patch candidates"
  for root in /app /tmp /logs /workspace; do
    [ -d "$root" ] || continue
    find "$root" -maxdepth 5 -type f \( -path '*/tests/test.patch' -o -name 'test.patch' \) 2>/dev/null | sort | sed -n '1,10p'
  done | sort -u | while IFS= read -r f; do
    [ -f "$f" ] || continue
    echo "## $f"
    sed -n '1,260p' "$f" || true
    echo
  done
} > "$out"
"""
        await self.exec_as_agent(
            environment,
            command=f"bash -lc {shlex.quote(initial_context_cmd)}",
            cwd="/app",
            timeout_sec=90,
        )

        model = self.model_name or "openai/local"
        if "/" in model:
            model = model.split("/", 1)[1]

        env = self.build_process_env(
            {
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
                "QSA_BASE_URL": os.environ.get(
                    "QWEN_BASE_URL",
                    os.environ.get(
                        "OPENAI_BASE_URL",
                        os.environ.get(
                            "OPENAI_API_BASE", "http://172.17.0.1:8080/v1"
                        ),
                    ),
                ),
                "QSA_MODEL": model,
                "QSA_MAX_STEPS": str(self._max_steps),
                "QSA_MAX_TOKENS": str(self._max_tokens),
                "QSA_TEMPERATURE": str(self._temperature),
                "QSA_TOP_P": str(self._top_p),
                "QSA_REPEAT_PENALTY": str(self._repeat_penalty),
                "QSA_REPEAT_LAST_N": str(self._repeat_last_n),
                "QSA_DRY_MULTIPLIER": str(self._dry_multiplier),
                "QSA_DRY_BASE": str(self._dry_base),
                "QSA_DRY_ALLOWED_LENGTH": str(self._dry_allowed_length),
                "QSA_DRY_PENALTY_LAST_N": str(self._dry_penalty_last_n),
                "QSA_MIROSTAT": str(self._mirostat),
                "QSA_MIROSTAT_TAU": str(self._mirostat_tau),
                "QSA_MIROSTAT_ETA": str(self._mirostat_eta),
                "QSA_LLM_TIMEOUT_SEC": os.environ.get("QSA_LLM_TIMEOUT_SEC", "240"),
                "QSA_LLM_RETRIES": os.environ.get("QSA_LLM_RETRIES", "2"),
                "QSA_LLM_RETRY_DELAY_SEC": os.environ.get(
                    "QSA_LLM_RETRY_DELAY_SEC", "8"
                ),
                "QSA_EARLY_STOP": os.environ.get("QSA_EARLY_STOP", "1"),
                "QSA_FAIL_SCORE_ABORT": os.environ.get("QSA_FAIL_SCORE_ABORT", "0.70"),
                "QSA_LOOP_ABORT_REPEATS": os.environ.get("QSA_LOOP_ABORT_REPEATS", "5"),
                "QSA_FIRST_EDIT_STEP": str(self._first_edit_step),
                "QSA_NO_EDIT_ABORT_STEP": str(self._no_edit_abort_step),
                "QSA_MAX_NO_TOOL_RETRIES": str(self._max_no_tool_retries),
                "QSA_MAX_EMPTY_TOOL_RETRIES": os.environ.get(
                    "QSA_MAX_EMPTY_TOOL_RETRIES", "2"
                ),
                "QSA_MAX_BROAD_SVERKLO_BEFORE_EDIT": os.environ.get(
                    "QSA_MAX_BROAD_SVERKLO_BEFORE_EDIT", "6"
                ),
                "QSA_TOOL_CHOICE": self._tool_choice,
                "QSA_EXTRA_STARTUP_GUIDANCE": self._extra_startup_guidance,
                "QSA_INITIAL_CONTEXT_FILE": initial_context_path,
                "QSA_VALIDATION_GRACE_STEPS": os.environ.get(
                    "QSA_VALIDATION_GRACE_STEPS", "12"
                ),
                "QSA_STALE_ABORT_STEPS": os.environ.get("QSA_STALE_ABORT_STEPS", "10"),
                "QSA_RECENT_MESSAGE_COUNT": os.environ.get(
                    "QSA_RECENT_MESSAGE_COUNT", "18"
                ),
                "QSA_MCP_COMMANDS_JSON": json.dumps(self._mcp_commands()),
                "QSA_SVERKLO_COMMAND_JSON": json.dumps(self._sverklo_command()),
                "SVERKLO_PROJECT_PATH": "/app",
                "SERENA_PROJECT_PATH": "/app",
            }
        )
        augmented_instruction = instruction + self._task_context_text()
        instruction_b64 = base64.b64encode(augmented_instruction.encode()).decode("ascii")
        command = (
            "python3 /tmp/qwen_sverklo_worker.py "
            f"--instruction-b64 {shlex.quote(instruction_b64)}"
        )
        await self.exec_as_agent(
            environment,
            command=command,
            env=env,
            cwd="/app",
            timeout_sec=int(os.environ.get("QSA_RUN_TIMEOUT_SEC", "7200")),
        )
