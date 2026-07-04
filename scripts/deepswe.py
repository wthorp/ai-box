#!/usr/bin/env python3
"""DeepSWE harness CLI with MCP overlay support."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import deepswe_harness


CODEX_MCP_AGENT_IMPORT_PATH = "scripts.pier_agents.codex_mcp_run:CodexMcpRun"
QWEN_SVERKLO_AGENT_IMPORT_PATH = "scripts.pier_agents.qwen_sverklo_run:QwenSverkloRun"
CANARY_URL = "http://127.0.0.1:3005/mcp"
CODEGRAPH_URL = "http://127.0.0.1:3006/mcp"
SVERKLO_URL = "http://127.0.0.1:3007/mcp"
SVERKLO_STDIO_COMMAND = "bash"
SVERKLO_STDIO_ARGS = [
    "-lc",
    'if command -v sverklo >/dev/null 2>&1; then '
    'exec sverklo "${SVERKLO_PROJECT_PATH:-.}"; '
    'fi; '
    'NODE24="$(find "$HOME/.nvm/versions/node" -maxdepth 1 -type d -name "v24*" '
    '| sort -V | tail -1 2>/dev/null || true)"; '
    'if [ -z "$NODE24" ]; then echo "Sverklo not installed" >&2; exit 127; fi; '
    'PATH="$NODE24/bin:$PATH" exec sverklo "${SVERKLO_PROJECT_PATH:-.}"',
]
SERENA_STDIO_COMMAND = "bash"
SERENA_STDIO_ARGS = [
    "-lc",
    'export PATH="/usr/local/bin:$HOME/.local/bin:$HOME/go/bin:$PATH"; '
    'exec serena start-mcp-server --transport stdio '
    '--context ide-assistant --project "${SERENA_PROJECT_PATH:-.}" '
    "--mode editing --mode interactive",
]

REWARD_GUARD_MARKER = "DEEPSWE REWARD GUARD"
REWARD_GUARD = f"""# --- {REWARD_GUARD_MARKER}: BEGIN ---
_deepswe_ensure_zero_reward() {{
    mkdir -p /logs/verifier 2>/dev/null || true
    if [ ! -s /logs/verifier/reward.txt ]; then
        printf '0\\n' > /logs/verifier/reward.txt 2>/dev/null || true
    fi
}}
_deepswe_ensure_zero_reward
trap _deepswe_ensure_zero_reward EXIT
# --- {REWARD_GUARD_MARKER}: END ---

"""


MCP_PROFILES: dict[str, list[dict[str, Any]]] = {
    "none": [],
    "canary": [
        {
            "name": "mcp-canary",
            "transport": "streamable_http",
            "url": CANARY_URL,
        }
    ],
    "codegraph": [
        {
            "name": "codegraph",
            "transport": "streamable_http",
            "url": CODEGRAPH_URL,
        }
    ],
    "sverklo": [
        {
            "name": "sverklo",
            "transport": "stdio",
            "command": SVERKLO_STDIO_COMMAND,
            "args": SVERKLO_STDIO_ARGS,
        }
    ],
    "sverklo-serena": [
        {
            "name": "sverklo",
            "transport": "stdio",
            "command": SVERKLO_STDIO_COMMAND,
            "args": SVERKLO_STDIO_ARGS,
        },
        {
            "name": "serena",
            "transport": "stdio",
            "command": SERENA_STDIO_COMMAND,
            "args": SERENA_STDIO_ARGS,
        },
    ],
    "sverklo-http": [
        {
            "name": "sverklo",
            "transport": "streamable_http",
            "url": SVERKLO_URL,
        }
    ],
}


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def quote_toml(value: str) -> str:
    return json.dumps(value)


def mcp_toml_block(servers: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for server in servers:
        lines.append("")
        lines.append("[[environment.mcp_servers]]")
        for key in ("name", "transport", "url", "command", "args"):
            if key not in server:
                continue
            value = server[key]
            if isinstance(value, list):
                lines.append(f"{key} = {json.dumps(value)}")
            else:
                lines.append(f"{key} = {quote_toml(str(value))}")
    return "\n".join(lines) + ("\n" if lines else "")


def inject_mcp_servers(task_toml: str, servers: list[dict[str, Any]]) -> str:
    filtered: list[str] = []
    skipping = False
    for line in task_toml.splitlines():
        stripped = line.strip()
        if stripped == "[[environment.mcp_servers]]":
            skipping = True
            continue
        if skipping and stripped.startswith("["):
            skipping = False
        if not skipping:
            filtered.append(line)
    result = "\n".join(filtered).rstrip() + "\n"
    return result + mcp_toml_block(servers)


def harden_reward_script(text: str) -> str:
    if REWARD_GUARD_MARKER in text:
        return text
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        return "".join([lines[0], REWARD_GUARD, *lines[1:]])
    return REWARD_GUARD + text


def link_or_copy_path(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def link_or_copy_tests_dir(src: Path, dst: Path, harden_rewards: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if harden_rewards and child.name == "test.sh" and child.is_file():
            target.write_text(
                harden_reward_script(child.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            target.chmod(child.stat().st_mode & 0o777)
            continue
        if target.exists() or target.is_symlink():
            continue
        if child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target)


def link_or_copy_tree(
    src: Path, dst: Path, servers: list[dict[str, Any]], harden_rewards: bool
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "task.toml":
            target.write_text(
                inject_mcp_servers(child.read_text(encoding="utf-8"), servers),
                encoding="utf-8",
            )
            continue
        if child.name == "tests" and child.is_dir() and harden_rewards:
            link_or_copy_tests_dir(child, target, harden_rewards)
            continue
        link_or_copy_path(child, target)


def create_task_overlay(
    task_path: Path,
    output_root: Path,
    servers: list[dict[str, Any]],
    harden_rewards: bool = False,
) -> Path:
    overlay_root = output_root / "mcp-task-overlay"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True)

    if (task_path / "task.toml").is_file():
        link_or_copy_tree(
            task_path, overlay_root / task_path.name, servers, harden_rewards
        )
        return overlay_root / task_path.name

    task_dirs = [
        path for path in sorted(task_path.iterdir()) if (path / "task.toml").is_file()
    ]
    if not task_dirs:
        raise FileNotFoundError(f"no task.toml files found under {task_path}")
    for task_dir in task_dirs:
        link_or_copy_tree(
            task_dir, overlay_root / task_dir.name, servers, harden_rewards
        )
    return overlay_root


def mcp_hosts(servers: list[dict[str, Any]]) -> list[str]:
    hosts: list[str] = []
    for server in servers:
        url = server.get("url")
        if not url:
            continue
        host = urllib.parse.urlparse(url).hostname
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def rpc(url: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def cmd_mcp_canary(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("MCP_CANARY_URL", CANARY_URL))
    parser.add_argument("--job-id", default=f"canary-{timestamp()}")
    parser.add_argument("--expect-nonce", default=os.environ.get("MCP_CANARY_NONCE"))
    args = parser.parse_args(argv)

    init = rpc(args.url, "initialize", {"clientInfo": {"name": "deepswe-harness"}})
    tools = rpc(args.url, "tools/list")
    call = rpc(
        args.url,
        "tools/call",
        {"name": "canary_nonce", "arguments": {"job_id": args.job_id}},
    )
    text = call["result"]["content"][0]["text"]
    result = json.loads(text)
    if args.expect_nonce and result.get("nonce") != args.expect_nonce:
        raise RuntimeError(
            f"nonce mismatch: expected {args.expect_nonce}, got {result.get('nonce')}"
        )
    print(json.dumps({"initialize": init, "tools": tools, "call": result}, indent=2))
    return 0


def cmd_codegraph_preflight(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=os.environ.get("CODEGRAPH_MCP_URL", CODEGRAPH_URL)
    )
    parser.add_argument("--require-agentic", action="store_true")
    args = parser.parse_args(argv)

    missing = [
        name
        for name in ("CODEGRAPH_LLM_PROVIDER", "CODEGRAPH_EMBEDDING_PROVIDER")
        if not os.environ.get(name)
    ]
    if args.require_agentic and missing:
        print(
            "CodeGraph agentic preflight requires provider env: " + ",".join(missing),
            file=sys.stderr,
        )
        return 2

    tools = rpc(args.url, "tools/list")
    print(json.dumps(tools, indent=2))
    return 0


def cmd_sverklo_preflight(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SVERKLO_MCP_URL", SVERKLO_URL))
    parser.add_argument("--show-tools", action="store_true")
    parser.add_argument("--skip-status", action="store_true")
    args = parser.parse_args(argv)

    init = rpc(args.url, "initialize", {"clientInfo": {"name": "deepswe-harness"}})
    tools = rpc(args.url, "tools/list")
    tool_names = [
        tool.get("name")
        for tool in tools.get("result", {}).get("tools", [])
        if isinstance(tool, dict)
    ]
    output: dict[str, Any] = {
        "initialize": init.get("result", {}),
        "tool_count": len(tool_names),
        "tools": tool_names,
    }
    if not args.skip_status:
        status = rpc(args.url, "tools/call", {"name": "status", "arguments": {}})
        content = status.get("result", {}).get("content", [])
        if content and isinstance(content[0], dict):
            output["status_excerpt"] = str(content[0].get("text", ""))[:1200]
        else:
            output["status_response"] = status
    if args.show_tools:
        output["tools_list_response"] = tools
    print(json.dumps(output, indent=2))
    return 0


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task-path",
        "--path",
        default=os.environ.get("DEEPSWE_DIR", "/deep-swe") + "/tasks",
    )
    parser.add_argument("--model", default="openai/local")
    parser.add_argument("--agent", default="mini-swe-agent")
    parser.add_argument("--agent-import-path", default=deepswe_harness.LOCAL_AGENT_IMPORT_PATH)
    parser.add_argument(
        "--environment-import-path",
        default=deepswe_harness.LOCAL_ENVIRONMENT_IMPORT_PATH,
    )
    parser.add_argument("--mcp-profile", choices=sorted(MCP_PROFILES), default="none")
    parser.add_argument("--mcp-url")
    parser.add_argument(
        "--results-dir", default=os.environ.get("EVAL_RESULTS_DIR", "/eval-results")
    )
    parser.add_argument("--debug-harness", action="store_true")
    parser.add_argument(
        "--harden-rewards",
        dest="harden_rewards",
        action="store_true",
        default=True,
        help="overlay task test.sh files so verifier setup failures emit reward 0",
    )
    parser.add_argument(
        "--no-harden-rewards",
        dest="harden_rewards",
        action="store_false",
        help="use task verifier scripts exactly as provided",
    )
    return parser


def overlay_from_args(args: argparse.Namespace, output_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    servers = [dict(server) for server in MCP_PROFILES[args.mcp_profile]]
    if args.mcp_url and servers:
        servers[0]["url"] = args.mcp_url

    task_path = Path(args.task_path)
    if servers or args.harden_rewards:
        task_path = create_task_overlay(
            task_path,
            output_root,
            servers,
            harden_rewards=args.harden_rewards,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "task-overlay.json").write_text(
            json.dumps(
                {
                    "source_task_path": args.task_path,
                    "overlay_task_path": str(task_path),
                    "harden_rewards": args.harden_rewards,
                    "mcp_servers": servers,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if servers:
            (output_root / "mcp-servers.json").write_text(
                json.dumps(servers, indent=2) + "\n", encoding="utf-8"
            )
            (output_root / "mcp-allowlist-hosts.txt").write_text(
                "\n".join(mcp_hosts(servers)) + "\n", encoding="utf-8"
            )
    return task_path, servers


def cmd_overlay(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(parents=[build_common_parser()])
    args = parser.parse_args(argv)
    task_path, _ = overlay_from_args(args, Path(args.results_dir))
    print(task_path)
    return 0


def cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(parents=[build_common_parser()])
    parser.add_argument("--job-name")
    parser.add_argument("--jobs-dir")
    parser.add_argument("--n-tasks")
    parser.add_argument("--sample-seed")
    parser.add_argument("--n-concurrent")
    parser.add_argument("--quiet-yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    known, extra = deepswe_harness.split_extra(argv)
    args = parser.parse_args(known)

    run_root = Path(args.results_dir) / f"deepswe-{args.mcp_profile}-{timestamp()}"
    if args.jobs_dir is None:
        args.jobs_dir = str(run_root / "pier-jobs")
    task_path, servers = overlay_from_args(args, run_root)
    if servers:
        if (
            args.agent == "codex"
            and args.agent_import_path == deepswe_harness.LOCAL_AGENT_IMPORT_PATH
        ):
            args.agent_import_path = CODEX_MCP_AGENT_IMPORT_PATH
        if (
            args.agent == "qwen-sverklo"
            and args.agent_import_path == deepswe_harness.LOCAL_AGENT_IMPORT_PATH
        ):
            args.agent_import_path = QWEN_SVERKLO_AGENT_IMPORT_PATH

    pier_args = deepswe_harness.build_pier_args(
        argparse.Namespace(
            task_path=str(task_path),
            model=args.model,
            agent=args.agent,
            agent_import_path=args.agent_import_path,
            environment_import_path=args.environment_import_path,
            debug_harness=args.debug_harness,
            job_name=args.job_name,
            jobs_dir=args.jobs_dir,
            n_tasks=args.n_tasks,
            sample_seed=args.sample_seed,
            n_concurrent=args.n_concurrent,
            quiet_yes=args.quiet_yes,
        ),
        extra,
    )
    if args.dry_run:
        print(json.dumps({"pier_args": pier_args, "task_path": str(task_path)}, indent=2))
        return 0
    os.execvp("pier", ["pier", *pier_args])
    return 127


def cmd_sweep(argv: list[str]) -> int:
    script = Path(__file__).with_name("deepswe-sweep.sh")
    os.execv(str(script), [str(script), *argv])
    return 127


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: deepswe.py <run|sweep|overlay|mcp-canary|codegraph-preflight|sverklo-preflight>",
            file=sys.stderr,
        )
        return 2
    command, rest = argv[0], argv[1:]
    if command == "run":
        return cmd_run(rest)
    if command == "sweep":
        return cmd_sweep(rest)
    if command == "overlay":
        return cmd_overlay(rest)
    if command == "mcp-canary":
        return cmd_mcp_canary(rest)
    if command == "codegraph-preflight":
        return cmd_codegraph_preflight(rest)
    if command == "sverklo-preflight":
        return cmd_sverklo_preflight(rest)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
