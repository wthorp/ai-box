#!/usr/bin/env python3
"""Run low-risk deterministic autofixers for changed repository files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PYTHON_EXTS = {".py", ".pyi"}
JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".css", ".scss", ".md", ".yaml", ".yml"}
GO_EXTS = {".go"}
RUST_EXTS = {".rs"}
SHELL_EXTS = {".sh", ".bash", ".zsh"}
SQL_EXTS = {".sql"}
CONFIG_EXTS = {".json", ".toml", ".yaml", ".yml"}
TEXT_HYGIENE_MAX_BYTES = 1_000_000
IGNORED_PATH_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".class"}
TEST_PATH_PARTS = {
    "__tests__",
    "spec",
    "specs",
    "test",
    "tests",
    "testing",
}
TEST_NAME_MARKERS = (
    ".spec.",
    ".test.",
    "_spec.",
    "_test.",
    "conftest.py",
)
DOC_EXTS = {".md", ".rst", ".txt"}
LOCKFILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "go.sum",
}
SOURCE_EXTS = PYTHON_EXTS | {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"} | GO_EXTS | RUST_EXTS | SHELL_EXTS | SQL_EXTS


@dataclass
class CommandResult:
    name: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    stdout: str
    stderr: str


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def run_git(cwd: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "-c", f"safe.directory={cwd.resolve()}", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def changed_files(cwd: Path) -> list[Path]:
    names = set(
        run_git(cwd, ["diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD", "--"])
    )
    names.update(run_git(cwd, ["ls-files", "--others", "--exclude-standard"]))
    paths: list[Path] = []
    for name in sorted(names):
        path = (cwd / name).resolve()
        try:
            path.relative_to(cwd.resolve())
        except ValueError:
            continue
        rel_parts = set(path.relative_to(cwd.resolve()).parts)
        if rel_parts & IGNORED_PATH_PARTS:
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if path.is_file():
            paths.append(path)
    return paths


def repo_bin(cwd: Path, name: str) -> str | None:
    local = cwd / "node_modules" / ".bin" / name
    if local.exists():
        return str(local)
    return shutil.which(name)


def existing_tool(cwd: Path, *names: str) -> str | None:
    for name in names:
        if found := repo_bin(cwd, name):
            return found
    return None


def has_any(cwd: Path, names: Iterable[str]) -> bool:
    return any((cwd / name).exists() for name in names)


def rels(cwd: Path, paths: Iterable[Path]) -> list[str]:
    return [str(path.relative_to(cwd)) for path in paths]


def _is_test_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return bool(parts & TEST_PATH_PARTS) or any(marker in name for marker in TEST_NAME_MARKERS)


def _is_doc_or_lock(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTS or path.name in LOCKFILE_NAMES


def _is_source_path(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTS


def command_plan(cwd: Path, paths: list[Path]) -> list[tuple[str, list[str]]]:
    by_ext: dict[str, list[Path]] = {}
    for path in paths:
        by_ext.setdefault(path.suffix, []).append(path)

    commands: list[tuple[str, list[str]]] = []
    py_files = [path for ext in PYTHON_EXTS for path in by_ext.get(ext, [])]
    if py_files and (ruff := existing_tool(cwd, "ruff")):
        commands.append(("ruff-check-fix", [ruff, "check", "--no-cache", "--fix", *rels(cwd, py_files)]))
        commands.append(("ruff-format", [ruff, "format", "--no-cache", *rels(cwd, py_files)]))

    js_files = [path for ext in JS_EXTS for path in by_ext.get(ext, [])]
    if js_files:
        if (biome := existing_tool(cwd, "biome")) and has_any(
            cwd, ("biome.json", "biome.jsonc")
        ):
            commands.append(("biome-check-write", [biome, "check", "--write", *rels(cwd, js_files)]))
        elif (prettier := existing_tool(cwd, "prettier")) and has_any(
            cwd,
            (
                ".prettierrc",
                ".prettierrc.json",
                ".prettierrc.yml",
                ".prettierrc.yaml",
                ".prettierrc.js",
                "prettier.config.js",
                "prettier.config.cjs",
                "prettier.config.mjs",
            ),
        ):
            commands.append(("prettier-write", [prettier, "--write", *rels(cwd, js_files)]))

    go_files = [path for ext in GO_EXTS for path in by_ext.get(ext, [])]
    if go_files and (gofmt := existing_tool(cwd, "gofmt")):
        commands.append(("gofmt", [gofmt, "-w", *rels(cwd, go_files)]))

    rust_files = [path for ext in RUST_EXTS for path in by_ext.get(ext, [])]
    if rust_files and (rustfmt := existing_tool(cwd, "rustfmt")):
        commands.append(("rustfmt", [rustfmt, *rels(cwd, rust_files)]))

    shell_files = [path for ext in SHELL_EXTS for path in by_ext.get(ext, [])]
    if shell_files and (shfmt := existing_tool(cwd, "shfmt")):
        commands.append(("shfmt", [shfmt, "-w", *rels(cwd, shell_files)]))

    return commands


def run_command(cwd: Path, name: str, command: list[str], timeout_sec: int, output_limit: int) -> CommandResult:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        return CommandResult(
            name=name,
            command=command,
            return_code=proc.returncode,
            elapsed_sec=round(time.time() - started, 3),
            stdout=truncate(proc.stdout, output_limit),
            stderr=truncate(proc.stderr, output_limit),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=command,
            return_code=124,
            elapsed_sec=round(time.time() - started, 3),
            stdout=truncate(exc.stdout or "", output_limit),
            stderr=truncate(exc.stderr or "timed out", output_limit),
        )


def ruff_critic(cwd: Path, paths: list[Path], timeout_sec: int, output_limit: int) -> dict[str, object]:
    py_paths = [path for path in paths if path.suffix.lower() in PYTHON_EXTS]
    if not py_paths:
        return {"available": bool(existing_tool(cwd, "ruff")), "analyzed_count": 0, "ok": True, "findings": []}
    ruff = existing_tool(cwd, "ruff")
    if not ruff:
        return {
            "available": False,
            "analyzed_count": len(py_paths),
            "ok": True,
            "findings": [],
            "warning": "ruff unavailable; python critic skipped",
        }
    command = [ruff, "check", "--no-cache", "--output-format=json", *rels(cwd, py_paths)]
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "analyzed_count": len(py_paths),
            "ok": False,
            "findings": [],
            "error": "ruff critic timed out",
            "stderr": truncate(exc.stderr or "", output_limit),
        }

    try:
        parsed = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        parsed = []
    findings = parsed if isinstance(parsed, list) else []
    return {
        "available": True,
        "analyzed_count": len(py_paths),
        "ok": proc.returncode == 0,
        "command": command,
        "elapsed_sec": round(time.time() - started, 3),
        "finding_count": len(findings),
        "findings": findings[:20],
        "stderr": truncate(proc.stderr, output_limit),
    }


def shellcheck_critic(cwd: Path, paths: list[Path], timeout_sec: int, output_limit: int) -> dict[str, object]:
    shell_paths = [path for path in paths if path.suffix.lower() in SHELL_EXTS]
    if not shell_paths:
        return {"available": bool(existing_tool(cwd, "shellcheck")), "analyzed_count": 0, "ok": True, "findings": []}
    shellcheck = existing_tool(cwd, "shellcheck")
    if not shellcheck:
        return {
            "available": False,
            "analyzed_count": len(shell_paths),
            "ok": True,
            "findings": [],
            "warning": "shellcheck unavailable; shell critic skipped",
        }
    command = [shellcheck, "--format=json", *rels(cwd, shell_paths)]
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "analyzed_count": len(shell_paths),
            "ok": False,
            "findings": [],
            "error": "shellcheck critic timed out",
            "stderr": truncate(exc.stderr or "", output_limit),
        }

    try:
        parsed = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        comments = parsed.get("comments", [])
    elif isinstance(parsed, list):
        comments = parsed
    else:
        comments = []
    findings = comments if isinstance(comments, list) else []
    return {
        "available": True,
        "analyzed_count": len(shell_paths),
        "ok": proc.returncode == 0,
        "command": command,
        "elapsed_sec": round(time.time() - started, 3),
        "finding_count": len(findings),
        "findings": findings[:20],
        "stderr": truncate(proc.stderr, output_limit),
    }


def patch_shape_critic(cwd: Path, paths: list[Path], output_limit: int) -> dict[str, object]:
    changed = rels(cwd, paths)
    source_paths = [path for path in paths if _is_source_path(path)]
    test_paths = [path for path in paths if _is_test_path(path)]
    implementation_paths = [
        path
        for path in source_paths
        if not _is_test_path(path) and not _is_doc_or_lock(path)
    ]
    doc_or_lock_paths = [path for path in paths if _is_doc_or_lock(path)]

    findings: list[dict[str, object]] = []
    if not changed:
        findings.append(
            {
                "ok": False,
                "kind": "no_changed_files",
                "message": "No repository files changed.",
            }
        )
    if source_paths and not implementation_paths:
        findings.append(
            {
                "ok": False,
                "kind": "tests_only_or_non_impl_patch",
                "message": "Changed source-like files are tests, docs, or lockfiles only.",
                "changed_files": changed[:40],
            }
        )
    if len(changed) > int(os.environ.get("QSA_PATCH_MAX_CHANGED_FILES", "25")):
        findings.append(
            {
                "ok": False,
                "kind": "too_many_changed_files",
                "message": "Patch touches too many files for a targeted DeepSWE fix.",
                "changed_count": len(changed),
            }
        )

    return {
        "available": True,
        "analyzed_count": len(paths),
        "ok": not findings,
        "implementation_count": len(implementation_paths),
        "test_count": len(test_paths),
        "doc_or_lock_count": len(doc_or_lock_paths),
        "changed_count": len(changed),
        "implementation_files": rels(cwd, implementation_paths)[:40],
        "test_files": rels(cwd, test_paths)[:40],
        "findings": findings,
    }


def typecheck_critic(cwd: Path, paths: list[Path], timeout_sec: int, output_limit: int) -> dict[str, object]:
    if os.environ.get("QSA_TYPECHECK_CRITIC", "1") == "0":
        return {"available": True, "analyzed_count": 0, "ok": True, "findings": [], "skipped": True}

    findings: list[dict[str, object]] = []
    commands: list[tuple[str, list[str]]] = []
    suffixes = {path.suffix.lower() for path in paths}

    if suffixes & PYTHON_EXTS:
        if pyright := existing_tool(cwd, "pyright"):
            commands.append(("pyright", [pyright, "--outputjson", *rels(cwd, [p for p in paths if p.suffix.lower() in PYTHON_EXTS])]))
        elif (mypy := existing_tool(cwd, "mypy")) and has_any(cwd, ("mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg")):
            commands.append(("mypy", [mypy, "--show-error-codes", *rels(cwd, [p for p in paths if p.suffix.lower() in PYTHON_EXTS])]))

    if suffixes & {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        if (tsc := existing_tool(cwd, "tsc")) and has_any(cwd, ("tsconfig.json", "jsconfig.json")):
            commands.append(("tsc-noemit", [tsc, "--noEmit", "--pretty", "false"]))

    if suffixes & GO_EXTS and (go := existing_tool(cwd, "go")) and (cwd / "go.mod").exists():
        commands.append(("go-test-compile", [go, "test", "./...", "-run", "^$"]))
        if staticcheck := existing_tool(cwd, "staticcheck"):
            commands.append(("staticcheck", [staticcheck, "./..."]))

    if suffixes & RUST_EXTS and (cargo := existing_tool(cwd, "cargo")) and (cwd / "Cargo.toml").exists():
        commands.append(("cargo-check", [cargo, "check", "--quiet"]))

    if not commands:
        return {
            "available": False,
            "analyzed_count": len(paths),
            "ok": True,
            "findings": [],
            "warning": "no applicable type/build critic command available",
        }

    for name, command in commands[: int(os.environ.get("QSA_TYPECHECK_MAX_COMMANDS", "3"))]:
        result = run_command(cwd, name, command, timeout_sec, output_limit)
        findings.append(
            {
                "name": name,
                "ok": result.return_code == 0,
                "command": result.command,
                "return_code": result.return_code,
                "elapsed_sec": result.elapsed_sec,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

    return {
        "available": True,
        "analyzed_count": len(paths),
        "ok": all(bool(finding.get("ok")) for finding in findings),
        "findings": findings,
    }


def semgrep_critic(cwd: Path, paths: list[Path], timeout_sec: int, output_limit: int) -> dict[str, object]:
    if os.environ.get("QSA_SEMGREP_CRITIC", "0") == "0":
        return {
            "available": bool(existing_tool(cwd, "semgrep")),
            "analyzed_count": 0,
            "ok": True,
            "findings": [],
            "skipped": True,
            "warning": "semgrep critic disabled by default; set QSA_SEMGREP_CRITIC=1 to enable",
        }
    semgrep = existing_tool(cwd, "semgrep")
    if not semgrep:
        return {
            "available": False,
            "analyzed_count": len(paths),
            "ok": True,
            "findings": [],
            "warning": "semgrep unavailable; semantic rule critic skipped",
        }
    source_paths = [path for path in paths if _is_source_path(path)]
    if not source_paths:
        return {"available": True, "analyzed_count": 0, "ok": True, "findings": []}
    command = [
        semgrep,
        "--config",
        "auto",
        "--json",
        "--error",
        "--disable-version-check",
        *rels(cwd, source_paths),
    ]
    result = run_command(cwd, "semgrep-auto", command, timeout_sec, output_limit)
    findings: list[object] = []
    try:
        parsed = json.loads(result.stdout or "{}")
        raw_findings = parsed.get("results", []) if isinstance(parsed, dict) else []
        findings = raw_findings[:20] if isinstance(raw_findings, list) else []
    except json.JSONDecodeError:
        findings = []
    return {
        "available": True,
        "analyzed_count": len(source_paths),
        "ok": result.return_code == 0,
        "command": command,
        "elapsed_sec": result.elapsed_sec,
        "finding_count": len(findings),
        "findings": findings,
        "stdout": truncate(result.stdout, output_limit),
        "stderr": truncate(result.stderr, output_limit),
    }


def config_syntax_critic(cwd: Path, paths: list[Path], output_limit: int) -> dict[str, object]:
    config_paths = [path for path in paths if path.suffix.lower() in CONFIG_EXTS]
    findings: list[dict[str, object]] = []
    yaml_module = None
    yaml_available = True
    yaml_needed = any(path.suffix.lower() in {".yaml", ".yml"} for path in config_paths)
    if yaml_needed:
        try:
            import yaml as yaml_module  # type: ignore[import-not-found,no-redef]
        except Exception:
            yaml_available = False

    for path in config_paths:
        rel = str(path.relative_to(cwd))
        suffix = path.suffix.lower()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append({"path": rel, "ok": False, "error": str(exc)})
            continue

        try:
            if suffix == ".json":
                json.loads(text)
            elif suffix == ".toml":
                tomllib.loads(text)
            elif suffix in {".yaml", ".yml"}:
                if yaml_module is None:
                    findings.append(
                        {
                            "path": rel,
                            "ok": True,
                            "skipped": True,
                            "warning": "pyyaml unavailable; yaml syntax skipped",
                        }
                    )
                    continue
                yaml_module.safe_load(text)
            else:
                continue
        except Exception as exc:
            findings.append(
                {
                    "path": rel,
                    "ok": False,
                    "parser": suffix.removeprefix("."),
                    "error": truncate(f"{type(exc).__name__}: {exc}", output_limit),
                }
            )
            continue
        findings.append({"path": rel, "ok": True, "parser": suffix.removeprefix(".")})

    return {
        "available": {"json": True, "toml": True, "yaml": yaml_available},
        "analyzed_count": len(config_paths),
        "ok": all(bool(finding.get("ok")) for finding in findings),
        "findings": findings,
    }


def text_hygiene_critic(cwd: Path, paths: list[Path], output_limit: int) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    markers = ("<<<<<<<", "=======", ">>>>>>>")
    for path in paths:
        rel = str(path.relative_to(cwd))
        try:
            data = path.read_bytes()
        except OSError as exc:
            findings.append({"path": rel, "ok": False, "error": str(exc)})
            continue
        if len(data) > TEXT_HYGIENE_MAX_BYTES:
            findings.append(
                {
                    "path": rel,
                    "ok": True,
                    "skipped": True,
                    "warning": f"text hygiene skipped; file exceeds {TEXT_HYGIENE_MAX_BYTES} bytes",
                }
            )
            continue
        if b"\x00" in data:
            findings.append({"path": rel, "ok": False, "kind": "nul_byte"})
            continue
        text = data.decode("utf-8", errors="replace")
        marker_hits: list[dict[str, object]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if any(stripped.startswith(marker) for marker in markers):
                marker_hits.append(
                    {
                        "line": line_no,
                        "text": truncate(line.strip(), output_limit),
                    }
                )
        if marker_hits:
            findings.append(
                {
                    "path": rel,
                    "ok": False,
                    "kind": "conflict_marker",
                    "markers": marker_hits[:20],
                }
            )
        else:
            findings.append({"path": rel, "ok": True})

    return {
        "available": True,
        "analyzed_count": len(paths),
        "ok": all(bool(finding.get("ok")) for finding in findings),
        "findings": findings,
    }


def sqlglot_critic(cwd: Path, paths: list[Path], output_limit: int) -> dict[str, object]:
    try:
        import sqlglot  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "available": False,
            "analyzed_count": 0,
            "ok": False,
            "findings": [],
            "error": f"required sqlglot unavailable: {type(exc).__name__}: {exc}",
        }

    sql_paths = [path for path in paths if path.suffix.lower() in SQL_EXTS]
    if not sql_paths:
        return {"available": True, "analyzed_count": 0, "ok": True, "findings": []}

    dialects = [
        item.strip()
        for item in os.environ.get(
            "QSA_SQLGLOT_DIALECTS",
            "bigquery,postgres,sqlite,mysql,snowflake,duckdb,tsql,spark",
        ).split(",")
        if item.strip()
    ]
    findings: list[dict[str, object]] = []
    for path in sql_paths:
        rel = str(path.relative_to(cwd))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append({"path": rel, "ok": False, "error": str(exc)})
            continue
        errors: list[str] = []
        parsed = False
        for dialect in dialects:
            try:
                expressions = sqlglot.parse(text, read=dialect)
            except Exception as exc:
                errors.append(f"{dialect}: {type(exc).__name__}: {exc}")
                continue
            previews: list[str] = []
            for expression in expressions[:3]:
                try:
                    previews.append(expression.sql(dialect=dialect))
                except Exception:
                    previews.append(str(expression))
            findings.append(
                {
                    "path": rel,
                    "ok": True,
                    "dialect": dialect,
                    "expression_count": len(expressions),
                    "normalized_preview": truncate("\n".join(previews), output_limit),
                }
            )
            parsed = True
            break
        if not parsed:
            findings.append(
                {
                    "path": rel,
                    "ok": False,
                    "errors": [truncate(error, output_limit) for error in errors[:4]],
                }
            )
    return {
        "available": True,
        "analyzed_count": len(sql_paths),
        "ok": all(bool(finding.get("ok")) for finding in findings),
        "findings": findings,
    }


def policy_summary(critics: dict[str, dict[str, object]], results: list[CommandResult]) -> dict[str, object]:
    hard_fail_critics = {
        item.strip()
        for item in os.environ.get(
            "QSA_HARD_FAIL_CRITICS",
            "text_hygiene,patch_shape,config_syntax,sqlglot,ruff,shellcheck,typecheck",
        ).split(",")
        if item.strip()
    }
    hard_failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for name, result in critics.items():
        ok = bool(result.get("ok", True))
        analyzed_count = int(result.get("analyzed_count") or 0)
        if ok:
            if result.get("warning"):
                warnings.append({"critic": name, "warning": result.get("warning")})
            continue
        target = (
            hard_failures
            if name in hard_fail_critics and (analyzed_count > 0 or name == "patch_shape")
            else warnings
        )
        target.append(
            {
                "critic": name,
                "available": result.get("available"),
                "analyzed_count": analyzed_count,
                "error": result.get("error"),
                "findings": result.get("findings", [])[:5]
                if isinstance(result.get("findings"), list)
                else [],
            }
        )
    for result in results:
        if result.return_code != 0:
            hard_failures.append(
                {
                    "critic": result.name,
                    "command": result.command,
                    "return_code": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
    return {
        "hard_fail": bool(hard_failures),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "rules": {
            "hard_fail_critics": sorted(hard_fail_critics),
            "description": (
                "Hard-fail critics make the rollout low-value for learning unless the agent fixes them "
                "before finishing. Warnings are surfaced but do not make the broker fail."
            ),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("QSA_AUTOFIX_TIMEOUT_SEC", "120")))
    parser.add_argument("--output-limit", type=int, default=int(os.environ.get("QSA_AUTOFIX_OUTPUT_LIMIT", "3000")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    before = run_git(cwd, ["status", "--porcelain", "--untracked-files=normal"])
    paths = changed_files(cwd)
    plan = command_plan(cwd, paths)
    results: list[CommandResult] = []
    if not args.dry_run:
        for name, command in plan:
            results.append(run_command(cwd, name, command, args.timeout_sec, args.output_limit))
    after = run_git(cwd, ["status", "--porcelain", "--untracked-files=normal"])
    critics = {
        "patch_shape": patch_shape_critic(cwd, paths, args.output_limit),
        "text_hygiene": text_hygiene_critic(cwd, paths, args.output_limit),
        "sqlglot": sqlglot_critic(cwd, paths, args.output_limit),
        "ruff": ruff_critic(cwd, paths, args.timeout_sec, args.output_limit),
        "shellcheck": shellcheck_critic(cwd, paths, args.timeout_sec, args.output_limit),
        "config_syntax": config_syntax_critic(cwd, paths, args.output_limit),
        "typecheck": typecheck_critic(cwd, paths, args.timeout_sec, args.output_limit),
        "semgrep": semgrep_critic(cwd, paths, args.timeout_sec, args.output_limit),
    }
    policy = policy_summary(critics, results)
    payload = {
        "changed_files": rels(cwd, paths),
        "planned": [{"name": name, "command": command} for name, command in plan],
        "ran": [result.__dict__ for result in results],
        "critics": critics,
        "policy": policy,
        "changed_by_autofix": before != after,
        "ok": not bool(policy["hard_fail"]),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(f"changed_files={len(paths)} planned={len(plan)} ran={len(results)} ok={payload['ok']}")
        for result in results:
            print(f"{result.name}: rc={result.return_code} elapsed={result.elapsed_sec}s")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
