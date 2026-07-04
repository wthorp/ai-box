from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepswe
import deepswe_harness
import autofix_broker
import autofix_fixture
import skill_mcp
import tabbyapi_config
import trace_scorer
import transition_analysis
from pier_agents import qwen_sverklo_worker


class DeepSweHarnessTests(unittest.TestCase):
    def test_pier_args_defaults_local_cost_limit(self) -> None:
        args = argparse.Namespace(
            task_path="/tasks",
            model="openai/local",
            job_name=None,
            jobs_dir=None,
            n_tasks="1",
            sample_seed=None,
            n_concurrent="1",
            quiet_yes=True,
            debug_harness=False,
            agent="mini-swe-agent",
            agent_import_path=deepswe_harness.LOCAL_AGENT_IMPORT_PATH,
            environment_import_path=deepswe_harness.LOCAL_ENVIRONMENT_IMPORT_PATH,
        )
        pier_args = deepswe_harness.build_pier_args(args, [])
        self.assertIn("--agent-kwarg", pier_args)
        self.assertIn("cost_limit=None", pier_args)

    def test_inject_mcp_servers_replaces_existing_blocks(self) -> None:
        original = """
[environment]
allow_internet = false

[[environment.mcp_servers]]
name = "old"
url = "http://old"

[agent]
timeout_sec = 1
""".lstrip()
        injected = deepswe.inject_mcp_servers(
            original,
            [
                {
                    "name": "mcp-canary",
                    "transport": "streamable_http",
                    "url": "http://127.0.0.1:3005/mcp",
                }
            ],
        )
        self.assertNotIn("old", injected)
        self.assertIn("[agent]", injected)
        self.assertIn('name = "mcp-canary"', injected)

    def test_create_task_overlay_and_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "tasks" / "sample"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("[environment]\nallow_internet = false\n")
            (task / "payload.txt").write_text("ok")
            (task / "tests").mkdir()
            (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 3\n")
            (task / "tests" / "test.patch").write_text("patch")

            servers = [
                {
                    "name": "x",
                    "transport": "streamable_http",
                    "url": "http://localhost:1/mcp",
                }
            ]
            overlay = deepswe.create_task_overlay(
                root / "tasks",
                root / "out",
                servers,
                harden_rewards=True,
            )

            self.assertEqual(["localhost"], deepswe.mcp_hosts(servers))
            self.assertTrue((overlay / "sample" / "payload.txt").exists())
            self.assertIn(
                "[[environment.mcp_servers]]",
                (overlay / "sample" / "task.toml").read_text(),
            )
            self.assertIn(
                deepswe.REWARD_GUARD_MARKER,
                (overlay / "sample" / "tests" / "test.sh").read_text(),
            )
            self.assertFalse((overlay / "sample" / "tests" / "test.patch").is_symlink())

    def test_harden_reward_script_is_idempotent(self) -> None:
        script = "#!/bin/bash\nexit 2\n"
        hardened = deepswe.harden_reward_script(script)
        self.assertTrue(hardened.startswith("#!/bin/bash\n# --- DEEPSWE"))
        self.assertEqual(hardened, deepswe.harden_reward_script(hardened))

    def test_qwen_sverklo_agent_import_path_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            (task / "task.toml").write_text("[environment]\nallow_internet = false\n")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "deepswe.py",
                    "run",
                    "--agent",
                    "qwen-sverklo",
                    "--mcp-profile",
                    "sverklo",
                    "--task-path",
                    str(task),
                    "--results-dir",
                    str(root / "results"),
                    "--dry-run",
                ]
                from io import StringIO
                from contextlib import redirect_stdout

                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, deepswe.main(sys.argv[1:]))
                dry_run = json.loads(output.getvalue())
                self.assertIn(
                    "scripts.pier_agents.qwen_sverklo_run:QwenSverkloRun",
                    dry_run["pier_args"],
                )
            finally:
                sys.argv = original_argv

    def test_qwen_sverklo_serena_profile_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            (task / "task.toml").write_text("[environment]\nallow_internet = false\n")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "deepswe.py",
                    "run",
                    "--agent",
                    "qwen-sverklo",
                    "--mcp-profile",
                    "sverklo-serena",
                    "--task-path",
                    str(task),
                    "--results-dir",
                    str(root / "results"),
                    "--dry-run",
                ]
                from io import StringIO
                from contextlib import redirect_stdout

                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, deepswe.main(sys.argv[1:]))
                dry_run = json.loads(output.getvalue())
                overlay = Path(dry_run["task_path"]) / "task.toml"
                if not overlay.exists():
                    overlay = Path(dry_run["task_path"]) / task.name / "task.toml"
                text = overlay.read_text()
                self.assertIn('name = "sverklo"', text)
                self.assertIn('name = "serena"', text)
                self.assertIn(
                    "scripts.pier_agents.qwen_sverklo_run:QwenSverkloRun",
                    dry_run["pier_args"],
                )
            finally:
                sys.argv = original_argv

    def test_tabbyapi_config_infers_gemma_tool_format(self) -> None:
        old_env = dict(tabbyapi_config.os.environ)
        try:
            for key in list(tabbyapi_config.os.environ):
                if key.startswith("TABBY_") or key == "CONTEXT_SIZE":
                    tabbyapi_config.os.environ.pop(key)
            tabbyapi_config.os.environ["CONTEXT_SIZE"] = "32769"
            config = tabbyapi_config.build_config("gemma-4-31b-dense-exl3")
            self.assertIn('model_name: "gemma-4-31b-dense-exl3"', config)
            self.assertIn("backend: \"exllamav3\"", config)
            self.assertIn("cache_size: 32768", config)
            self.assertIn('cache_mode: "Q8"', config)
            self.assertIn("autosplit_reserve: [1024]", config)
            self.assertIn("gpu_split: []", config)
            self.assertIn('override_preset: "qsa_coding"', config)
            self.assertIn("tool_format: \"gemma4\"", config)
            self.assertIn("force_enable_thinking: true", config)
            self.assertIn('dummy_model_names: ["local", "gemma-4-31b-dense-exl3", "gpt-3.5-turbo"]', config)
            preset = tabbyapi_config.build_sampler_preset()
            self.assertIn("temperature:\n  override: 0.8\n  force: false", preset)
            self.assertIn("mirostat_mode:\n  override: 2\n  force: false", preset)
            self.assertIn("presence_penalty:\n  override: 0.02\n  force: false", preset)
        finally:
            tabbyapi_config.os.environ.clear()
            tabbyapi_config.os.environ.update(old_env)

    def test_tabbyapi_config_infers_qwen_tool_format(self) -> None:
        old_env = dict(tabbyapi_config.os.environ)
        try:
            for key in list(tabbyapi_config.os.environ):
                if key.startswith("TABBY_") or key == "CONTEXT_SIZE":
                    tabbyapi_config.os.environ.pop(key)
            config = tabbyapi_config.build_config("qwen3-coder-next-4.0bpw")
            self.assertIn('model_name: "qwen3-coder-next-4.0bpw"', config)
            self.assertIn("tool_format: \"qwen3_coder\"", config)
            self.assertIn("force_enable_thinking: false", config)
            self.assertIn("reasoning: true", config)
            self.assertIn('dummy_model_names: ["local", "qwen3-coder-next-4.0bpw", "gpt-3.5-turbo"]', config)
        finally:
            tabbyapi_config.os.environ.clear()
            tabbyapi_config.os.environ.update(old_env)

    def test_skill_shortcut_arguments(self) -> None:
        tool, arguments = skill_mcp.shortcut_arguments("search", ["foo", "bar"])
        self.assertEqual("search", tool)
        self.assertEqual({"query": "foo bar"}, arguments)
        tool, arguments = skill_mcp.shortcut_arguments("refs", ["Widget"])
        self.assertEqual("refs", tool)
        self.assertEqual({"symbol": "Widget"}, arguments)

    def test_job_telemetry_summary_includes_reward_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            trial = job / "trial-1"
            (trial / "telemetry").mkdir(parents=True)
            (trial / "verifier").mkdir()
            (trial / "verifier" / "reward.txt").write_text("1\n")
            (trial / "telemetry" / "summary.json").write_text(
                json.dumps({"mcp_call_count": 2, "changed_file_count": 1})
            )

            deepswe_harness.write_job_telemetry_summary(job)
            summary = json.loads((job / "telemetry-summary.json").read_text())
            self.assertEqual(1, summary["totals"]["trial_count"])
            self.assertEqual(2, summary["totals"]["mcp_call_count"])
            self.assertEqual(1, summary["totals"]["passed_trial_count"])
            self.assertEqual("passed", summary["trials"][0]["failure_class"])
            self.assertIn("transition_analysis", summary["trials"][0])

    def test_transition_analysis_flags_repeated_no_edit_qwen_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen-sverklo.jsonl"
            events = [
                {"event": "llm_response", "usage": {"prompt_tokens": 81000}},
            ]
            for _ in range(41):
                events.append(
                    {
                        "event": "tool_call",
                        "name": "read_file",
                        "arguments": {
                            "path": "src/main.ts",
                            "start_line": 1,
                            "line_count": 100,
                        },
                    }
                )
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

            summary = transition_analysis.analyze_qwen_jsonl(path)
            self.assertEqual(41, summary["step_count"])
            self.assertEqual(41, summary["max_same_action_run"])
            self.assertIn("identical_action_loop", summary["failure_signals"])
            self.assertIn("no_edit_by_step_40", summary["failure_signals"])
            self.assertGreater(summary["fail_score"], 0.5)

    def test_job_summary_aggregates_transition_failure_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            trial = job / "trial-loop"
            (trial / "agent").mkdir(parents=True)
            (trial / "verifier").mkdir()
            (trial / "verifier" / "reward.txt").write_text("0\n")
            events = [
                {
                    "event": "tool_call",
                    "name": "read_file",
                    "arguments": {"path": "src/main.ts"},
                }
                for _ in range(40)
            ]
            (trial / "agent" / "qwen-sverklo.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )

            deepswe_harness.write_job_telemetry_summary(job)
            summary = json.loads((job / "telemetry-summary.json").read_text())
            self.assertGreaterEqual(
                summary["totals"]["failure_signal:identical_action_loop"], 1
            )
            self.assertIn(
                "identical_action_loop",
                summary["trials"][0]["transition_analysis"]["failure_signals"],
            )

    def test_job_summary_aggregates_deterministic_critic_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            trial = job / "trial-sqlglot"
            (trial / "agent").mkdir(parents=True)
            (trial / "verifier").mkdir()
            (trial / "verifier" / "reward.txt").write_text("0\n")
            events = [
                {
                    "event": "autofix_broker",
                    "result": {
                        "ok": False,
                        "critics": {
                            "sqlglot": {
                                "available": True,
                                "analyzed_count": 1,
                                "ok": False,
                                "findings": [
                                    {
                                        "path": "query.sql",
                                        "ok": False,
                                        "errors": ["bigquery: ParseError"],
                                    }
                                ],
                            }
                        },
                    },
                }
            ]
            (trial / "agent" / "qwen-sverklo.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )

            deepswe_harness.write_job_telemetry_summary(job)
            summary = json.loads((job / "telemetry-summary.json").read_text())
            signal = "failure_signal:deterministic_critic_failure:sqlglot"
            self.assertEqual(1, summary["totals"][signal])
            analysis = summary["trials"][0]["transition_analysis"]
            self.assertIn(
                "deterministic_critic_failure:sqlglot",
                analysis["failure_signals"],
            )
            self.assertEqual("sqlglot", analysis["deterministic_critic_findings"][0]["critic"])

    def test_deterministic_critic_note_reports_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            job.mkdir()
            (job / "telemetry-summary.json").write_text(
                json.dumps(
                    {
                        "totals": {
                            "failure_signal:deterministic_critic_failure:sqlglot": 2,
                            "failure_signal:deterministic_critic_failure:ruff": 1,
                        }
                    }
                )
            )

            note = deepswe_harness.deterministic_critic_note(job)

        self.assertIn("critic:sqlglot=2", note)
        self.assertIn("critic:ruff=1", note)

    def test_qwen_sverklo_sampler_defaults_and_overrides(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            for key in list(qwen_sverklo_worker.os.environ):
                if key.startswith("QSA_"):
                    qwen_sverklo_worker.os.environ.pop(key)
            defaults = qwen_sverklo_worker.sampler_options()
            self.assertEqual(2048, defaults["max_tokens"])
            self.assertEqual(1.0, defaults["dry_multiplier"])
            self.assertEqual(0, defaults["mirostat"])
            self.assertEqual(1.12, defaults["repeat_penalty"])

            qwen_sverklo_worker.os.environ["QSA_DRY_MULTIPLIER"] = "0.45"
            qwen_sverklo_worker.os.environ["QSA_MIROSTAT"] = "0"
            qwen_sverklo_worker.os.environ["QSA_REPEAT_LAST_N"] = "1024"
            overrides = qwen_sverklo_worker.sampler_options()
            self.assertEqual(0.45, overrides["dry_multiplier"])
            self.assertEqual(0, overrides["mirostat"])
            self.assertEqual(1024, overrides["repeat_last_n"])
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_policy_denies_excess_broad_search(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_MAX_BROAD_SVERKLO_BEFORE_EDIT"] = "1"
            qwen_sverklo_worker.os.environ["QSA_FIRST_EDIT_STEP"] = "20"
            policy = qwen_sverklo_worker.AgentPolicy()
            allowed, reason = policy.allow_tool(1, "sverklo_search", {})
            self.assertTrue(allowed)
            self.assertEqual("", reason)
            policy.note_tool(1, "sverklo_search", {"query": "x"}, "result")
            allowed, reason = policy.allow_tool(2, "sverklo_overview", {})
            self.assertFalse(allowed)
            self.assertIn("exploration budget exhausted", reason)
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_policy_enforces_first_edit_deadline(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_FIRST_EDIT_STEP"] = "3"
            policy = qwen_sverklo_worker.AgentPolicy()

            allowed, reason = policy.allow_tool(3, "read_file", {"path": "x.py"})
            self.assertFalse(allowed)
            self.assertIn("first-edit deadline", reason)

            allowed, reason = policy.allow_tool(
                3, "run_shell", {"command": "grep -n foo x.py"}
            )
            self.assertFalse(allowed)
            self.assertIn("first-edit deadline", reason)

            allowed, reason = policy.allow_tool(
                3, "run_shell", {"command": "sed -i 's/a/b/' x.py"}
            )
            self.assertTrue(allowed)
            self.assertEqual("", reason)

            allowed, reason = policy.allow_tool(
                3,
                "serena_replace_symbol_body",
                {"relative_path": "x.py", "name_path": "f", "body": "def f(): pass"},
            )
            self.assertTrue(allowed)
            self.assertEqual("", reason)
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_policy_clamps_read_file_lines(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_READ_FILE_MAX_LINES"] = "80"
            policy = qwen_sverklo_worker.AgentPolicy()
            arguments = {"path": "x.py", "line_count": 500}
            allowed, _ = policy.allow_tool(1, "read_file", arguments)
            self.assertTrue(allowed)
            self.assertEqual(80, arguments["line_count"])
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_policy_read_summary_keeps_excerpt(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_READ_SUMMARY_CHARS"] = "80"
            policy = qwen_sverklo_worker.AgentPolicy()
            policy.note_tool(
                1,
                "read_file",
                {"path": "src/styles.ts", "start_line": 1, "line_count": 20},
                "1: export type Styles = { display?: 'flex' | 'none' }",
            )
            self.assertIn("display", policy.summary_lines[-1])
            self.assertIn("src/styles.ts", policy.summary_lines[-1])
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_serena_tool_selection_and_guidance(self) -> None:
        tools = [
            {"name": "find_symbol"},
            {"name": "replace_symbol_body"},
            {"name": "execute_shell_command"},
        ]
        selected = qwen_sverklo_worker.selected_mcp_tools("serena", tools)
        self.assertEqual(["find_symbol", "replace_symbol_body"], [tool["name"] for tool in selected])
        guidance = qwen_sverklo_worker.startup_guidance()
        self.assertIn("Patch discipline", guidance)
        self.assertIn("Caveman mode", guidance)
        self.assertIn("Serena skill", guidance)

    def test_qwen_sverklo_policy_repeated_denial_sets_abort(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_MAX_REPEATED_DENIALS"] = "2"
            policy = qwen_sverklo_worker.AgentPolicy()
            policy.note_denial("read_file", {"path": "x.py"}, "blocked")
            self.assertEqual("", policy.abort_reason)
            policy.note_denial("read_file", {"path": "x.py"}, "blocked")
            self.assertIn("repeated policy denial", policy.abort_reason)
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_qwen_sverklo_policy_denies_interactive_editor_and_premature_finish(self) -> None:
        policy = qwen_sverklo_worker.AgentPolicy()

        allowed, reason = policy.allow_tool(1, "run_shell", {"command": "vi test/index.ts"})
        self.assertFalse(allowed)
        self.assertIn("interactive editors", reason)

        allowed, reason = policy.allow_tool(2, "finish", {})
        self.assertFalse(allowed)
        self.assertIn("no implementation edit", reason)

        policy.edit_seen = True
        allowed, reason = policy.allow_tool(3, "finish", {})
        self.assertFalse(allowed)
        self.assertIn("git diff", reason)

        policy.diff_seen = True
        allowed, reason = policy.allow_tool(4, "finish", {})
        self.assertFalse(allowed)
        self.assertIn("validation", reason)

        policy.test_seen = True
        allowed, reason = policy.allow_tool(5, "finish", {})
        self.assertTrue(allowed)
        self.assertEqual("", reason)

    def test_qwen_sverklo_policy_early_stops_repeated_action(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_EARLY_STOP"] = "1"
            qwen_sverklo_worker.os.environ["QSA_LOOP_ABORT_REPEATS"] = "3"
            qwen_sverklo_worker.os.environ["QSA_FAIL_SCORE_ABORT"] = "0.4"
            policy = qwen_sverklo_worker.AgentPolicy()
            for step in range(1, 4):
                policy.note_tool(
                    step,
                    "read_file",
                    {"path": "src/main.ts", "start_line": 1},
                    "content",
                )
            self.assertIn("early-stop", policy.abort_reason)
            self.assertIn("same tool action repeated", policy.abort_reason)
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_autofix_broker_plans_changed_file_fixers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess_run = __import__("subprocess").run
            subprocess_run(["git", "init"], cwd=root, check=True, stdout=__import__("subprocess").PIPE)
            subprocess_run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess_run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "node_modules" / ".bin").mkdir(parents=True)
            prettier = root / "node_modules" / ".bin" / "prettier"
            prettier.write_text("#!/bin/sh\nexit 0\n")
            prettier.chmod(0o755)
            (root / ".prettierrc").write_text("{}\n")
            (root / "main.js").write_text("const x=1\n")
            subprocess_run(["git", "add", "."], cwd=root, check=True)
            subprocess_run(["git", "commit", "-m", "base"], cwd=root, check=True, stdout=__import__("subprocess").PIPE)
            (root / "main.js").write_text("const x=2\n")

            paths = autofix_broker.changed_files(root)
            plan = autofix_broker.command_plan(root, paths)

            self.assertEqual(["main.js"], autofix_broker.rels(root, paths))
            self.assertEqual("prettier-write", plan[0][0])
            self.assertIn("main.js", plan[0][1])

    def test_autofix_broker_ignores_cache_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess_run = __import__("subprocess").run
            subprocess_run(["git", "init"], cwd=root, check=True, stdout=__import__("subprocess").PIPE)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"\x00cache")
            (root / "src.py").write_text("x = 1\n")

            paths = autofix_broker.changed_files(root)

        self.assertEqual(["src.py"], autofix_broker.rels(root, paths))

    def test_ruff_critic_reports_changed_python_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "node_modules" / ".bin").mkdir(parents=True)
            ruff = root / "node_modules" / ".bin" / "ruff"
            ruff.write_text(
                "#!/bin/sh\n"
                "printf '[{\"filename\":\"bad.py\",\"code\":\"F401\",\"message\":\"unused import\"}]'\n"
                "exit 1\n"
            )
            ruff.chmod(0o755)
            bad = root / "bad.py"
            bad.write_text("import os\n")

            summary = autofix_broker.ruff_critic(root, [bad], 30, 1000)

        self.assertTrue(summary["available"])
        self.assertFalse(summary["ok"])
        self.assertEqual(1, summary["analyzed_count"])
        self.assertEqual("F401", summary["findings"][0]["code"])

    def test_shellcheck_critic_reports_changed_shell_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "node_modules" / ".bin").mkdir(parents=True)
            shellcheck = root / "node_modules" / ".bin" / "shellcheck"
            shellcheck.write_text(
                "#!/bin/sh\n"
                "printf '[{\"file\":\"bad.sh\",\"code\":2086,\"message\":\"Double quote\"}]'\n"
                "exit 1\n"
            )
            shellcheck.chmod(0o755)
            bad = root / "bad.sh"
            bad.write_text("#!/bin/sh\necho $x\n")

            summary = autofix_broker.shellcheck_critic(root, [bad], 30, 1000)

        self.assertTrue(summary["available"])
        self.assertFalse(summary["ok"])
        self.assertEqual(1, summary["analyzed_count"])
        self.assertEqual(2086, summary["findings"][0]["code"])

    def test_patch_shape_critic_hard_fails_tests_only_and_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "tests" / "test_widget.py"
            test_file.parent.mkdir()
            test_file.write_text("def test_widget(): pass\n")
            source_file = root / "src" / "widget.py"
            source_file.parent.mkdir()
            source_file.write_text("def widget(): return 1\n")

            empty = autofix_broker.patch_shape_critic(root, [], 1000)
            tests_only = autofix_broker.patch_shape_critic(root, [test_file], 1000)
            source = autofix_broker.patch_shape_critic(root, [source_file, test_file], 1000)

        self.assertFalse(empty["ok"])
        self.assertEqual("no_changed_files", empty["findings"][0]["kind"])
        self.assertFalse(tests_only["ok"])
        self.assertEqual("tests_only_or_non_impl_patch", tests_only["findings"][0]["kind"])
        self.assertTrue(source["ok"])
        self.assertEqual(1, source["implementation_count"])

    def test_policy_summary_marks_config_and_patch_shape_as_hard_failures(self) -> None:
        critics = {
            "patch_shape": {
                "available": True,
                "analyzed_count": 0,
                "ok": False,
                "findings": [{"kind": "no_changed_files"}],
            },
            "semgrep": {
                "available": False,
                "analyzed_count": 3,
                "ok": True,
                "warning": "skipped",
            },
        }

        policy = autofix_broker.policy_summary(critics, [])

        self.assertTrue(policy["hard_fail"])
        self.assertEqual("patch_shape", policy["hard_failures"][0]["critic"])
        self.assertEqual("semgrep", policy["warnings"][0]["critic"])

    def test_typecheck_critic_runs_available_pyright(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "node_modules" / ".bin").mkdir(parents=True)
            pyright = root / "node_modules" / ".bin" / "pyright"
            pyright.write_text("#!/bin/sh\nprintf '{\"summary\":{\"errorCount\":0}}'\nexit 0\n")
            pyright.chmod(0o755)
            source = root / "app.py"
            source.write_text("x: int = 1\n")

            summary = autofix_broker.typecheck_critic(root, [source], 30, 1000)

        self.assertTrue(summary["available"])
        self.assertTrue(summary["ok"])
        self.assertEqual("pyright", summary["findings"][0]["name"])

    def test_config_syntax_critic_reports_parse_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.json"
            bad_json = root / "bad.json"
            bad_toml = root / "bad.toml"
            bad_yaml = root / "bad.yaml"
            valid.write_text("{\"ok\": true}\n")
            bad_json.write_text("{bad json}\n")
            bad_toml.write_text("[broken\n")
            bad_yaml.write_text("key: [unterminated\n")

            summary = autofix_broker.config_syntax_critic(
                root, [valid, bad_json, bad_toml, bad_yaml], 1000
            )

        self.assertFalse(summary["ok"])
        self.assertEqual(4, summary["analyzed_count"])
        failed = {
            finding["path"]: finding
            for finding in summary["findings"]
            if not finding["ok"]
        }
        expected = {"bad.json", "bad.toml"}
        if importlib.util.find_spec("yaml") is not None:
            expected.add("bad.yaml")
        self.assertEqual(expected, set(failed))
        self.assertIn("JSONDecodeError", failed["bad.json"]["error"])
        self.assertIn("TOMLDecodeError", failed["bad.toml"]["error"])

    def test_text_hygiene_critic_reports_conflict_markers_and_nul_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conflict = root / "conflict.py"
            binary = root / "bad.bin"
            clean = root / "clean.txt"
            conflict.write_text("<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> branch\n")
            binary.write_bytes(b"abc\x00def")
            clean.write_text("plain text\n")

            summary = autofix_broker.text_hygiene_critic(
                root, [conflict, binary, clean], 1000
            )

        self.assertFalse(summary["ok"])
        self.assertEqual(3, summary["analyzed_count"])
        failed = {
            finding["path"]: finding
            for finding in summary["findings"]
            if not finding["ok"]
        }
        self.assertEqual({"conflict.py", "bad.bin"}, set(failed))
        self.assertEqual("conflict_marker", failed["conflict.py"]["kind"])
        self.assertEqual("nul_byte", failed["bad.bin"]["kind"])

    def test_sqlglot_critic_requires_dependency(self) -> None:
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "sqlglot":
                raise ModuleNotFoundError("No module named 'sqlglot'")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.__import__", guarded_import):
            summary = autofix_broker.sqlglot_critic(Path(tmp), [], 1000)

        self.assertFalse(summary["available"])
        self.assertFalse(summary["ok"])
        self.assertIn("required sqlglot unavailable", summary["error"])

    def test_sqlglot_critic_reports_parse_success(self) -> None:
        class FakeExpression:
            def sql(self, dialect=None):
                return f"SELECT 1 /* {dialect} */"

        fake_sqlglot = types.SimpleNamespace(parse=lambda text, read=None: [FakeExpression()])
        old_module = sys.modules.get("sqlglot")
        try:
            sys.modules["sqlglot"] = fake_sqlglot
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                query = root / "query.sql"
                query.write_text("select 1")
                summary = autofix_broker.sqlglot_critic(root, [query], 1000)
        finally:
            if old_module is None:
                sys.modules.pop("sqlglot", None)
            else:
                sys.modules["sqlglot"] = old_module

        self.assertTrue(summary["available"])
        self.assertTrue(summary["ok"])
        self.assertEqual(1, summary["analyzed_count"])
        self.assertEqual("bigquery", summary["findings"][0]["dialect"])

    @unittest.skipIf(
        importlib.util.find_spec("sqlglot") is None,
        "real sqlglot package is not installed in this interpreter",
    )
    def test_sqlglot_critic_real_fixture_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.sql"
            invalid = root / "invalid.sql"
            valid.write_text("select 1 as value\n")
            invalid.write_text("select from\n")

            valid_summary = autofix_broker.sqlglot_critic(root, [valid], 1000)
            invalid_summary = autofix_broker.sqlglot_critic(root, [invalid], 1000)

        self.assertTrue(valid_summary["available"])
        self.assertTrue(valid_summary["ok"])
        self.assertEqual(1, valid_summary["analyzed_count"])
        self.assertFalse(invalid_summary["ok"])
        self.assertEqual(1, invalid_summary["analyzed_count"])
        self.assertFalse(invalid_summary["findings"][0]["ok"])
        self.assertIn("ParseError", "\n".join(invalid_summary["findings"][0]["errors"]))

    def test_trace_scorer_classifies_no_edit_policy_abort(self) -> None:
        events = [
            {"event": "tool_call", "name": "read_file", "arguments": {"path": "a.py"}},
            {"event": "tool_call", "name": "read_file", "arguments": {"path": "a.py"}},
            {"event": "tool_call", "name": "read_file", "arguments": {"path": "a.py"}},
            {
                "event": "tool_denied",
                "name": "read_file",
                "reason": "first-edit deadline reached; no more inspection before a concrete edit",
            },
            {"event": "policy_abort", "reason": "repeated post-deadline inspection"},
        ]
        summary = trace_scorer.score_events(events)

        self.assertIn("no_edit", summary["failure_signals"])
        self.assertIn("post_deadline_inspection", summary["failure_signals"])
        self.assertIn("repeated_tool_loop", summary["failure_signals"])
        self.assertGreaterEqual(summary["fail_score"], 0.7)

    def test_qwen_sverklo_autofix_broker_reports_missing_script_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = json.loads(qwen_sverklo_worker.run_autofix_broker(Path(tmp)))
            self.assertIn("ok", result)

    def test_qwen_sverklo_autofix_after_diff_trigger_is_opt_in(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            for key in list(qwen_sverklo_worker.os.environ):
                if key.startswith("QSA_AUTOFIX_"):
                    qwen_sverklo_worker.os.environ.pop(key)
            policy = qwen_sverklo_worker.AgentPolicy()
            policy.edit_seen = True
            self.assertFalse(
                qwen_sverklo_worker.should_run_autofix_after_shell(policy, "git diff")
            )

            qwen_sverklo_worker.os.environ["QSA_AUTOFIX_AFTER_DIFF"] = "1"
            self.assertTrue(
                qwen_sverklo_worker.should_run_autofix_after_shell(
                    policy, "git diff -- src/query.sql"
                )
            )
            self.assertFalse(
                qwen_sverklo_worker.should_run_autofix_after_shell(
                    policy, "git status --short"
                )
            )
            policy.autofix_runs = 1
            self.assertFalse(
                qwen_sverklo_worker.should_run_autofix_after_shell(policy, "git diff")
            )
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_autofix_fixture_writes_trace_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            payload = {
                "ok": False,
                "critics": {
                    "sqlglot": {
                        "available": True,
                        "analyzed_count": 1,
                        "ok": False,
                        "findings": [{"path": "query.sql", "ok": False}],
                    }
                },
            }

            autofix_fixture.write_fixture_job(job, payload)
            note = deepswe_harness.deterministic_critic_note(job)
            trace_exists = (
                job / "trial-autofix-fixture" / "agent" / "qwen-sverklo.jsonl"
            ).exists()

        self.assertEqual("critic:sqlglot=1", note)
        self.assertTrue(trace_exists)

    def test_qwen_sverklo_compaction_keeps_valid_recent_prefix(self) -> None:
        old_env = dict(qwen_sverklo_worker.os.environ)
        try:
            qwen_sverklo_worker.os.environ["QSA_RECENT_MESSAGE_COUNT"] = "3"
            policy = qwen_sverklo_worker.AgentPolicy()
            messages = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
                {"role": "tool", "tool_call_id": "a", "content": "old"},
                {"role": "assistant", "content": "next"},
                {"role": "tool", "tool_call_id": "orphan-if-first", "content": "x"},
                {"role": "assistant", "content": "latest"},
            ]
            compacted = policy.compact_messages(messages, 22)
            self.assertEqual("system", compacted[0]["role"])
            self.assertEqual("user", compacted[1]["role"])
            self.assertNotEqual("tool", compacted[2]["role"])
            self.assertIn("CONTROL:", compacted[-1]["content"])
        finally:
            qwen_sverklo_worker.os.environ.clear()
            qwen_sverklo_worker.os.environ.update(old_env)

    def test_result_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "n_total_trials": 1,
                        "finished_at": "now",
                        "stats": {
                            "n_completed_trials": 1,
                            "n_errored_trials": 0,
                            "n_cancelled_trials": 0,
                            "evals": {"x": {"reward_stats": {"reward": {"1": ["t"]}}}},
                        },
                    }
                )
            )
            summary = deepswe_harness.summarize_result_json(path)
            self.assertEqual("1", summary["passed_trials"])
            self.assertEqual("100.0", summary["pass_rate_pct"])


if __name__ == "__main__":
    unittest.main()
