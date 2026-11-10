"""Check the v2-to-v3 updater and measured adaptive journals without WSL."""
import ast
import asyncio
import contextlib
import copy
import hashlib
import io
import json
import secrets
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock, AsyncMock
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import update_project2
from nebula.frontend.research_view import research_payload
from nebula.controller.container_lifecycle import start_participant, stop_scenario_participants, remove_empty_network

MEASURED = HERE / "results/smoke_v3_20261104"
BEFORE = HERE / "tests/fixtures/legacy_v2"


class AdaptiveReleaseTests(unittest.TestCase):
    def test_reviewed_sources_match_manifest_and_parse(self):
        manifest = json.loads((HERE / "provenance/update_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 26)
        for relative, row in manifest.items():
            path = HERE / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["after"], relative)
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)

    def test_updater_accepts_v2_preserves_logs_and_refuses_unknown_edits(self):
        # All mutations are confined to a new, disposable copy under test_artifacts.
        root = HERE / ".local" / "test_artifacts" / ("updater_v3_" + secrets.token_hex(4))
        shutil.copytree(BEFORE, root)
        (root / ".project2-managed").write_text("525b31c674644c653117126b5a1473a5321ac240", encoding="ascii")
        log = root / "app/logs/keep.txt"
        log.parent.mkdir(parents=True)
        log.write_text("saved experimental evidence", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            update_project2.update(root, apply=False)
        edited = root / "nebula/core/training/dp.py"
        original = edited.read_bytes()
        edited.write_bytes(original + b"\n# An unreviewed local change\n")
        with self.assertRaisesRegex(ValueError, "Unexpected local edits"):
            update_project2.update(root, apply=False)
        self.assertTrue(edited.read_bytes().endswith(b"# An unreviewed local change\n"))
        edited.write_bytes(original)
        with patch.object(update_project2.platform, "system", return_value="Linux"), \
             patch.object(update_project2.subprocess, "check_output", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            update_project2.update(root, apply=True)
            update_project2.update(root, apply=False)
        self.assertEqual(log.read_text(encoding="utf-8"), "saved experimental evidence")
        backups = list((root / "project2-backups").glob("*/nebula/core/training/dp.py"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual(edited.read_bytes(), (HERE / "nebula/core/training/dp.py").read_bytes())

    def test_measured_feedback_noise_scaling_and_accounting(self):
        payloads = {}
        for mode in ("baseline", "fixed", "adaptive"):
            name = "smoke_" + mode
            p = research_payload(MEASURED / "logs" / name, MEASURED / "config" / name,
                                 {"name": name, "status": "completed"})
            payloads[mode] = p
            self.assertEqual(p["protocol"], "project2-public-evaluation-v3")
            self.assertEqual(len(p["evaluations"]), 3)
            self.assertEqual(p["final_test"]["records"], 114)
            self.assertEqual(p["errors"], [])
            if mode != "baseline":
                self.assertTrue(p["accounting_complete"])
                self.assertEqual(len(p["privacy"]), 9)
                self.assertLessEqual(p["epsilon_max"], 8)
        for participant in (1, 2, 3):
            fixed = [r for r in payloads["fixed"]["privacy"] if r["participant"] == participant]
            rows = [r for r in payloads["adaptive"]["privacy"] if r["participant"] == participant]
            self.assertEqual(len({r["sigma"] for r in rows}), 1)
            self.assertGreater(len({r["clip"] for r in rows}), 1)
            self.assertEqual([r["epsilon"] for r in fixed], [r["epsilon"] for r in rows])
            for row in rows:
                self.assertEqual(row["adaptation"]["reference_records"], 91)
                self.assertAlmostEqual(row["noise_std_sum"], row["sigma"] * row["clip"])
                self.assertEqual(row["adaptation"]["selected_clip"], row["clip"])
        self.assertIsNone(payloads["baseline"]["epsilon_max"])
        self.assertEqual(payloads["baseline"]["comparison_key"], payloads["adaptive"]["comparison_key"])

    def test_historical_results_keep_their_original_mode_and_protocol(self):
        old = HERE / "results/history/smoke_v2_20261104"
        p = research_payload(old / "logs/smoke_dynamic", old / "config/smoke_dynamic",
                             {"name": "smoke_dynamic", "status": "completed"})
        self.assertEqual(p["dp"]["mode"], "dynamic")
        self.assertEqual(p["protocol"], "project2-public-evaluation-v2")
        self.assertTrue(p["accounting_complete"])
        self.assertNotIn("adaptation", p["privacy"][0])

    def test_container_creation_failure_never_starts_a_stale_id(self):
        api = Mock()
        api.create_container.side_effect = RuntimeError("409 name conflict")
        with self.assertRaisesRegex(RuntimeError, "409"):
            start_participant(api, name="test-participant0")
        api.start.assert_not_called()
        api.remove_container.assert_not_called()
        api.create_container.side_effect = None
        api.create_container.return_value = {"Id": "new-owned-container"}
        api.start.side_effect = RuntimeError("start failed")
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            start_participant(api, name="test-participant0")
        api.remove_container.assert_called_once_with("new-owned-container", force=True)

    def test_late_cleanup_cannot_remove_the_next_scenario(self):
        def container(name, scenario, labelled):
            return SimpleNamespace(name=name, attrs={"Config": {
                "Labels": {"nebula.scenario": scenario} if labelled else {},
                "Cmd": ["/bin/bash", "-c", f"python /nebula/app/config/{scenario}/participant_0.json"]}}, remove=Mock())
        old = container("demo_ADMIN-participant0", "old", False)
        new = container("demo_ADMIN-participant1", "new", True)
        another_user = container("demo_OTHER-participant0", "old", True)
        client = SimpleNamespace(containers=SimpleNamespace(list=Mock(return_value=[old, new, another_user])))
        stop_scenario_participants(client, "demo_ADMIN-participant", "old")
        old.remove.assert_called_once_with(force=True)
        new.remove.assert_not_called()
        another_user.remove.assert_not_called()
        network = SimpleNamespace(attrs={"Containers": {"new": {}}}, remove=Mock())
        client.networks = SimpleNamespace(get=Mock(return_value=network))
        remove_empty_network(client, "scenario-network")
        network.remove.assert_not_called()

    def test_queue_waits_for_cleanup_before_starting_next_scenario(self):
        source = ast.parse((HERE / "nebula/frontend/app.py").read_text(encoding="utf-8"))
        function = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_scenarios")
        state = SimpleNamespace(scenarios_list=[{"id": "a"}, {"id": "b"}],
            finish_scenario_event=asyncio.Event(), stop_all_scenarios_event=asyncio.Event(),
            scenarios_finished=0, scenarios_list_length=2, current_scenario=None)
        calls = []
        async def run(config, role, user):
            calls.append("start:" + config["id"])
            state.finish_scenario_event.set()
            return config["id"]
        async def cleanup(name, user):
            calls.append("clean:" + name)
        state.scenarios_list = [{"id": "a", "scenario_title": "a"}, {"id": "b", "scenario_title": "b"}]
        scope = {"asyncio": SimpleNamespace(sleep=AsyncMock()), "user_data_store": {"ADMIN": state},
                 "logging": SimpleNamespace(info=Mock()),
                 "run_scenario": run, "stop_scenario_by_name": cleanup,
                 "check_enough_resources": AsyncMock(return_value=True)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_run_scenarios", "exec"), scope)
        asyncio.run(scope["run_scenarios"]("admin", "ADMIN"))
        self.assertEqual(calls, ["start:a", "clean:a", "start:b", "clean:b"])
        self.assertEqual(state.scenarios_list_length, 0)

    def test_startup_failure_is_visible_without_training_rows(self):
        root = HERE / ".local" / "test_artifacts" / ("startup_failure_" + secrets.token_hex(4))
        root.mkdir(parents=True)
        (root / "scenario_startup_error.jsonl").write_text(json.dumps({"kind": "startup_error", "error": "409 name conflict"}) + "\n", encoding="utf-8")
        payload = research_payload(root, root, {"name": "failed", "status": "failed"})
        self.assertEqual(payload["errors"][0]["kind"], "startup_error")
        self.assertIsNone(payload["final_test"])

    def test_actual_stop_endpoint_loads_docker_and_marks_only_its_scenario(self):
        source = ast.parse((HERE / "nebula/controller/controller.py").read_text(encoding="utf-8"))
        function = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "stop_scenario")
        function = copy.deepcopy(function)
        function.decorator_list = []
        client = SimpleNamespace(containers=SimpleNamespace(list=Mock(return_value=[])),
            networks=SimpleNamespace(get=Mock(return_value=SimpleNamespace(attrs={"Containers": {}}, remove=Mock()))))
        finished, all_finished = Mock(), Mock()
        scope = {"Body": lambda *a, **k: None, "os": __import__("os"),
            "scenario_set_status_to_finished": finished, "scenario_set_all_status_to_finished": all_finished,
            "logging": SimpleNamespace(exception=Mock()), "HTTPException": RuntimeError}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_stop_scenario", "exec"), scope)
        with patch.dict(sys.modules, {"docker": SimpleNamespace(from_env=Mock(return_value=client))}), \
             patch.dict(__import__("os").environ, {"NEBULA_CONTROLLER_NAME": "demo"}):
            asyncio.run(scope["stop_scenario"]("finished-scenario", "ADMIN", False))
        finished.assert_called_once_with("finished-scenario")
        all_finished.assert_not_called()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdaptiveReleaseTests))
    if result.wasSuccessful():
        fixtures = [research_payload(MEASURED / "logs" / name, MEASURED / "config" / name,
            {"name": name, "title": name, "status": "completed"})
            for name in ("smoke_adaptive", "smoke_fixed", "smoke_baseline")]
        (HERE / ".local/test_artifacts/frontend_evidence_v3.json").write_text(json.dumps(fixtures), encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())
