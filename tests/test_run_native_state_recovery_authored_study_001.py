"""Exercise the actual authored coordinator against a synthetic runtime."""

from pathlib import Path

import pytest
from test_run_native_experience_utility_001 import SHA, Runtime

from scripts import run_native_experience_utility_001 as old
from scripts import run_native_state_recovery_authored_study_001 as study
from scripts.native_appworld_worker_client import _decode, _encode


class AuthoredRuntime(Runtime):
    def __init__(self, *, failure=None):
        super().__init__(failure=failure)
        self.plan["id"] = study.ID
        self.plan["draft"] = {"expected_pairs": self.plan["expected_pairs"], "protocol": self.plan["utility"],
                              "split": "authored", "model_id": self.plan["model_id"],
                              "initial_state_manifest_sha256": "f" * 64}

    def launch(self, kind, unit, arguments, *, log_name):
        if kind == "episode":
            path = arguments[arguments.index("--config") + 1]
            config = _decode(self.files[path])
            experiment = Path(path).stem
            config["run_dir"] = f"/synthetic/private/runs/{experiment}"
            self.files[path] = _encode(config) + b"\n"
        return super().launch(kind, unit, arguments, log_name=log_name)

    def unit_status(self, spec):
        if self.failure == "branch" and spec["kind"] == "episode" and spec["index"] == 17:
            return {"verified": True, "running": False, "main_pid": 0, "exit_code": 1, "success": False}
        return super().unit_status(spec)

    def score_result(self, path):
        arm = Path(path).name.split("-")[2]
        run = "/synthetic/private/runs/" + Path(path).name
        return {"status": "scored", "success": arm == "experience",
                "receipt_sha256": self.evidence[run]["receipt_binding"]["sha256"],
                "manifest_sha256": "e" * 64, "is_policy_episode": True}


@pytest.fixture(autouse=True)
def bindings(monkeypatch):
    monkeypatch.setattr(old, "_experiment", lambda index, arm, owner: f"pair-{index:02d}-{arm}-{owner}")

    def config(runtime, _root, name, value):
        binding = runtime.write(name, value)
        return binding["path"], binding["sha256"]

    monkeypatch.setattr(study, "_config", config)


def test_all_54_branches_use_the_authored_path_and_release_14_times():
    runtime = AuthoredRuntime()
    result = study.run_study(SHA, runtime=runtime)
    assert result["status"] == "complete"
    assert result["utility"]["complete_pairs"] == 27
    assert result["utility"]["global_execution_and_costs_verified"] is True
    assert len(result["executions"]) == 54
    assert all(row["split"] == "authored" for row in result["summaries"])
    assert [call[1] for call in runtime.calls if call[0] == "release"] == list(range(14))
    for index, call in enumerate(runtime.calls):
        if call[:2] == ("launch", "scorer"):
            args = call[3]
            receipt = args[args.index("--receipt") + 1]
            assert ("worker_closed", str(Path(receipt).parent)) in runtime.calls[:index]


def test_interruption_keeps_17_complete_pairs_and_releases_last_lease():
    runtime = AuthoredRuntime(failure="branch")
    result = study.run_study(SHA, runtime=runtime)
    assert result["status"] == "unknown"
    assert result["utility"]["planned_pairs"] == 27 and result["utility"]["complete_pairs"] == 17
    assert len(result["executions"]) == 34 and result["utility"]["statistics"] is None
    assert len(result["costs"]["episodes"]) == 35 and result["costs"]["episodes"][-1]["unknown_cost"]
    assert [call[1] for call in runtime.calls if call[0] == "release"] == list(range(9))


def test_full_scores_with_unverified_release_do_not_produce_effect():
    runtime = AuthoredRuntime(failure="release_last")
    result = study.run_study(SHA, runtime=runtime)
    assert result["status"] == "unknown" and len(result["executions"]) == 54
    assert result["utility"]["status"] == "not_evaluable" and result["utility"]["statistics"] is None


def test_queue_namespace_uses_existing_guard_and_requires_real_registration():
    queue = study._queue()
    job = queue.make_job(13, "a" * 32, SHA)
    assert queue._get_core().make_job(13, "a" * 32, SHA) == job
    assert job["max_runtime_seconds"] == 7200 and job["kind"] == "comparison"
    assert queue.GUARD_NAME == "native-state-recovery-transition-utility-003-operations-001.lock"
    with pytest.raises((ValueError, OSError)):
        queue.load_registration(study.WORKSPACE, SHA)


def test_launch_without_an_authorized_plan_never_calls_systemd(monkeypatch):
    monkeypatch.setattr(study.subprocess, "run", lambda *_a, **_kw: pytest.fail("unexpected systemd start"))
    with pytest.raises((ValueError, OSError)):
        study.launch_study(SHA)


def test_unit_commands_use_authored_entry_points_and_inherited_limits():
    for kind, mode, module, limit in (
        ("pilot", "--run", study.MODULE, study.frozen.PARENT_SECONDS),
        ("episode", "--config", study.MODULE, 900),
        ("scorer", "--receipt", study.assets.__name__, 120),
    ):
        role = "score" if kind == "scorer" else "parent"
        argv = study._unit_command(kind, f"wll-authored-{role}-{'c' * 32}.service", [mode, "value"],
                                   root=study.WORKSPACE, log_path=Path("/synthetic/log"))
        assert f"--property=RuntimeMaxSec={limit}" in argv
        assert any(f"run_module('{module}'" in part for part in argv)
        assert argv.count("--episode") == (kind == "episode")
