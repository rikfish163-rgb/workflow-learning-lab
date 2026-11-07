"""Verify the official routing and stopping rules with synthetic evidence only."""

from pathlib import Path

import pytest
from test_run_native_experience_utility_001 import SHA, Runtime

from scripts import run_native_state_recovery_official_001 as study
from scripts import run_native_state_recovery_official_002 as successor


@pytest.mark.parametrize("failure, complete_pairs", [(None, 27), ("scorer", 0), ("release_last", 27)])
def test_official_pair_lifecycle(monkeypatch, failure, complete_pairs):
    runtime = Runtime(failure=failure)
    plan = runtime.plan
    plan.update(id=study.ID, dev_or_test_authorized=True)
    for value in (plan["source"], runtime.bundle):
        value["source_gate_kind"] = study.review.SOURCE_GATE_KIND
        value["experience_bundle_id"] = study.review.BUNDLE_PREFIX + study._hash(runtime.bundle["insertion_text"].encode())
    plan["source"]["qualification_proof"] = "synthetic-proof.json"
    plan["utility"]["split"] = "test_normal"
    plan["draft"] = {"expected_pairs": plan["expected_pairs"], "protocol": plan["utility"],
                     "split": "test_normal", "model_id": plan["model_id"]}
    monkeypatch.setattr(study.frozen, "_experiment", study._experiment)
    result = study.run_study(SHA, runtime=runtime)
    assert result["utility"]["planned_pairs"] == 27
    assert result["utility"]["complete_pairs"] == complete_pairs
    assert all(row["split"] == "test_normal" for row in result["summaries"])
    assert result["writer_calls"] == result["training_jobs"] == 0
    assert result["dev_or_test_calls"] == (1 if failure == "scorer" else 54)
    for index, call in enumerate(runtime.calls):
        if call[:2] == ("launch", "scorer"):
            args = call[3]
            receipt = args[args.index("--receipt") + 1]
            assert ("worker_closed", str(Path(receipt).parent)) in runtime.calls[:index]
    assert [call[1] for call in runtime.calls if call[0] == "release"] == list(
        range(1 if failure == "scorer" else 14))
    assert (result["utility"]["statistics"] is not None) == (failure is None)


def test_commands_use_official_scorer_and_existing_limits():
    for kind, mode, module, duration in (
        ("pilot", "--run", study.MODULE, study.frozen.PARENT_SECONDS),
        ("episode", "--config", study.MODULE, 900),
        ("scorer", "--receipt", study.scorer.__name__, 120),
    ):
        argv = study._unit_command(kind, study._coordinator_unit(SHA), [mode, "value"],
                                   root=study.WORKSPACE, log_path=Path("/synthetic/log"))
        assert f"--property=RuntimeMaxSec={duration}" in argv
        assert any(f"run_module('{module}'" in part for part in argv)
        assert argv.count("--episode") == (kind == "episode")
    assert study.run_episode.__kwdefaults__ == study.frozen.run_episode.__kwdefaults__


def test_live_authored_coordinator_blocks_official_launch(monkeypatch):
    def main_pid(unit):
        assert unit == study.AUTHORED_UNIT
        return 100

    monkeypatch.setattr(study.scorer, "_main_pid", main_pid)
    with pytest.raises(study.frozen.NativeExperienceUtilityError):
        study.launch(SHA)


def test_successor_service_and_health_probe_use_the_registered_port(monkeypatch):
    core = successor.bindings()
    prior = {"port": 18999, "model_arm": "B3", "model_id": "synthetic-model"}
    monkeypatch.setattr(core, "_prior", lambda: prior)
    core._queue.cache_clear()
    try:
        service = core._queue().serve.__globals__["registration"]
        assert service.PORT == core._registry().PORT == prior["port"]
    finally:
        core._queue.cache_clear()
