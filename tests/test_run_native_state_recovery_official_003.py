"""Check fresh-scenario selection, bounded generation and the actual study loop."""

from types import SimpleNamespace

import pytest
from test_run_native_experience_utility_001 import SHA, Runtime

from scripts import freeze_native_state_recovery_official_003 as panel
from scripts import run_native_state_recovery_official_003 as entry


def test_selection_excludes_the_entire_previous_scenario():
    tasks = [f"{scene:07x}_{variant}" for scene in range(1, 21) for variant in (1, 2, 3)]
    excluded = [f"{scene:07x}" for scene in range(1, 10)]
    selected = panel.select_roster(tasks, excluded)
    assert selected == [f"{scene:07x}_{variant}" for scene in range(10, 19) for variant in (1, 2, 3)]
    with pytest.raises(ValueError):
        panel.select_roster(tasks, [f"{scene:07x}" for scene in range(1, 14)])


def test_engine_limit_retains_the_return_value_and_generation_arguments():
    calls = []
    returned = object()

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return returned

    model = SimpleNamespace(generate=generate)
    original = entry.limit_generation(model)
    assert model.generate("input", max_new_tokens=1024, max_time=1000) is returned
    assert calls == [(("input",), {"max_new_tokens": 1024, "max_time": 75.0})]
    assert original is generate


@pytest.mark.parametrize("failure, pairs", [(None, 27), ("scorer", 0), ("release_last", 27)])
def test_registered_loop_retains_scores_and_unknowns(monkeypatch, failure, pairs):
    core = entry.bindings()
    runtime = Runtime(failure=failure)
    plan = runtime.plan
    plan.update(id=entry.ID, dev_or_test_authorized=True)
    plan["utility"].update(split="test_normal", generation_max_time_seconds=entry.LIMIT)
    for source in (plan["source"], runtime.bundle):
        source["source_gate_kind"] = core.review.SOURCE_GATE_KIND
        source["experience_bundle_id"] = core.review.BUNDLE_PREFIX + core._hash(runtime.bundle["insertion_text"].encode())
    plan["draft"] = {"expected_pairs": plan["expected_pairs"], "protocol": plan["utility"],
                     "split": "test_normal", "model_id": plan["model_id"]}
    monkeypatch.setattr(core.frozen, "_experiment", core._experiment)
    result = core.run_study(SHA, runtime=runtime)
    assert result["utility"]["planned_pairs"] == 27
    assert result["utility"]["complete_pairs"] == pairs
    assert (result["utility"]["statistics"] is not None) is (failure is None)
    assert result["writer_calls"] == result["training_jobs"] == 0
    assert [call[1] for call in runtime.calls if call[0] == "release"] == list(range(1 if failure == "scorer" else 14))


def test_port_and_backend_bind_the_generation_limit(monkeypatch):
    core = entry.bindings()
    prior = {"port": 18999, "model_arm": "B3", "model_id": "synthetic",
             "expected_backend_static": {"B3": {"fixture": True}}}
    monkeypatch.setattr(core, "_prior", lambda: prior)
    core._queue.cache_clear()
    try:
        assert core._queue().PORT == core._registry().PORT == 18999
        assert core._queue()._get_core().make_job(13, "a" * 32, SHA) == core._job(13, "a" * 32, SHA)
        assert entry._backend(prior) == {"B3": {"fixture": True, "generation_max_time_seconds": 75.0}}
        assert "generation_max_time_seconds" not in prior["expected_backend_static"]["B3"]
    finally:
        core._queue.cache_clear()
