import json

import pytest
from test_runner import TASK, Model, World

from wll import recovery
from wll.protocol import Episode


def test_failed_draft_and_rollouts_remain_costed(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "AppWorldEnvironment", lambda *a: World())
    source = Episode(TASK, "test-double", 1, [], False, "step_budget")
    stats = recovery.recover_failures([source], Model(), data_dir="unused", deps_dir=None,
        image="test", output=tmp_path / "mine", max_steps=2, attempts=1)
    assert stats["recovered_tasks"] == 1
    assert stats["generation_errors"] == 1
    assert stats["lesson_generation_tokens"] == 1
    assert stats["rollout_completion_tokens"] == 2
    assert stats["status"] == "complete"


def test_partial_recovery_cost_survives_infrastructure_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "AppWorldEnvironment", lambda *a: World())

    class Broken(Model):
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transport")
            return super().generate(*args, **kwargs)

    source = Episode(TASK, "test-double", 1, [], False, "step_budget")
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError):
        recovery.recover_failures([source], Broken(), data_dir="unused", deps_dir=None,
            image="test", output=output, max_steps=2, attempts=1)
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "interrupted"
    assert report["rollout_completion_tokens"] == 1
