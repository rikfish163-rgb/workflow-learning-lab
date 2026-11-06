import json

import pytest
from test_experience import lesson
from test_runner import Model, World

from wll.protocol import Task
from wll.utility import validate_candidate


def test_utility_records_all_pairs_and_rejects_tie(tmp_path):
    tasks = [Task(f"s{i}_1", f"s{i}", "train", "inc") for i in range(8)]
    output = tmp_path / "utility"
    result = validate_candidate(lesson(), tasks, lambda task: World(), Model(),
                                output=output, max_steps=2, seed=17,
                                environment_id="test-double", min_scenarios=8)
    assert result["n_pairs"] == 8
    assert not result["accepted"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["thresholds"]["min_scenarios"] == 8
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert sum(e["experience_id"] is not None for e in rows) == 8


def test_utility_rejects_origin_and_test_tasks_before_allocating(tmp_path):
    def fail(task):
        raise AssertionError("must not run")

    for task in [Task("origin_1", "origin", "train", "x"),
                 Task("new_1", "new", "dev", "x")]:
        with pytest.raises(ValueError, match="independent train"):
            validate_candidate(lesson(), [task], fail, Model(), output=tmp_path / "bad",
                               max_steps=2, seed=0, environment_id="mock")
    assert not (tmp_path / "bad").exists()


def test_utility_infrastructure_failure_is_incomplete_and_keeps_cost(tmp_path):
    class Broken(Model):
        calls = 0

        def generate(self, messages, seed):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transport")
            return super().generate(messages, seed)

    output = tmp_path / "incomplete"
    with pytest.raises(RuntimeError):
        validate_candidate(lesson(), [Task("x_1", "x", "train", "inc")],
                           lambda task: World(), Broken(), output=output,
                           max_steps=2, seed=0, environment_id="mock")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["cost"]["completion_tokens"] == 1
    assert not (output / "report.json").exists()
