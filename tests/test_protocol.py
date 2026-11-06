import pytest

from wll.protocol import Episode, Step, Task, assert_disjoint, paired_estimate
from wll.training import export_first_attempt_examples


def task(task_id="scenario-a_1", split="train"):
    return Task(task_id, "scenario-a", split, "Move the item")


def episode(t=None, success=True, steps=(), experience=None, branch="first"):
    return Episode(t or task(), "model@revision", 7, list(steps), success, "completed",
                   experience_id=experience, branch=branch)


def test_scenarios_cannot_leak_across_splits_even_with_different_task_ids():
    with pytest.raises(ValueError, match="scenario"):
        assert_disjoint([task()], [task("scenario-a_2", "dev")])


def test_paired_estimate_requires_same_tasks_and_seeds():
    with pytest.raises(ValueError, match="pair"):
        paired_estimate([episode()], [])


def test_paired_estimate_reports_harm_and_help_not_only_average():
    a = [episode(), episode(Task("b_1", "b", "train", "b"), False)]
    b = [episode(success=False), episode(Task("b_1", "b", "train", "b"))]
    report = paired_estimate(a, b, bootstrap_samples=100)
    assert report["helped"] == report["harmed"] == 1
    assert report["delta"] == 0
    assert report["scenario_count"] == 2


def test_test_and_recovery_data_cannot_become_first_attempt_sft():
    step = Step("move()", "ok", 2, 1)
    for e in [episode(task(split="test_challenge"), steps=[step]),
              episode(steps=[step], branch="recovery")]:
        with pytest.raises(ValueError):
            export_first_attempt_examples([e])


def test_student_prompt_never_contains_training_only_experience():
    e = episode(steps=[Step("move()", "ok", 2, 1)], experience="lesson-1")
    rows = export_first_attempt_examples([e])
    assert len(rows) == 1
    assert "lesson-1" not in str(rows[0]["messages"])
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "move()"}
    assert rows[0]["target_role"] == "assistant"


def test_failed_trajectory_never_becomes_positive_sft():
    assert export_first_attempt_examples([episode(success=False)]) == []


def test_all_ties_are_not_presented_as_proof_of_equivalence():
    report = paired_estimate([episode(success=False)], [episode(success=False)])
    assert report["bootstrap_degenerate"]
    assert "equivalence" in report["warning"]
