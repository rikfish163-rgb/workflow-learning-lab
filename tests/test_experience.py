import pytest

from wll.experience import Experience, validate_experience
from wll.protocol import Episode, Task


def ep(scenario, success, split="train", branch="first", exp=None):
    return Episode(Task(scenario + "_1", scenario, split, "instruction"), "model", 1,
                   [], success, "completed", experience_id=exp, branch=branch)


def lesson():
    return Experience("candidate-1", "pagination", "multiple pages", "single page",
                      "Iterate until no next page", ["origin"], ["trace-hash"])


def test_source_scenario_is_not_independent_validation():
    with pytest.raises(ValueError, match="origin"):
        validate_experience(lesson(), [ep("origin", False)],
                            [ep("origin", True, exp="candidate-1")])


def test_test_feedback_cannot_promote_a_lesson():
    with pytest.raises(ValueError, match="train"):
        validate_experience(lesson(), [ep("new", False, "test_challenge")],
                            [ep("new", True, "test_challenge", exp="candidate-1")])


def test_validation_must_compare_first_attempts_with_correct_treatment():
    with pytest.raises(ValueError):
        validate_experience(lesson(), [ep("new", False)],
                            [ep("new", True, branch="recovery", exp="candidate-1")])


def test_lesson_with_help_and_harm_is_not_accepted_on_positive_cases_alone():
    control = [ep("a", False), ep("b", True)]
    treatment = [ep("a", True, exp="candidate-1"), ep("b", False, exp="candidate-1")]
    report = validate_experience(lesson(), control, treatment, min_scenarios=2)
    assert not report["accepted"]
    assert report["harmed"] == 1


def test_extra_budget_cannot_be_misattributed_to_an_experience():
    control, treatment = ep("new", False), ep("new", True, exp="candidate-1")
    control.metadata["max_steps"] = 10
    treatment.metadata["max_steps"] = 20
    with pytest.raises(ValueError, match="budget"):
        validate_experience(lesson(), [control], [treatment])
