"""Synthetic schedule and score-gate regressions for both draft splits."""

import pytest

from scripts import prepare_native_state_recovery_comparison_001 as comparison

TASKS = [f"{index:07x}_{variant}" for index in range(1, 10) for variant in (1, 2, 3)]


def _draft(split):
    return {
        **comparison._layout(TASKS, ["abcdef0"], seed=17),
        "split": split, "model_id": "scripted",
        "protocol": {"require_complete_planned_pairs": True, "bootstrap_samples": 2000,
                     "bootstrap_seed": 0, "minimum_scenarios": 8, "maximum_harm_rate": .05,
                     "both_lower_bounds_positive": True},
    }


def _scores(values):
    return [[index, arm, value] for index, pair in enumerate(values)
            for arm, value in zip(comparison.ARMS, pair, strict=True)]


def test_frozen_layout_covers_each_pair_once_and_rejects_origin_overlap():
    draft = _draft("authored")
    assert len(draft["batches"]) == 14
    assert [index for batch in draft["batches"] for index in batch] == list(range(27))
    assert sum(order[0] == "experience" for order in draft["arm_orders"]) == 13
    with pytest.raises(ValueError):
        comparison._layout(TASKS, ["0000001"], seed=17)


@pytest.mark.parametrize("split", ["authored", "test_normal"])
def test_missing_or_unknown_branch_disables_the_full_comparison(split):
    scores = _scores([(False, True)] * 17) + [[17, "experience", None]]
    result = comparison.summarize(_draft(split), scores)
    assert result == {"status": "not_evaluable", "planned_pairs": 27, "complete_pairs": 17,
                      "known_branches": 34, "split": split, "passed": None, "statistics": None}


@pytest.mark.parametrize(
    ("values", "expected", "harmed"),
    [
        ([(False, True)] * 27, True, 0),
        ([(False, True)] * 25 + [(True, False)] * 2, False, 2),
        ([(False, False)] * 27, False, 0),
    ],
)
def test_complete_panels_apply_both_intervals_and_observed_harm_gate(values, expected, harmed):
    result = comparison.summarize(_draft("authored"), _scores(values))
    assert result["status"] == "complete" and result["passed"] is expected
    assert result["statistics"]["harmed"] == harmed
    assert result["statistics"]["exact_resample_count"] == 9**9


@pytest.mark.parametrize("scores", [[[0, "control", 1]], [[0, "control", True]] * 2])
def test_rejects_non_boolean_scores_and_duplicate_slots(scores):
    with pytest.raises(ValueError):
        comparison.summarize(_draft("authored"), scores)
