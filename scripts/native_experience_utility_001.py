"""Ordinary experience versus no experience, from train-pair summaries.

The caller supplies the complete frozen expected cohort, source-scenario IDs,
shared identities, and authentic closed first-attempt summaries from a newly
registered utility study. This helper verifies structural consistency, not
registration, provenance, content safety, receipt authenticity, scenario
independence or population coverage. It runs no task, model, scorer or worker.

The comparison is ordinary model-written experience versus no experience, not
a T/U/P control claim and not an unseen/dev/test result. Ephemeral protocol Task
objects use empty instructions solely to invoke the existing paired estimator;
they are not returned, stored, or used as training data.
"""

import re
from collections import Counter

from scripts.exact_cluster_resampling import exact_cluster_resampling
from wll.protocol import Episode, Task, fingerprint, paired_estimate

MIN_SCENARIOS = 8
MAX_HARM_NUMERATOR = 1
MAX_HARM_DENOMINATOR = 20
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
_ERROR = "native_experience_utility_invalid"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_IDENTITY_FIELDS = frozenset({"model_id", "scaffold_id", "backend_id", "budget_id"})
_ROW_FIELDS = frozenset({"task_id", "scenario_id", "seed", "arm", "split", "status", "closed",
                         "branch", "replayed_steps", "success", "outcome_unknown", "unknown_cost",
                         "is_policy_episode", "worker_exit_code", "score_status", "identity",
                         "experience_bundle_id"})
_STATUSES = frozenset({"known", "unknown", "incomplete", "unstarted"})
_SCORE_STATUSES = frozenset({"scored", "unknown", "not_scored"})


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _identifier(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def _seed(value):
    return type(value) is int and 0 <= value < 2**31


def _identity(value):
    _require(type(value) is dict and set(value) == _IDENTITY_FIELDS
             and all(_identifier(item) for item in value.values()))
    return dict(value)


def _manifest(expected_pairs, source_scenario_ids):
    _require(type(expected_pairs) in (list, tuple) and bool(expected_pairs)
             and type(source_scenario_ids) in (list, tuple, set, frozenset) and bool(source_scenario_ids))
    origins = tuple(source_scenario_ids)
    _require(all(_identifier(value) for value in origins) and len(set(origins)) == len(origins))
    pairs, keys, task_scenarios = [], set(), {}
    for triple in expected_pairs:
        _require(type(triple) in (tuple, list) and len(triple) == 3)
        task, scenario, seed = triple
        _require(_identifier(task) and _identifier(scenario) and _seed(seed))
        key = task, seed
        _require(key not in keys and (task not in task_scenarios or task_scenarios[task] == scenario))
        keys.add(key)
        task_scenarios[task] = scenario
        pairs.append((task, scenario, seed))
    _require(not (set(origins) & {scenario for _, scenario, _ in pairs}))
    return pairs


def _row_status(row):
    if (row["status"] == "unknown" or row["outcome_unknown"] or row["unknown_cost"]
            or row["score_status"] == "unknown"):
        return "unknown"
    if row["status"] == "unstarted":
        return "unstarted"
    if (row["status"] == "incomplete" or not row["closed"] or row["success"] is None
            or not row["is_policy_episode"] or row["worker_exit_code"] != 0
            or row["score_status"] != "scored"):
        return "incomplete"
    return "known"


def _matrix(pairs, summaries, identity, bundle):
    _require(type(summaries) in (list, tuple))
    planned = {(task, seed): scenario for task, scenario, seed in pairs}
    rows = {}
    for row in summaries:
        _require(type(row) is dict and set(row) == _ROW_FIELDS
                 and _identifier(row["task_id"]) and _identifier(row["scenario_id"]) and _seed(row["seed"])
                 and type(row["arm"]) is str and row["arm"] in {"control", "experience"}
                 and type(row["split"]) is str and row["split"] == "train"
                 and type(row["branch"]) is str and row["branch"] == "first"
                 and type(row["replayed_steps"]) is int and row["replayed_steps"] == 0
                 and type(row["status"]) is str and row["status"] in _STATUSES
                 and type(row["closed"]) is bool and type(row["outcome_unknown"]) is bool
                 and type(row["unknown_cost"]) is bool
                 and type(row["is_policy_episode"]) is bool
                 and (type(row["worker_exit_code"]) is int or row["worker_exit_code"] is None)
                 and type(row["score_status"]) is str and row["score_status"] in _SCORE_STATUSES
                 and (type(row["success"]) is bool or row["success"] is None)
                 and _identity(row["identity"]) == identity)
        key = row["task_id"], row["seed"]
        _require(key in planned and planned[key] == row["scenario_id"])
        if row["arm"] == "control":
            _require(row["experience_bundle_id"] is None)
        else:
            _require(type(row["experience_bundle_id"]) is str and row["experience_bundle_id"] == bundle)
        slot = key, row["arm"]
        _require(slot not in rows)
        rows[slot] = row, _row_status(row)
    return rows


def evaluate_utility(expected_pairs, summaries, *, identity, source_scenario_ids, experience_bundle_id):
    """Evaluate every frozen train pair or return a full-denominator non-result.

    ``expected_pairs`` contains (task_id, scenario_id, seed) triples. Each row is
    an exact metadata projection described by ``_ROW_FIELDS``: arms are
    ``control`` and ``experience``, status is known/unknown/incomplete/unstarted,
    branch is first, replayed_steps is zero, and split is train. A known result
    requires a closed policy episode, zero worker exit, a scored receipt, exact
    boolean success and both unknown flags false.
    Extra fields, unplanned/duplicate rows, changed identities and leakage are
    rejected. An incomplete cohort never yields subset effect or harm estimates.
    """
    pairs = _manifest(expected_pairs, source_scenario_ids)
    common = _identity(identity)
    _require(_identifier(experience_bundle_id))
    rows = _matrix(pairs, summaries, common, experience_bundle_id)
    categories = Counter(status for _, status in rows.values())
    completed = sum(all((key, arm) in rows and rows[key, arm][1] == "known"
                        for arm in ("control", "experience"))
                    for key in ((task, seed) for task, _, seed in pairs))
    counts = {"planned_pairs": len(pairs), "planned_scenarios": len({scenario for _, scenario, _ in pairs}),
              "planned_summaries": 2 * len(pairs), "observed_summaries": len(rows),
              "known_summaries": categories["known"], "unknown_summaries": categories["unknown"],
              "incomplete_summaries": categories["incomplete"], "unstarted_summaries": categories["unstarted"],
              "missing_summaries": 2 * len(pairs) - len(rows), "complete_pairs": completed}
    thresholds = {"min_scenarios": MIN_SCENARIOS, "max_observed_harm_rate": 0.05,
                  "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                  "both_lower_bounds_strictly_positive": True}
    result = {"comparison": "ordinary_experience_vs_no_experience_train_only", "experience_bundle_id": experience_bundle_id,
              "cohort_sha256": fingerprint(sorted(pairs)),
              "source_scenarios_sha256": fingerprint(sorted(source_scenario_ids)),
              "identity": common, **counts, "thresholds": thresholds,
              "status": "not_evaluable", "reason": "incomplete_cohort", "passed": None,
              "checks": None, "statistics": None,
              "limits": "Metadata consistency only; caller must authenticate registration, receipts, "
                        "source provenance and experience content. Scenario IDs do not prove independence, "
                        "and these intervals do not establish population coverage or a harm-risk bound. "
                        "Train scenarios may have prior exposure; no unseen generalization is established. "
                        "This helper does not account for generation, execution or validation costs "
                        "and does not authorize training or deployment."}
    if completed != len(pairs):
        return result

    controls, treatments, groups = [], [], {}
    for task_id, scenario_id, seed in sorted(pairs, key=lambda value: (value[0], value[2])):
        control, treatment = (rows[(task_id, seed), arm][0] for arm in ("control", "experience"))
        task = Task(task_id=task_id, scenario_id=scenario_id, split="train", instruction="")
        common_episode = {"task": task, "model_id": common["model_id"], "seed": seed,
                          "steps": [], "termination": "closed_first_attempt"}
        controls.append(Episode(**common_episode, success=control["success"]))
        treatments.append(Episode(**common_episode, success=treatment["success"],
                                  experience_id=experience_bundle_id))
        groups.setdefault(scenario_id, []).append(int(treatment["success"]) - int(control["success"]))
    paired = paired_estimate(controls, treatments, bootstrap_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    try:
        exact = exact_cluster_resampling(list(groups.values()))
    except ValueError:
        result["reason"] = "exact_resampling_unavailable"
        return result
    stats = {**paired, "ties": len(pairs) - paired["helped"] - paired["harmed"],
             "net_helped": paired["helped"] - paired["harmed"],
             "observed_harm_rate": paired["harmed"] / len(pairs), "exact_cluster_ci95": exact["ci95"],
             "exact_resample_count": exact["resample_count"], "exact_degenerate": exact["degenerate"],
             "exact_warning": exact["warning"]}
    checks = {"at_least_8_scenarios": paired["scenario_count"] >= MIN_SCENARIOS,
              "bootstrap_lower_positive": paired["cluster_bootstrap_ci95"][0] > 0,
              "exact_cluster_lower_positive": exact["ci95"][0] > 0,
              "observed_harm_at_most_5_percent": (
                  paired["harmed"] * MAX_HARM_DENOMINATOR <= len(pairs) * MAX_HARM_NUMERATOR)}
    result.update(status="complete", reason=None, passed=all(checks.values()), checks=checks, statistics=stats)
    return result
