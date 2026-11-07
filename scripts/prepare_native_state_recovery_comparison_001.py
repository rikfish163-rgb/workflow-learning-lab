"""Prepare two unregistered comparison drafts, with no task or model execution."""

import hashlib
import os
import socket
from collections import Counter
from pathlib import Path

from scripts import native_state_recovery_transition_review_002 as review
from scripts import prepare_native_state_recovery_holdout_001 as holdout
from scripts.exact_cluster_resampling import exact_cluster_resampling
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import _public_hash, public_read
from scripts.native_state_recovery_authored_panel_001 import PANEL
from wll.protocol import Episode, Task, paired_estimate

PRIOR_PLAN = "artifacts/qwen38-native-state-recovery-transition-utility-003-plan-001.json"
PRIOR_TERMINAL = "artifacts/native-state-recovery-transition-utility-terminal-003.json"
WRITER_TERMINAL = "artifacts/native-state-recovery-transition-writer-terminal-002.json"
PANEL_PROOF = "artifacts/native-state-recovery-authored-panel-CPU-001.json"
RUNTIME_PROOF = "artifacts/native-state-recovery-authored-runtime-CPU-001.json"
INPUTS = {
    PRIOR_PLAN: "cfc48519d9805f969e81d7e69df0c723de79e62c72ec6dd8f861f8458302c68c",
    PRIOR_TERMINAL: "26368a9241cf758580c776f0237d07c74a6daf60b79913e384c3fc015cf1b4b5",
    WRITER_TERMINAL: "e2c746d8d65a82caa24b99e9cf28d4a43361cea84176ae39b935e8da9b4c77a4",
    PANEL_PROOF: "ea07ace30f46e6c1beba0bb5fce1021d503adf5ca79c0c52aabb2ebe87003337",
    RUNTIME_PROOF: "2888f6ca02030d0d831db122c10a4eabf7764162b5f93cb4a5a2fb849f11f0f8",
}
AUTHORED_SOURCES = {
    "scripts/native_state_recovery_authored_panel_001.py":
        "e3e5cb95fc94afbac2156fb5387109151f37e586ac3a084717196d3a611cc3b6",
    "scripts/probe_native_state_recovery_authored_panel_001.py":
        "7f4ffdb644f3a9abf60b0246d3c2dc9ea0b73f53a512d5d7b9cf96234c8b52d6",
    "scripts/native_state_recovery_authored_runtime_001.py":
        "9b142ebc12ad44c6e628f7b7bbe4ff256ea2b0600249c7b2d5db96a985ccddb4",
    "scripts/probe_native_state_recovery_authored_runtime_001.py":
        "32225e4c35a8c19eac0c27611899997830ef41ed4f610eb37cd80f415c5fec75",
}
ARMS = ("control", "experience")


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_comparison_draft_invalid")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_inputs():
    verified = {}
    for path, digest in INPUTS.items():
        raw = public_read(WORKSPACE, path)
        _require(_hash(raw) == digest)
        verified[path] = _decode(raw)
    _require(all(_public_hash(WORKSPACE, path) == digest for path, digest in AUTHORED_SOURCES.items()))
    return verified


def _layout(task_ids, origins, *, seed):
    _require(len(task_ids) == len(set(task_ids)) == 27
             and all(holdout.TASK_ID.fullmatch(task) is not None for task in task_ids))
    scenes = list(dict.fromkeys(task.rsplit("_", 1)[0] for task in task_ids))
    _require(len(scenes) == 9 and not set(scenes) & set(origins)
             and task_ids == [f"{scene}_{variant}" for scene in scenes for variant in (1, 2, 3)])
    pairs = [[task, task.rsplit("_", 1)[0], seed] for task in task_ids]
    orders = [list(ARMS if index % 2 == 0 else reversed(ARMS)) for index in range(len(pairs))]
    batches = [list(range(index, min(index + 2, len(pairs)))) for index in range(0, len(pairs), 2)]
    return {"expected_pairs": pairs, "arm_orders": orders, "batches": batches}


def drafts():
    _require(socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
             and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    evidence = _read_inputs()
    old = evidence[PRIOR_PLAN]
    source = old["source"]
    writer = evidence[WRITER_TERMINAL]
    _require(writer["status"] == "plausible_pending_utility"
             and writer["plan_sha256"] == source["writer_plan_sha256"]
             and writer["result_sha256"] == source["writer_result_sha256"])
    bundle = review.load_reviewed_bundle(writer["plan_sha256"], writer["result_sha256"],
                                         source["review_manifest_sha256"])
    _require(bundle["private"] is True and bundle["status"] == "plausible_pending_utility"
             and all(_same(bundle[key], source[key]) for key in (
                 "source_plan_sha256", "source_result_sha256", "source_scenario_ids",
                 "experience_bundle_id", "review_manifest_sha256",
             )))
    protocol = {key: value for key, value in old["utility"].items() if key != "split"}
    policy = "config/gpu-policy.json"
    _require(_public_hash(WORKSPACE, policy) == old["gpu_policy_sha256"]
             and protocol["allowed_physical_gpus"] == [8, 9]
             and protocol["continuous_idle_seconds"] == 180
             and protocol["maximum_simultaneous_project_gpus"] == 1
             and protocol["serial_gpu_jobs"] == 14 and protocol["require_complete_planned_pairs"] is True)
    panel = evidence[PANEL_PROOF]
    isolated = evidence[RUNTIME_PROOF]
    _require(panel["variants_validated"] == panel["expected_positive_passes"] == 27
             and panel["expected_negative_rejections"] == 54
             and isolated["correct_state_passed"] is isolated["cross_app_write_rejected"] is True)
    official_ids, official_metadata = holdout.inspect_roster()
    authored_ids = [task.task_id for task in sorted(PANEL.values(), key=lambda task: (task.scenario, task.variant))]
    _require(not set(official_ids) & set(authored_ids))
    common = {
        "state": "draft_no_execution", "model_arm": old["model_arm"], "model_id": old["model_id"],
        "identity": old["identity"], "service_plan_sha256": old["service_plan_sha256"],
        "gpu_policy_sha256": old["gpu_policy_sha256"], "protocol": protocol,
        "gpu_seconds_per_batch_ceiling": old["gpu_seconds"],
        "gpu_seconds_total_ceiling": old["total_gpu_seconds"],
        "admission_wait_seconds": old["admission_wait_seconds"], "automatic_retry": False,
        "writer_result_sha256": writer["result_sha256"], "review_manifest_sha256": source["review_manifest_sha256"],
        "inputs_sha256": INPUTS,
        "source_sha256": AUTHORED_SOURCES | {
            path: _public_hash(WORKSPACE, path) for path in (
                "scripts/prepare_native_state_recovery_comparison_001.py",
                "scripts/prepare_native_state_recovery_holdout_001.py",
                "scripts/native_appworld_scorer.py",
                "scripts/exact_cluster_resampling.py",
                "wll/protocol.py",
            )
        },
        "model_execution_authorized": False, "official_test_execution_authorized": False,
        "pool_with_prior_outcomes": False, "fault_policy": "natural_observations_only",
    }
    options = {
        "official_test_normal": {
            "split": "test_normal", "scorer": "official_appworld",
            "scope": "selected_official_test_normal_scenarios",
            "test_set_use_if_run": True, "id_metadata": official_metadata,
        },
        "authored": {
            "split": "authored", "scorer": "authored_terminal_state",
            "scope": "fixed_authored_scenarios_only",
            "test_set_use_if_run": False,
            "initial_state_manifest_sha256": panel["initial_state_manifest_sha256"],
        },
    }
    for name, task_ids in (("official_test_normal", official_ids), ("authored", authored_ids)):
        options[name].update(common, **_layout(task_ids, source["source_scenario_ids"],
                                               seed=protocol["generation_seed"]))
    _require(all(len(option["batches"]) == 14 for option in options.values()))
    return options


def summarize(draft, verified_scores):
    """Reduce exact scores from an upstream receipt auditor; no receipt trust is added here."""
    pairs = draft["expected_pairs"]
    protocol = draft["protocol"]
    _require(len(pairs) == 27 and len({task for task, _, _ in pairs}) == 27
             and sorted(Counter(scene for _, scene, _ in pairs).values()) == [3] * 9
             and draft["split"] in {"test_normal", "authored"}
             and protocol["require_complete_planned_pairs"] is protocol["both_lower_bounds_positive"] is True
             and protocol["minimum_scenarios"] == 8 and protocol["maximum_harm_rate"] == 0.05
             and protocol["bootstrap_samples"] == 2000 and protocol["bootstrap_seed"] == 0)
    slots = {}
    for index, arm, value in verified_scores:
        _require(type(index) is int and 0 <= index < len(pairs)
                 and arm in ARMS and (value is None or type(value) is bool)
                 and (index, arm) not in slots)
        slots[index, arm] = value
    complete = sum(all(type(slots.get((index, arm))) is bool for arm in ARMS)
                   for index in range(len(pairs)))
    result = {"status": "not_evaluable", "planned_pairs": len(pairs), "complete_pairs": complete,
              "known_branches": sum(type(value) is bool for value in slots.values()),
              "split": draft["split"], "passed": None, "statistics": None}
    if complete != len(pairs):
        return result
    control, experience, groups = [], [], {}
    for index, (task_id, scene, seed) in enumerate(pairs):
        task = Task(task_id, scene, draft["split"], "")
        episode = {"task": task, "model_id": draft["model_id"], "seed": seed,
                   "steps": [], "termination": "verified_closed_first_attempt"}
        before, after = (slots[index, arm] for arm in ARMS)
        control.append(Episode(**episode, success=before))
        experience.append(Episode(**episode, success=after))
        groups.setdefault(scene, []).append(int(after) - int(before))
    stats = paired_estimate(control, experience, bootstrap_samples=protocol["bootstrap_samples"],
                            seed=protocol["bootstrap_seed"])
    exact = exact_cluster_resampling(list(groups.values()))
    harm = stats["harmed"] / len(pairs)
    stats.update(exact_cluster_ci95=exact["ci95"], exact_resample_count=exact["resample_count"],
                 observed_harm_rate=harm)
    passed = (stats["scenario_count"] >= protocol["minimum_scenarios"]
              and stats["cluster_bootstrap_ci95"][0] > 0 and exact["ci95"][0] > 0
              and harm <= protocol["maximum_harm_rate"])
    return {**result, "status": "complete", "passed": passed, "statistics": stats}


def report():
    options = {}
    choices = drafts()
    for name, draft in choices.items():
        orders = draft["arm_orders"]
        options[name] = {
            "state": draft["state"], "split": draft["split"], "scorer": draft["scorer"],
            "scope": draft["scope"], "draft_sha256": _hash(_encode(draft)),
            "schedule_sha256": _hash(_encode([draft["expected_pairs"], orders, draft["batches"]])),
            "pairs": len(draft["expected_pairs"]), "batches": len(draft["batches"]),
            "control_first": sum(order[0] == "control" for order in orders),
            "experience_first": sum(order[0] == "experience" for order in orders),
            "test_set_use_if_run": draft["test_set_use_if_run"],
        }
    draft = choices["authored"]
    protocol = draft["protocol"]
    common = {key: protocol[key] for key in (
        "allowed_physical_gpus", "continuous_idle_seconds", "maximum_simultaneous_project_gpus",
        "allocator_cap_gib", "max_actions", "max_api_requests", "max_generations",
        "context_tokens", "generation_seed", "require_complete_planned_pairs", "bootstrap_samples",
        "bootstrap_seed", "both_lower_bounds_positive", "minimum_scenarios", "maximum_harm_rate",
    )}
    common.update({key: draft[key] for key in (
        "model_arm", "fault_policy", "gpu_seconds_per_batch_ceiling", "gpu_seconds_total_ceiling",
        "admission_wait_seconds", "model_execution_authorized", "official_test_execution_authorized",
    )})
    return {"common": common, "options": options, "study_registered": False,
            "test_task_documents_opened": 0, "model_calls": 0, "gpu_jobs": 0}


if __name__ == "__main__":
    try:
        print(_encode(report()).decode())
    except Exception:
        raise SystemExit("native_state_recovery_comparison_draft_unverified") from None
