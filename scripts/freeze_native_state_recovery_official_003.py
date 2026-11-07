"""Freeze nine unused official scenarios from split IDs and prior registrations."""

import copy
import hashlib
import os
import socket

from scripts import freeze_native_state_recovery_official_001 as first
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import prepare_native_state_recovery_holdout_001 as holdout
from scripts.native_appworld_worker_client import (
    ASSET_ROOT,
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _private_tree,
)
from scripts.native_source_registration import public_read

PRIVATE = RUNTIME_TMP / "official-test-normal-003"
ROSTER = PRIVATE / "frozen-panel.json"
SUMMARY = "artifacts/native-state-recovery-official-freeze-003.json"
GENERATION_MAX_TIME_SECONDS = 75.0


def select_roster(task_ids, excluded_scenes):
    scenes = sorted({task.rsplit("_", 1)[0] for task in task_ids} - set(excluded_scenes))[:9]
    if len(scenes) != 9 or any(f"{scene}_{variant}" not in task_ids for scene in scenes for variant in (1, 2, 3)):
        raise ValueError("official_fresh_panel_insufficient_scenarios")
    return [f"{scene}_{variant}" for scene in scenes for variant in (1, 2, 3)]


def validate_selection(draft, reference):
    pairs = draft["expected_pairs"]
    scenes = {scene for _, scene, _ in pairs}
    roster = comparison._hash(_encode([task for task, _, _ in pairs]))
    schedule = comparison._hash(_encode([pairs, draft["arm_orders"], draft["batches"]]))
    expected = comparison._layout([task for task, _, _ in pairs], [], seed=17)
    comparison._require(draft["split"] == "test_normal" and draft["scorer"] == "official_appworld"
                        and len(scenes) == 9 and len(pairs) == 27
                        and all(draft[key] == value for key, value in expected.items())
                        and roster == reference["roster_sha256"] and schedule == reference["schedule_sha256"]
                        and draft["pool_with_prior_outcomes"] is False
                        and draft["fault_policy"] == "natural_observations_only"
                        and draft["protocol"]["generation_max_time_seconds"] == GENERATION_MAX_TIME_SECONDS)
    return roster, schedule


def freeze():
    from scripts.register_native_state_recovery_authored_study_001 import _publish

    comparison._require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    # This verifies split inventories and registered-plan metadata without task bodies.
    _, metadata = holdout.inspect_roster()
    old_summary = _decode(public_read(WORKSPACE, first.SUMMARY))
    raw = first.ROSTER.read_bytes()
    comparison._require(comparison._hash(raw) == old_summary["private_frozen_panel_sha256"])
    old = _decode(raw)
    draft = copy.deepcopy(old["draft"])
    excluded = set()
    for path in RUNTIME_TMP.glob("official-test-normal-*/frozen-panel.json"):
        if path == ROSTER:
            continue
        panel = _decode(path.read_bytes())
        excluded.update(scene for _, scene, _ in panel["draft"]["expected_pairs"])
    for path in sorted({*WORKSPACE.glob("artifacts/*plan*.json"), *WORKSPACE.glob("artifacts/*/*plan*.json")}):
        text = public_read(WORKSPACE, str(path.relative_to(WORKSPACE))).decode()
        excluded.update(holdout.PLAN_TASK_ID.findall(text))
    index = (ASSET_ROOT / "data/datasets/test_normal.txt").read_bytes()
    comparison._require(hashlib.sha256(index).hexdigest() == holdout.TEST_ID_SHA256)
    task_ids = index.decode().splitlines()
    roster = select_roster(task_ids, excluded)
    comparison._require(not {task.rsplit("_", 1)[0] for task in roster} & excluded)
    draft.update(comparison._layout(roster, excluded, seed=17))
    draft["protocol"]["generation_max_time_seconds"] = GENERATION_MAX_TIME_SECONDS
    draft["identity"]["budget_id"] += ":generation-max-time-75s"
    draft["id_metadata"] = {
        **metadata, "selection_rule": "first_9_lexicographic_unreferenced_and_unfrozen_test_normal_scenarios",
        "candidate_roster_sha256": comparison._hash(_encode(roster)),
        "excluded_scenarios_in_split": len({task.rsplit("_", 1)[0] for task in task_ids} & excluded),
    }
    summary = {
        "state": "roster_frozen_execution_pending", "split": "test_normal", "scorer": "official_appworld",
        "scenarios": 9, "pairs": 27, "batches": 14,
        "roster_sha256": comparison._hash(_encode(roster)),
        "schedule_sha256": comparison._hash(_encode([draft["expected_pairs"], draft["arm_orders"], draft["batches"]])),
        "previous_roster_sha256": old_summary["roster_sha256"], "overlap_with_previous_panel_scenarios": 0,
        "generation_max_time_seconds": GENERATION_MAX_TIME_SECONDS,
        "test_task_documents_opened_at_freeze": 0, "model_calls_at_freeze": 0,
    }
    validate_selection(draft, summary)
    raw = _encode({"selection": summary, "draft": draft}) + b"\n"
    summary["private_frozen_panel_sha256"] = comparison._hash(raw)
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    os.close(_private_tree(PRIVATE))
    _publish(ROSTER, raw, 0o600)
    _publish(WORKSPACE / SUMMARY, _encode(summary) + b"\n", 0o644)
    return summary


if __name__ == "__main__":
    try:
        print(_encode(freeze()).decode())
    except Exception:
        raise SystemExit("official_fresh_panel_freeze_failed") from None
