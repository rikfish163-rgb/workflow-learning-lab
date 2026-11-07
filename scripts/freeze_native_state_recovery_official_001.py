"""Freeze the selected official panel from IDs and prior evidence only."""

import os

from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts.native_appworld_worker_client import (
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _private_tree,
)
from scripts.native_source_registration import public_read

PRIVATE = RUNTIME_TMP / "official-test-normal-001"
ROSTER = PRIVATE / "frozen-panel.json"
SUMMARY = "artifacts/native-state-recovery-official-freeze-001.json"
REFERENCE = "artifacts/native-state-recovery-comparison-drafts-001.json"
REFERENCE_SHA = "7db242b9c0fa42bcce6a08ef0c2d432701f6ecb0cd72490d96174dd99202cf86"
ROSTER_SHA = "59a5b4f7ee4e4b0fa1e4b646d0b5bb47cff4aa1308c2111cb3daf738d1433992"


def validate_selection(draft, reference):
    pairs = draft["expected_pairs"]
    roster = comparison._hash(_encode([task for task, _, _ in pairs]))
    schedule = comparison._hash(_encode([pairs, draft["arm_orders"], draft["batches"]]))
    metadata = draft["id_metadata"]
    if (draft["split"] != "test_normal" or draft["scorer"] != "official_appworld"
            or draft["fault_policy"] != "natural_observations_only"
            or roster != ROSTER_SHA or metadata["candidate_roster_sha256"] != roster
            or metadata["test_task_documents_opened"] != 0
            or schedule != reference["schedule_sha256"]
            or not len(pairs) == reference["pairs"] == 27 or metadata["candidate_scenarios"] != 9
            or not len(draft["batches"]) == reference["batches"] == 14
            or draft["pool_with_prior_outcomes"] is not False):
        raise ValueError("official_panel_selection_mismatch")
    return roster, schedule


def freeze():
    from scripts.register_native_state_recovery_authored_study_001 import _publish

    raw = public_read(WORKSPACE, REFERENCE)
    if comparison._hash(raw) != REFERENCE_SHA:
        raise ValueError("official_panel_reference_mismatch")
    reference = _decode(raw)["options"]["official_test_normal"]
    draft = comparison.drafts()["official_test_normal"]
    roster, schedule = validate_selection(draft, reference)
    summary = {
        "state": "roster_frozen_execution_pending", "split": "test_normal",
        "scorer": "official_appworld", "scenarios": 9, "pairs": 27, "batches": 14,
        "roster_sha256": roster, "schedule_sha256": schedule,
        "approved_draft_summary_sha256": REFERENCE_SHA,
        "test_task_documents_opened_at_freeze": 0, "model_calls_at_freeze": 0,
    }
    frozen = _encode({"selection": summary, "draft": draft}) + b"\n"
    summary = {**summary, "private_frozen_panel_sha256": comparison._hash(frozen)}
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    os.close(_private_tree(PRIVATE))
    _publish(ROSTER, frozen, 0o600)
    _publish(WORKSPACE / SUMMARY, _encode(summary) + b"\n", 0o644)
    return summary


if __name__ == "__main__":
    try:
        print(_encode(freeze()).decode())
    except Exception:
        raise SystemExit("official_panel_freeze_failed") from None
