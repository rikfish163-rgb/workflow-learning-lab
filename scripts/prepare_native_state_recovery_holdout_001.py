"""Check an independent 27-pair candidate from ID metadata alone."""

import hashlib
import json
import os
import re
import socket
from collections import defaultdict
from pathlib import Path

from scripts import native_source_registration as registration
from scripts.native_appworld_worker_client import WORKSPACE, _encode

SPLITS = {"train": 30, "dev": 19, "test_normal": 56, "test_challenge": 139}
TEST_ID_SHA256 = "c3af41497b6f2f0860a2ff8c09b335dca527e2cf48e59b4aabdb301b6b68db8f"
TASK_ID = re.compile(r"([a-f0-9]{7})_([123])")
PLAN_TASK_ID = re.compile(r"(?<![a-f0-9])([a-f0-9]{7})_[123](?![0-9])")


def inspect_roster():
    if socket.gethostname() != "amax" or Path.cwd() != WORKSPACE or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("native_state_recovery_holdout_metadata_host_invalid")
    directory = Path(registration.ASSET_ROOT) / "data/datasets"
    splits = {}
    for name, expected_count in SPLITS.items():
        raw = (directory / f"{name}.txt").read_bytes()
        if name == "test_normal" and hashlib.sha256(raw).hexdigest() != TEST_ID_SHA256:
            raise ValueError("native_state_recovery_holdout_index_changed")
        scenes = defaultdict(set)
        for task in raw.decode().splitlines():
            match = TASK_ID.fullmatch(task)
            if match is None or match[2] in scenes[match[1]]:
                raise ValueError("native_state_recovery_holdout_index_invalid")
            scenes[match[1]].add(match[2])
        if len(scenes) != expected_count or any(values != {"1", "2", "3"} for values in scenes.values()):
            raise ValueError("native_state_recovery_holdout_index_invalid")
        splits[name] = set(scenes)
    if len(set.union(*splits.values())) != sum(map(len, splits.values())):
        raise ValueError("native_state_recovery_holdout_split_overlap")
    paths = sorted({*WORKSPACE.glob("artifacts/*plan*.json"), *WORKSPACE.glob("artifacts/*/*plan*.json")})
    references = set()
    for path in paths:
        raw = registration.public_read(WORKSPACE, str(path.relative_to(WORKSPACE))).decode()
        json.loads(raw)
        references.update(PLAN_TASK_ID.findall(raw))
    heldout = splits["test_normal"]
    selected = sorted(heldout - references)[:9]
    if len(selected) != 9:
        raise ValueError("native_state_recovery_holdout_insufficient_scenarios")
    roster = [f"{scene}_{variant}" for scene in selected for variant in (1, 2, 3)]
    metadata = {"scope": "ID_and_registered_plan_metadata_only", "scenarios_by_split": SPLITS,
            "plan_files_scanned": len(paths), "test_normal_scenarios_with_plan_references": len(heldout & references),
            "selection_rule": "first_9_lexicographic_unreferenced_test_normal_scenarios_all_3_variants",
            "candidate_scenarios": len(selected), "candidate_pairs": len(roster),
            "candidate_roster_sha256": hashlib.sha256(_encode(roster)).hexdigest(),
            "test_normal_id_file_sha256": TEST_ID_SHA256, "test_task_documents_opened": 0,
            "scores": 0, "model_calls": 0, "study_registered": False}
    return roster, metadata


def report():
    return inspect_roster()[1]


if __name__ == "__main__":
    try:
        print(json.dumps(report(), sort_keys=True, allow_nan=False))
    except Exception:
        raise SystemExit("native_state_recovery_holdout_metadata_unverified") from None
