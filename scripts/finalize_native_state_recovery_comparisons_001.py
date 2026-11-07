"""Audit the authored run, start the registered holdout once, and audit it."""

import argparse
import os
import socket
from pathlib import Path

from scripts import report_native_state_recovery_official_001 as official_audit
from scripts import run_native_state_recovery_authored_study_003 as authored
from scripts import run_native_state_recovery_official_001 as official
from scripts.native_appworld_scorer import _main_pid, _stable_read
from scripts.native_appworld_worker_client import WORKSPACE, _encode, _private_tree
from scripts.register_native_state_recovery_authored_study_001 import _publish
from scripts.report_native_state_recovery_transition_utility_003 import _wait

AUTHORED_REPORT = "artifacts/native-state-recovery-authored-terminal-003.json"
OFFICIAL_REPORT = "artifacts/native-state-recovery-official-terminal-001.json"
WAIT_SECONDS = official.frozen.PARENT_SECONDS + 120


def _result_sha(path):
    fd = _private_tree(path.parent)
    try:
        return official._hash(_stable_read(fd, path.name, official.frozen.MAX_DOCUMENT))
    finally:
        os.close(fd)


def finish(authored_pid, plan_sha):
    official._require(socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
                       and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and authored_pid > 0)
    official.load_public(WORKSPACE, plan_sha)
    official.load_public(official.DEPLOYMENT, plan_sha)
    official._require(not (official.PRIVATE / "parent-launch-intent.json").exists())
    print(f"waiting_for={official.AUTHORED_UNIT} pid={authored_pid}", flush=True)
    _wait(official.AUTHORED_UNIT, authored_pid, WAIT_SECONDS)
    lineage = authored.lineage()
    reporter = lineage._clone(lineage.REPORTER, root=WORKSPACE, study=lineage.bindings())
    audit = reporter.report(official.AUTHORED_PLAN_SHA, _result_sha(official.AUTHORED_RESULT))
    _publish(WORKSPACE / AUTHORED_REPORT, _encode(audit) + b"\n", 0o644)
    print(f"authored_audit_written={AUTHORED_REPORT}", flush=True)

    unit = official.launch(plan_sha)
    print(f"official_unit={unit}", flush=True)
    if pid := _main_pid(unit):
        _wait(unit, pid, WAIT_SECONDS)
    audit = official_audit.report(plan_sha, _result_sha(official.PRIVATE / "study-result.json"))
    _publish(WORKSPACE / OFFICIAL_REPORT, _encode(audit) + b"\n", 0o644)
    print(f"official_audit_written={OFFICIAL_REPORT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authored-pid", type=int, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    try:
        finish(args.authored_pid, args.plan_sha256)
    except Exception:
        raise SystemExit("state_recovery_comparison_handoff_failed") from None
