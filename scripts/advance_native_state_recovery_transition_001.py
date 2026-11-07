"""Audit the closed ordered-evidence writer and launch its eligible V27 panel."""

import argparse
import os
import socket
import subprocess
from pathlib import Path

from scripts import advance_native_state_recovery_001 as staging
from scripts import advance_native_state_recovery_fixture_001 as handoff
from scripts import native_source_registration as public
from scripts import native_state_recovery_transition_review_002 as review
from scripts import run_native_state_recovery_transition_utility_001 as utility
from scripts import run_native_state_recovery_transition_writer_002 as writer
from scripts import run_native_state_recovery_utility_001 as common
from scripts.native_appworld_worker_client import WORKSPACE, _encode

REPORT = "artifacts/native-state-recovery-transition-writer-terminal-002.json"
ERROR = "native_state_recovery_transition_handoff_not_admitted"


def _preflight_deployment(plan, plan_sha):
    check = """
import importlib
import sys
from pathlib import Path

module = importlib.import_module(sys.argv[1])
registry, queue, runner = module.base.bindings(module.LINEAGE)
sha = sys.argv[2]
root = Path.cwd()
assert registry.load_registration(root, sha, private=False)["id"] == registry.ID
queue._core()._validate_job(queue.make_job(0, "1" * 32, sha))
runner._services(root)
"""
    subprocess.run([plan["python"], "-B", "-c", check, utility.LINEAGE.module, plan_sha],
                   cwd=staging.DEPLOYMENT, check=True, capture_output=True, timeout=120)


def advance(writer_pid):
    registry, _queue, runner = writer.bindings()
    reader = writer.core_for(runner).episode
    private = Path(registry.PRIVATE_RUN_ROOT)
    launch = handoff._intent(reader, private)
    writer_sha = launch["plan_sha256"]
    result_sha = handoff._closed(reader, private, launch, writer_pid)
    outcome = review.report(writer_sha, result_sha)
    staging._stage(WORKSPACE, REPORT, _encode(outcome))
    if outcome["status"] != "plausible_pending_utility":
        print(f"writer_status={outcome['status']} next_stage=none", flush=True)
        return 0
    manifest = review.prepare_manifest(writer_sha, result_sha)["manifest"]
    manifest_sha = handoff._private_manifest(reader, private, manifest)
    proof = common.qualification(manifest, manifest_sha, lineage=utility.LINEAGE)
    registry = common.bindings(utility.LINEAGE)[0]
    raw = _encode(proof)
    staging._stage(WORKSPACE, registry.QUALIFICATION_PROOF, raw)
    plan = utility.build_registration(WORKSPACE, staging._digest(raw))
    plan_sha = handoff._seal(registry, plan)
    _preflight_deployment(plan, plan_sha)
    handoff._start(plan, plan_sha, utility.LINEAGE.module)
    print(f"writer_status=plausible_pending_utility next_stage={registry.ID} plan_sha256={plan_sha}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--writer-pid", required=True, type=int)
    args = parser.parse_args()
    handoff._require(args.writer_pid > 0 and socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
                     and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                     and os.environ.get("TMPDIR") == public.PRIVATE_STATE_ROOT + "/runtime-tmp")
    return advance(args.writer_pid)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(ERROR) from None
