"""Run the original V27 panel after the frozen zero-branch preflight failure."""

from functools import cache

from scripts import report_native_state_recovery_utility_001 as reporter
from scripts import run_native_state_recovery_transition_utility_001 as previous
from scripts.native_appworld_worker_client import _encode
from scripts.native_source_registration import _public_hash, _read

base = previous.base
LINEAGE = base.UtilityLineage(
    "native-state-recovery-transition-utility-002",
    "scripts.run_native_state_recovery_transition_utility_002",
    previous.LINEAGE.review,
    4,
)
FAILURE = "artifacts/native-state-recovery-transition-utility-startup-terminal-001.json"
FAILURE_SHA = "93d12e91b4c3ac43a9ebb36fe9e26e3911666c2c07f8e077bfe8cecd467cc682"
PREVIOUS_PLAN = "artifacts/qwen38-native-state-recovery-transition-utility-plan-001.json"
PREVIOUS_PLAN_SHA = "4d8c0637430d5834f2ee8f38ca6ef5c277e937bf13bae1ebf92a28e0a0a58d63"


def _failure(root):
    base._require(_public_hash(root, FAILURE) == FAILURE_SHA and _public_hash(root, PREVIOUS_PLAN) == PREVIOUS_PLAN_SHA)
    proof = _read(root, FAILURE)
    base._require(proof["plan_sha256"] == PREVIOUS_PLAN_SHA and proof["executed_branches"] == proof["GPU_jobs"] == 0
                  and proof["status"] == "CPU_preflight_failed_before_tasks_and_GPU_enqueue"
                  and proof["automatic_retry"] is False)


@cache
def bindings():
    registry, queue, runner = base.bindings(LINEAGE)
    load = registry.load_registration

    def load_registration(root, sha, *, private=False):
        _failure(root)
        plan = load(root, sha, private=private)
        base._require(plan["input_sha256"].get(FAILURE) == FAILURE_SHA
                      and plan["input_sha256"].get(PREVIOUS_PLAN) == PREVIOUS_PLAN_SHA)
        return plan

    registry.load_registration = load_registration
    registry.REQUIRED_NEW_SOURCES |= {previous.LINEAGE.module.replace(".", "/") + ".py"}
    return registry, queue, runner


def build_registration(root, proof_sha):
    registry, _queue, _runner = bindings()
    _failure(root)
    plan = base.build_registration(root, proof_sha, lineage=LINEAGE)
    plan["input_sha256"] |= {FAILURE: FAILURE_SHA, PREVIOUS_PLAN: PREVIOUS_PLAN_SHA}
    return registry.validate_registration(root, plan)


def _verification_payload(spec):
    return bindings()[2]._verification_payload(spec)


def report(plan_sha, result_sha):
    bindings()
    return reporter.report(plan_sha, result_sha, lineage=LINEAGE)


__all__ = ["_encode"]


if __name__ == "__main__":
    bindings()
    raise SystemExit(base.main(lineage=LINEAGE))
