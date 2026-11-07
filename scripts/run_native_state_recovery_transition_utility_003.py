"""Run V27 with the authenticated source backend interface on the GPU service."""

from functools import cache

from scripts import report_native_state_recovery_utility_001 as reporter
from scripts import run_native_state_recovery_transition_utility_002 as previous
from scripts.native_appworld_worker_client import _encode, _same
from scripts.native_source_registration import _public_hash, _read

base = previous.base
LINEAGE = base.UtilityLineage(
    "native-state-recovery-transition-utility-003",
    "scripts.run_native_state_recovery_transition_utility_003",
    previous.LINEAGE.review,
    4,
)
FAILURE = "artifacts/native-state-recovery-transition-utility-startup-terminal-002.json"
FAILURE_SHA = "182ba338dbb48ab83ceac587a68edf58036db00890971294602fd0fa64fc8d92"
PREVIOUS_PLAN = "artifacts/qwen38-native-state-recovery-transition-utility-002-plan-001.json"
PREVIOUS_PLAN_SHA = "2710afd0e2b7ca24bebb782dd5006bcfbd2b9f9c179c07b7db5b2ff5f43df779"
PREDECESSORS = {
    previous.FAILURE: previous.FAILURE_SHA, previous.PREVIOUS_PLAN: previous.PREVIOUS_PLAN_SHA,
    FAILURE: FAILURE_SHA, PREVIOUS_PLAN: PREVIOUS_PLAN_SHA,
}


def _failure(root):
    previous._failure(root)
    base._require(_public_hash(root, FAILURE) == FAILURE_SHA and _public_hash(root, PREVIOUS_PLAN) == PREVIOUS_PLAN_SHA)
    proof = _read(root, FAILURE)
    base._require(proof["plan_sha256"] == PREVIOUS_PLAN_SHA and proof["executed_branches"] == 0
                  and proof["status"] == "GPU_service_failed_before_ready_and_tasks"
                  and proof["release_verified"] is True and proof["GPU_ready"] is proof["automatic_retry"] is False)


@cache
def bindings():
    registry, queue, runner = base.bindings(LINEAGE)
    registry._same = _same
    registry.static_backend = LINEAGE.source.bindings()[0].static_backend
    load = registry.load_registration

    def load_registration(root, sha, *, private=False):
        _failure(root)
        plan = load(root, sha, private=private)
        base._require(all(plan["input_sha256"].get(path) == digest for path, digest in PREDECESSORS.items()))
        return plan

    registry.load_registration = load_registration
    registry.REQUIRED_NEW_SOURCES |= {
        previous.LINEAGE.module.replace(".", "/") + ".py",
        "scripts/run_native_state_recovery_transition_utility_001.py",
        "tests/test_native_state_recovery_transition_utility_003.py",
    }
    release = runner._Runtime.release

    def release_with_fresh_state(runtime, job, pids):
        start = runtime.now()
        result = release(runtime, job, pids)
        state = result.get("state")
        if result.get("verified") is True and type(state) is dict and type(state.get("job")) is dict:
            fresh = runtime.queue("status", job, pids)
            elapsed = runtime.now() - start
            base._require(_same(fresh["job"], state["job"]) and fresh["owned"] is fresh["monitor_live"] is True
                          and fresh["admission_cancelled"] is fresh["released"] is True and not fresh["model_pids"]
                          and fresh["epoch"] >= state["epoch"] and elapsed <= 90)
            return {**result, "state": fresh, "release_wall_seconds": elapsed}
        return result

    runner._Runtime.release = release_with_fresh_state
    return registry, queue, runner


def build_registration(root, proof_sha):
    registry, _queue, _runner = bindings()
    _failure(root)
    plan = base.build_registration(root, proof_sha, lineage=LINEAGE)
    plan["input_sha256"] |= PREDECESSORS
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
