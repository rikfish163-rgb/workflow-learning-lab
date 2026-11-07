"""One prospective writer lease using independently verified ordered evidence."""

import sys
from functools import cache
from types import FunctionType

from scripts import native_state_recovery_fixture_transitions_001 as transitions
from scripts import native_state_recovery_transition_stage_001 as stage
from scripts import report_native_state_recovery_fixture_writer_001 as old_audit
from scripts import run_native_state_recovery_fixture_writer_001 as previous
from scripts import run_native_state_recovery_writer_001 as base
from scripts.native_appworld_worker_client import WORKSPACE, _same
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _read, _sha

source = previous.source
ID = "qwen38-native-state-recovery-transition-writer-001"
PLAN = "artifacts/qwen38-native-state-recovery-transition-writer-plan-001.json"
MODULE = "scripts.run_native_state_recovery_transition_writer_001"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-transition-writer-001"
QUALIFICATION_ID = "native-state-recovery-transition-writer-qualification-001"
QUALIFICATION = f"artifacts/{QUALIFICATION_ID}.json"
CPU_PROOF = "artifacts/native-state-recovery-transition-evidence-CPU-001.json"
CPU_PROOF_SHA = "83a4d2b5c0506e52ae61e82b0f1a5731495215f504a21ea9c04b01d85bdfae19"
OLD_WRITER_PLAN_SHA = "c0824ee9239ff68c2ce843c8541b2405928f05ee6265dc61d475090a3fde69e1"
OLD_WRITER_RESULT_SHA = "bda6ecc6d0a9f2fa8d2e38a2c7d3fecf49d052970bcf7ee5394d091c76e0f821"
BASE_SHA = "b4b0ddc5bb22d36c4c34ddd706685541cf74803c12d68b6749f2bb448a2ef571"
OLD_LAUNCHER_SHA = "7a322ffd5d46a4ebe06c734a76a5c6dad383c148ee2ef07083211f756036923f"
SOURCES = previous.SOURCES | {
    MODULE.replace(".", "/") + ".py",
    "scripts/native_state_recovery_fixture_transitions_001.py",
    "scripts/native_state_recovery_transition_stage_001.py",
    "scripts/native_state_recovery_transition_review_001.py",
    "scripts/report_native_state_recovery_fixture_writer_001.py",
    "tests/test_native_state_recovery_transition_writer_001.py",
}


def _cpu_proof(root):
    base._require(_public_hash(root, CPU_PROOF) == CPU_PROOF_SHA)
    proof = _read(root, CPU_PROOF)
    base._require(all(_public_hash(root, path) == sha for path, sha in proof["sha256_bindings"].items()))
    return proof


def _qualification(proof, cpu):
    base._require(type(proof) is dict and set(proof) == {
        "id", "source_plan_sha256", "source_result_sha256", "source_denominator",
        "witness_scenarios", "private_input_sha256",
    } and proof["id"] == QUALIFICATION_ID and proof["source_denominator"] == proof["witness_scenarios"] == 4
        and proof["source_plan_sha256"] == cpu["source_plan_sha256"]
        and proof["source_result_sha256"] == cpu["source_result_sha256"] and _sha(proof["private_input_sha256"]))
    return proof


def _qualified():
    cpu = _cpu_proof(WORKSPACE)
    old = old_audit.report(OLD_WRITER_PLAN_SHA, OLD_WRITER_RESULT_SHA)
    base._require(old["status"] == "abstained" and old["writer"]["reason"] == "model_abstained"
                  and old["review"] is None and old["source_witness_scenarios"] == 4)
    prepared = transitions.prepare(cpu["source_plan_sha256"], cpu["source_result_sha256"])
    stage._validate_cases(prepared["cases"])
    base._require(base._digest(prepared["cases"]) == cpu["ordered_evidence_sha256"])
    proof = {"id": QUALIFICATION_ID, "source_plan_sha256": cpu["source_plan_sha256"],
             "source_result_sha256": cpu["source_result_sha256"], "source_denominator": 4,
             "witness_scenarios": len(prepared["cases"]), "private_input_sha256": base._digest(prepared)}
    return _qualification(proof, cpu), prepared


def qualify():
    return _qualified()[0]


def _prepare(plan):
    proof, prepared = _qualified()
    base._require(_same(proof, plan["qualification"]))
    return prepared


@cache
def bindings():
    base._require(_public_hash(base.ROOT, previous.MODULE.replace(".", "/") + ".py") == OLD_LAUNCHER_SHA)
    reg, queue, runner = previous.bindings.__wrapped__()
    for component in (reg, queue, runner.episode, runner):
        component.ID, component.PLAN = ID, PLAN
    reg.PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOT
    queue.GUARD_NAME = "native-state-recovery-transition-writer-operations-001.lock"

    def build(root):
        cpu = _cpu_proof(root)
        proof = _qualification(reg.public._read(root, QUALIFICATION), cpu)
        source_plan = source.bindings()[0].load_registration(root, proof["source_plan_sha256"], private=False)
        sources = source_plan["source_sha256"] | {name: _public_hash(root, name) for name in SOURCES}
        inputs = source_plan["input_sha256"] | cpu["sha256_bindings"] | {
            CPU_PROOF: CPU_PROOF_SHA, previous.PLAN: OLD_WRITER_PLAN_SHA,
            source.PLAN: proof["source_plan_sha256"], base.DESIGN: base.DESIGN_SHA256,
            QUALIFICATION: _public_hash(root, QUALIFICATION),
        }
        base._require(all(_public_hash(root, name) == digest for name, digest in inputs.items()))
        return {**source_plan, "id": ID, "private_run_root": PRIVATE_RUN_ROOT, "source_sha256": sources,
                "input_sha256": inputs, "scope": "writer_and_conditional_review_no_task",
                "writer_authorized": True, "writer_max_calls": 1, "review_max_calls": 1, "total_gpu_seconds": 7200,
                "writer_seed": stage.WRITER_SEED, "review_seed": stage.REVIEW_SEED,
                "writer_max_tokens": stage.WRITER_MAX_TOKENS, "review_max_tokens": stage.REVIEW_MAX_TOKENS,
                "writer_design": {"path": base.DESIGN, "sha256": base.DESIGN_SHA256},
                "evidence_representation": "ordered_anonymous_boolean_observations", "qualification": proof}

    reg.build = build
    make_job = queue.make_job

    def job(index, owner, sha):
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    return reg, queue, runner


def core_for(runner):
    core = base.core_for(runner, source_module=source.MODULE, module=MODULE)
    core.episode._journal = core._journal
    return core


def main():
    base._require(_public_hash(base.ROOT, base.MODULE.replace(".", "/") + ".py") == BASE_SHA)
    scope = {**vars(base), "stage": stage}
    scope["run"] = FunctionType(base.run.__code__, scope)
    command = FunctionType(base.main.__code__, scope)
    return command(binding_factory=bindings, prepare_input=_prepare, make_core=core_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_transition_writer_not_admitted\n")
        raise SystemExit(2) from None
