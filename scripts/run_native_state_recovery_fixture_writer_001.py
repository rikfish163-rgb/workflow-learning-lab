"""Run the state-recovery writer on audited authored workflow transitions."""

import sys
from functools import cache

from scripts import report_native_state_recovery_fixture_001 as audit
from scripts import run_native_state_recovery_fixture_001 as source
from scripts import run_native_state_recovery_writer_001 as base
from scripts.native_appworld_worker_client import _same
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _sha

ID = "qwen38-native-state-recovery-fixture-writer-001"
PLAN = "artifacts/qwen38-native-state-recovery-fixture-writer-plan-001.json"
MODULE = "scripts.run_native_state_recovery_fixture_writer_001"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-fixture-writer-001"
QUALIFICATION_ID = "native-state-recovery-fixture-writer-qualification-001"
QUALIFICATION = f"artifacts/{QUALIFICATION_ID}.json"
SOURCES = base.SOURCES | {
    "scripts/run_native_state_recovery_fixture_writer_001.py",
    "scripts/native_state_recovery_fixture_review_001.py",
    "scripts/native_state_recovery_review_001.py",
    "tests/test_native_state_recovery_fixture_lineage_001.py",
}


def _qualified(source_plan_sha, source_result_sha):
    base._require(_sha(source_plan_sha) and _sha(source_result_sha))
    prepared = audit.prepare(source_plan_sha, source_result_sha)
    proof = {
        "id": QUALIFICATION_ID,
        "source_plan_sha256": source_plan_sha,
        "source_result_sha256": source_result_sha,
        "source_denominator": 4,
        "witness_scenarios": len(prepared["cases"]),
        "private_input_sha256": base._digest(prepared),
    }
    return _qualification(proof), prepared


def qualify(source_plan_sha, source_result_sha):
    """Project the authenticated source gate without exposing any witness text."""
    return _qualified(source_plan_sha, source_result_sha)[0]


def _qualification(proof):
    base._require(type(proof) is dict and set(proof) == {
        "id", "source_plan_sha256", "source_result_sha256", "source_denominator",
        "witness_scenarios", "private_input_sha256",
    } and proof["id"] == QUALIFICATION_ID and proof["source_denominator"] == 4
        and type(proof["witness_scenarios"]) is int and 2 <= proof["witness_scenarios"] <= 4
        and all(_sha(proof[key]) for key in ("source_plan_sha256", "source_result_sha256", "private_input_sha256")))
    return proof


@cache
def bindings():
    reg, queue, episode, runner = source.bindings.__wrapped__()
    for component in (reg, queue, episode, runner):
        component.ID, component.PLAN = ID, PLAN
    reg.PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOT
    queue.GUARD_NAME = "native-state-recovery-fixture-writer-operations-001.lock"

    def build(root):
        proof = _qualification(reg.public._read(root, QUALIFICATION))
        previous = source.bindings()[0].load_registration(root, proof["source_plan_sha256"], private=False)
        sources = previous["source_sha256"] | {name: _public_hash(root, name) for name in SOURCES}
        inputs = previous["input_sha256"] | {
            source.PLAN: proof["source_plan_sha256"],
            base.DESIGN: base.DESIGN_SHA256,
            QUALIFICATION: _public_hash(root, QUALIFICATION),
        }
        base._require(all(_public_hash(root, name) == digest for name, digest in inputs.items()))
        stage = base.stage
        return {
            **previous, "id": ID, "private_run_root": PRIVATE_RUN_ROOT, "source_sha256": sources,
            "input_sha256": inputs, "scope": "writer_and_conditional_review_no_task",
            "writer_authorized": True, "writer_max_calls": 1, "review_max_calls": 1, "total_gpu_seconds": 7200,
            "writer_seed": stage.WRITER_SEED, "review_seed": stage.REVIEW_SEED,
            "writer_max_tokens": stage.WRITER_MAX_TOKENS, "review_max_tokens": stage.REVIEW_MAX_TOKENS,
            "writer_design": {"path": base.DESIGN, "sha256": base.DESIGN_SHA256}, "qualification": proof,
        }

    reg.build = build
    make_job = queue.make_job

    def job(index, owner, sha):
        base._require(type(index) is int and index == 0)
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    return reg, queue, runner


def _prepare(plan):
    proof = _qualification(plan["qualification"])
    fresh, prepared = _qualified(proof["source_plan_sha256"], proof["source_result_sha256"])
    base._require(_same(fresh, proof))
    return prepared


def core_for(runner):
    return base.core_for(runner, source_module=source.MODULE, module=MODULE)


if __name__ == "__main__":
    try:
        raise SystemExit(base.main(binding_factory=bindings, prepare_input=_prepare, make_core=core_for))
    except Exception:
        sys.stderr.write("native_state_recovery_fixture_writer_not_admitted\n")
        raise SystemExit(2) from None
