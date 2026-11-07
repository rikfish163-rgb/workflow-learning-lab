"""Admit one writer/reviewer lease only after an authenticated source gate."""

import argparse
import hashlib
import signal
import sys
import time
from functools import cache
from pathlib import Path

from scripts import native_appworld_model_client_004 as model
from scripts import native_state_recovery_writer_evidence_001 as evidence
from scripts import native_state_recovery_writer_stage_001 as stage
from scripts import report_native_state_recovery_source_003 as audit
from scripts import run_native_state_recovery_source_003 as source
from scripts.native_appworld_worker_client import WORKSPACE, _encode, _same
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _sha

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-state-recovery-writer-001"
PLAN = "artifacts/qwen38-native-state-recovery-writer-plan-001.json"
MODULE = "scripts.run_native_state_recovery_writer_001"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-writer-001"
DESIGN = "artifacts/native-state-recovery-writer-design-001.json"
DESIGN_SHA256 = "c8263ecbbe932782f38c0ae01888e1f8b4c79b8b8afbded6d04ef262af1e50b9"
QUALIFICATION = "artifacts/native-state-recovery-writer-qualification-001.json"
SOURCES = {"scripts/run_native_state_recovery_writer_001.py", "scripts/native_state_recovery_writer_evidence_001.py",
           "scripts/native_state_recovery_writer_stage_001.py", "scripts/report_native_state_recovery_source_003.py",
           "scripts/native_state_recovery_review_001.py",
           "scripts/native_experience_content_001.py", "scripts/native_experience_writer_001.py",
           "scripts/paired_omission_writer_evidence_001.py", "tests/test_native_state_recovery_writer_stage_001.py"}


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_writer_registration_invalid")


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _qualified(result_sha256):
    observed = audit.report(terminal_sha256=result_sha256)
    gate = observed["gate"]
    _require(gate is not None and gate["status"] == "passed" and observed["audited_scored_and_released"] == 10)
    prepared = evidence.prepare(result_sha256)
    _require(len(prepared["cases"]) == gate["witness_scenarios"])
    proof = {"id": "native-state-recovery-writer-qualification-001", "source_plan_sha256": evidence.SOURCE_PLAN_SHA256,
             "source_result_sha256": result_sha256, "source_denominator": 10,
             "witness_scenarios": gate["witness_scenarios"], "private_input_sha256": _digest(prepared)}
    return proof, prepared


def qualify(result_sha256):
    """Build public-safe metadata from independently audited terminal records."""
    return _qualified(result_sha256)[0]


def _qualification(value):
    _require(type(value) is dict and set(value) == {"id", "source_plan_sha256", "source_result_sha256",
        "source_denominator", "witness_scenarios", "private_input_sha256"}
        and value["id"] == "native-state-recovery-writer-qualification-001"
        and value["source_plan_sha256"] == evidence.SOURCE_PLAN_SHA256 and value["source_denominator"] == 10
        and type(value["witness_scenarios"]) is int and 2 <= value["witness_scenarios"] <= 10
        and _sha(value["source_result_sha256"]) and _sha(value["private_input_sha256"]))
    return value


@cache
def bindings():
    reg, queue, episode, runner = source.bindings.__wrapped__()
    for component in (reg, queue, episode, runner):
        component.ID, component.PLAN = ID, PLAN
    reg.PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOT
    queue.GUARD_NAME = "native-state-recovery-writer-operations-001.lock"
    reg.FIXED_INPUTS = reg.FIXED_INPUTS | {DESIGN: DESIGN_SHA256, QUALIFICATION: _public_hash(ROOT, QUALIFICATION),
                                          source.PLAN: evidence.SOURCE_PLAN_SHA256}
    reg.NEW_SOURCES = reg.NEW_SOURCES | SOURCES
    source_build = reg.build

    def build(root):
        plan = source_build(root)
        proof = _qualification(reg.public._read(root, QUALIFICATION))
        plan.update(scope="writer_and_conditional_review_no_task", writer_authorized=True,
                    writer_max_calls=1, review_max_calls=1, total_gpu_seconds=7200,
                    writer_seed=stage.WRITER_SEED, review_seed=stage.REVIEW_SEED,
                    writer_max_tokens=stage.WRITER_MAX_TOKENS, review_max_tokens=stage.REVIEW_MAX_TOKENS,
                    writer_design={"path": DESIGN, "sha256": DESIGN_SHA256}, qualification=proof)
        return plan

    reg.build = build
    make_job = queue.make_job

    def job(index, owner, sha):
        _require(type(index) is int and index == 0)
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    return reg, queue, runner


def core_for(runner, *, source_module=source.MODULE, module=MODULE):
    core = runner._core()
    command = core.unit_command

    def writer_command(kind, *args, **kwargs):
        _require(kind == "pilot")
        values = command(kind, *args, **kwargs)
        duration = f"--property=RuntimeMaxSec={runner.PARENT_SECONDS}"
        _require(values.count(duration) == 1 and sum(item.count(source_module) for item in values) == 1)
        return ["--property=RuntimeMaxSec=94200" if item == duration else item.replace(source_module, module)
                for item in values]

    core.unit_command = writer_command
    return core


def _prepare(plan):
    proof = _qualification(plan["qualification"])
    fresh, prepared = _qualified(proof["source_result_sha256"])
    _require(_same(fresh, proof))
    return prepared


def run(sha, *, binding_factory=bindings, prepare_input=_prepare, make_core=core_for):
    reg, queue, runner = binding_factory()
    core = make_core(runner)
    runtime = core._Runtime(WORKSPACE)
    plan = runtime.load_plan(sha)
    prepared = prepare_input(plan)
    runtime.begin(plan, sha, [])
    result = {"id": reg.ID, "status": "unknown", "plan_sha256": sha, "source_tasks": 0, "actions": 0,
              "official_scores": 0, "costs": {"GPU_jobs": []}, "automatic_retry": False,
              "stage": None, "qualification": plan["qualification"]}
    try:
        job = queue.make_job(0, runtime.owner(), sha)
        with runner._lease(runtime, core, job, result) as (service, guard):
            guard()

            def client(tokens):
                return model.NativeModel(f"http://127.0.0.1:{reg.PORT}/v1", reg.MODEL_ID,
                    expected_backend=service["backend"], max_tokens=tokens, context_tokens=16384,
                    temperature=0.7, timeout=300, deadline=time.monotonic() + 420)

            result["stage"] = stage.run(prepared["cases"], model_factory=client, guard=guard, record=runtime.record)
        result["status"] = result["stage"]["status"]
    except BaseException as error:
        core._capture_initiating_exception(runtime, error)
    finally:
        runtime.record({"event": "writer_study_stop", "report": result})
        runtime.write("study-result.json", result)
        runtime.close()
    return int(result["status"] == "unknown")


def main(*, binding_factory=bindings, prepare_input=_prepare, make_core=core_for):
    if "--serve" in sys.argv[1:]:
        return binding_factory()[1].main()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    _require(_sha(args.plan_sha256))
    if args.launch:
        reg, _queue, runner = binding_factory()
        plan = reg.load_registration(WORKSPACE, args.plan_sha256, private=True)
        prepare_input(plan)
        print(make_core(runner).launch_pilot(args.plan_sha256, root=WORKSPACE)["unit"])
        return 0

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    return run(args.plan_sha256, binding_factory=binding_factory, prepare_input=prepare_input, make_core=make_core)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_writer_not_admitted\n")
        raise SystemExit(2) from None
