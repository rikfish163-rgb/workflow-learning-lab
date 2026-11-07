"""Admit the ordered-evidence model call after the frozen pre-entry import failure."""

import sys
from functools import cache
from pathlib import Path
from types import FunctionType

from scripts import run_native_state_recovery_transition_writer_001 as previous
from scripts import run_native_state_recovery_writer_001 as base
from scripts.native_appworld_worker_client import WORKSPACE
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _read

source = previous.source
stage = previous.stage
ID = "qwen38-native-state-recovery-transition-writer-002"
PLAN = "artifacts/qwen38-native-state-recovery-transition-writer-plan-002.json"
MODULE = "scripts.run_native_state_recovery_transition_writer_002"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-transition-writer-002"
FAILURE = "artifacts/native-state-recovery-transition-writer-startup-terminal-001.json"
FAILURE_SHA = "0f7c2b584f807e883cc33df4625ada28c9a80fbd8a025c585aa2986dc1e81400"
FAILED_PLAN_SHA = "162d8e119cbff89f167abb9c2e40b5cfd09c395db43823af606d362f11478bda"
FAILED_RESULT_SHA = "06fce89d74cc242a66cbcc94ac0a86cf9590db55ecc5cbb06fab815b0bd83062"
SOURCES = {MODULE.replace(".", "/") + ".py", "scripts/native_state_recovery_transition_review_002.py",
           "scripts/native_experience_review_001.py", "scripts/native_experience_registration_001.py"}


def _failure(root):
    base._require(_public_hash(root, FAILURE) == FAILURE_SHA and _public_hash(root, previous.PLAN) == FAILED_PLAN_SHA)
    proof = _read(root, FAILURE)
    base._require(proof["plan_sha256"] == FAILED_PLAN_SHA and proof["result_sha256"] == FAILED_RESULT_SHA
                  and proof["status"] == "deployment_import_failed_before_GPU_entry"
                  and proof["model_calls"] == 0 and proof["release_verified"] is True
                  and proof["GPU_ready"] is proof["idle_ticket_consumed"] is proof["automatic_retry"] is False)
    return proof


@cache
def bindings():
    reg, queue, runner = previous.bindings.__wrapped__()
    for component in (reg, queue, runner.episode, runner):
        component.ID, component.PLAN = ID, PLAN
    reg.PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOT
    queue.GUARD_NAME = "native-state-recovery-transition-writer-operations-002.lock"
    prior_build = reg.build

    def build(root):
        _failure(root)
        plan = prior_build(root)
        source_files = {name: _public_hash(root, name) for name in SOURCES}
        return {**plan, "id": ID, "private_run_root": PRIVATE_RUN_ROOT,
                "source_sha256": plan["source_sha256"] | source_files,
                "input_sha256": plan["input_sha256"] | {FAILURE: FAILURE_SHA, previous.PLAN: FAILED_PLAN_SHA},
                "startup_predecessor": {"plan_sha256": FAILED_PLAN_SHA, "result_sha256": FAILED_RESULT_SHA}}

    reg.build = build
    make_job = queue.make_job

    def job(index, owner, sha):
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    return reg, queue, runner


def _prepare(plan):
    proof = _failure(WORKSPACE)
    reg, _queue, runner = previous.bindings()
    core = previous.core_for(runner).episode
    result_path = Path(reg.PRIVATE_RUN_ROOT) / "study-result.json"
    base._require(base.hashlib.sha256(core._read_private(result_path, maximum=64_000_000)).hexdigest()
                  == proof["result_sha256"])
    return previous._prepare(plan)


def core_for(runner):
    core = base.core_for(runner, source_module=source.MODULE, module=MODULE)
    core.episode._journal = core._journal
    return core


def main():
    base._require(_public_hash(base.ROOT, base.MODULE.replace(".", "/") + ".py") == previous.BASE_SHA)
    scope = {**vars(base), "stage": stage}
    scope["run"] = FunctionType(base.run.__code__, scope)
    command = FunctionType(base.main.__code__, scope)
    return command(binding_factory=bindings, prepare_input=_prepare, make_core=core_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_transition_writer_002_not_admitted\n")
        raise SystemExit(2) from None
