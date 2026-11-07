"""Evaluate a fresh official panel with a registered per-generation time limit."""

import argparse
import copy
import os
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts import freeze_native_state_recovery_official_003 as freeze
from scripts import run_native_state_recovery_official_001 as base
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read

ID = "qwen38-native-state-recovery-official-003"
PLAN = "artifacts/qwen38-native-state-recovery-official-model-plan-003.json"
MODULE = "scripts.run_native_state_recovery_official_003"
SOURCE = MODULE.replace(".", "/") + ".py"
BASE_SOURCE = "scripts/run_native_state_recovery_official_001.py"
BASE_SHA = "faa9b3b9dde998f51ef3b5beb7977d635119c544dfddec5c53740ec2a87b20aa"
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = freeze.PRIVATE
REPORT = "artifacts/native-state-recovery-official-terminal-003.json"
LIMIT = freeze.GENERATION_MAX_TIME_SECONDS


def limit_generation(model):
    """Bound the engine instance using Transformers' token-boundary time stop."""
    original = model.generate

    def generate(*args, **kwargs):
        kwargs["max_time"] = LIMIT
        return original(*args, **kwargs)

    model.generate = generate
    return original


def _backend(prior):
    expected = copy.deepcopy(prior["expected_backend_static"])
    expected["B3"]["generation_max_time_seconds"] = LIMIT
    return expected


def _load_private(core, root, sha, *, private=False):
    core._require(root == WORKSPACE and private is False
                  and core.socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    public = core.load_public(root, sha)
    summary = _decode(public_read(root, freeze.SUMMARY))
    panel = core.authored.episode._read(freeze.ROSTER, summary["private_frozen_panel_sha256"])
    core._require(_same(panel["selection"], {key: value for key, value in summary.items()
                                            if key != "private_frozen_panel_sha256"}))
    draft = panel["draft"]
    freeze.validate_selection(draft, summary)
    prior = core._prior()
    protocol = {key: value for key, value in prior["utility"].items() if key != "split"}
    protocol["generation_max_time_seconds"] = LIMIT
    identity = {**prior["identity"], "budget_id": prior["identity"]["budget_id"] + ":generation-max-time-75s"}
    core._require(_same(draft["protocol"], protocol) and _same(draft["identity"], identity)
                  and all(_same(draft[key], prior[key]) for key in (
                      "model_arm", "model_id", "service_plan_sha256", "gpu_policy_sha256")))
    return {**prior, "id": ID, "draft": draft, "identity": identity,
            "expected_backend_static": _backend(prior), "expected_pairs": draft["expected_pairs"],
            "batches": draft["batches"], "branch_orders": draft["arm_orders"],
            "utility": {**protocol, "split": "test_normal"}, "private_run_root": str(PRIVATE),
            "public_sha256": sha, "dev_or_test_authorized": True,
            "source_sha256": {**prior["source_sha256"], **public["source_sha256"]}}


def _serve(core, job):
    from http.server import ThreadingHTTPServer

    from scripts.serve_qwen38 import make_handler
    from scripts.serve_qwen38_b3_runtime_relocation_001 import loaded_engine
    from wll.protocol import fingerprint

    queue = core._queue()
    queue._validate_job(job)
    core._require(core.socket.gethostname() == "amax" and ROOT == core.DEPLOYMENT
                  and Path.cwd().resolve() == ROOT)
    plan = core.load_public(ROOT, job["plan_sha256"])
    queue.check_admission_window(ROOT, job)
    with loaded_engine(plan["runtime_relocation"]["sha256"], plan["service_plan_sha256"], plan["model_id"],
                       admitted=lambda gpu: queue.record_gpu_entry(ROOT, job, gpu)) as engine:
        original = limit_generation(engine.model)
        engine.backend["generation_max_time_seconds"] = LIMIT
        core._require(_same(core.static_backend(core.source.wire_backend(engine.backend)),
                            plan["expected_backend_static"]["B3"]))
        server = ThreadingHTTPServer(("127.0.0.1", core._registry().PORT), make_handler(engine))
        try:
            runtime = queue._get_core()
            runtime.write_new(queue.ready_path(ROOT / "artifacts", job), {
                "job_sha256": fingerprint(job), "ready_epoch": runtime.epoch(),
                "ready_monotonic": runtime.monotonic(), "backend_sha256": fingerprint(engine.backend),
                "pid": os.getpid(),
            })
            server.serve_forever()
        finally:
            server.server_close()
            engine.model.generate = original
        core._require(not engine.failed)


@cache
def bindings():
    raw = public_read(ROOT, BASE_SOURCE)
    base._require(base._hash(raw) == BASE_SHA)
    code = raw.decode()
    replacements = (
        (base.ID, ID, 1), (base.PLAN, PLAN, 1),
        ("run_native_state_recovery_official_001", "run_native_state_recovery_official_003", 2),
        ("freeze_native_state_recovery_official_001", "freeze_native_state_recovery_official_003", 2),
        ("native-official-001-", "native-official-003-", 1),
        ("PORT=queue.PORT,", 'PORT=_prior()["port"],', 1),
        ("queue = study._queue()", "study = study.bindings()\nqueue = study._queue()", 1),
        ("prior = study._prior()", "prior = study._prior()\n"
         "assert queue.PORT == study._registry().PORT == prior['port']", 1),
    )
    for before, after, count in replacements:
        base._require(code.count(before) == count)
        code = code.replace(before, after)
    core = ModuleType("scripts._official_003_execution")
    core.__file__ = str(ROOT / SOURCE)
    exec(compile(code, str(ROOT / BASE_SOURCE), "exec", dont_inherit=True), vars(core))
    core.SOURCES |= {BASE_SOURCE, "scripts/freeze_native_state_recovery_official_001.py"}
    public = core.load_public

    def load_public(root, sha):
        value = public(root, sha)
        return {**value, "expected_backend_static": _backend(core._prior())}

    core.load_public = load_public
    core._load_private = lambda root, sha, private=False: _load_private(core, root, sha, private=private)
    core._queue_template = core._bind(core.authored._queue.__wrapped__, ID=ID, PLAN=PLAN,
                                      _job=core._job, load_public=load_public)
    queue_factory = core._queue

    @cache
    def queue():
        value = queue_factory()
        value.PORT = core._prior()["port"]
        value.serve = lambda job: _serve(core, job)
        return value

    core._queue = queue
    return core


def report(plan_sha, result_sha):
    from scripts import report_native_state_recovery_official_001 as audit

    core = bindings()
    scores = base._bind(audit._scores, study=core)
    return base._bind(audit.report, study=core, _scores=scores)(plan_sha, result_sha)


def _launch_and_audit(sha):
    from scripts.finalize_native_state_recovery_comparisons_001 import (
        WAIT_SECONDS,
        _result_sha,
        _wait,
    )
    from scripts.register_native_state_recovery_authored_study_001 import _publish

    unit = bindings().launch(sha)
    print(f"official_unit={unit}", flush=True)
    if pid := _main_pid(unit):
        _wait(unit, pid, WAIT_SECONDS)
    outcome = report(sha, _result_sha(PRIVATE / "study-result.json"))
    _publish(WORKSPACE / REPORT, _encode(outcome) + b"\n", 0o644)
    print(f"official_audit_written={REPORT}", flush=True)


def main():
    core = bindings()
    if sys.argv[1:] == ["--register"]:
        print(core.register())
        return 0
    if sys.argv[1:2] == ["--launch-and-audit"]:
        parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
        parser.add_argument("--launch-and-audit", action="store_true", required=True)
        parser.add_argument("--plan-sha256", required=True)
        args = parser.parse_args()
        _launch_and_audit(args.plan_sha256)
        return 0
    return core.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("official_fresh_panel_execution_failed") from None
