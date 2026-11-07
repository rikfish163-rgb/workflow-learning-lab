"""Run the frozen official AppWorld panel with independent terminal scoring."""

import argparse
import os
import shutil
import signal
import socket
import subprocess
from functools import cache
from pathlib import Path
from types import SimpleNamespace

from scripts import freeze_native_state_recovery_official_001 as freeze
from scripts import native_appworld_scorer as scorer
from scripts import native_state_recovery_gpu_queue_001 as service_queue
from scripts import native_state_recovery_transition_review_002 as review
from scripts import run_native_experience_utility_001 as frozen
from scripts import run_native_state_recovery_authored_study_001 as authored
from scripts import run_native_state_recovery_source_002 as source
from scripts.native_appworld_worker_client import (
    ASSET_ROOT,
    PYTHON,
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _private_tree,
    _same,
)
from scripts.native_source_registration import _public_hash, public_read, static_backend

ID = "qwen38-native-state-recovery-official-001"
PLAN = "artifacts/qwen38-native-state-recovery-official-model-plan-001.json"
MODULE = "scripts.run_native_state_recovery_official_001"
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = freeze.PRIVATE
DEPLOYMENT = authored.episode.DEPLOYMENT
AUTHORED_UNIT = "wll-authored-parent-0905ce07cc03a360d918f0b322326993.service"
AUTHORED_RESULT = RUNTIME_TMP / "authored-panel-model-003" / "study-result.json"
AUTHORED_PLAN_SHA = "877ad6a6f2182a21192c2251d704393e1b6fa9deea8bdd406b10e358bdd90f42"
SOURCES = {
    MODULE.replace(".", "/") + ".py",
    authored.SOURCE, authored.episode.EPISODE_SOURCE, authored.QUEUE_SOURCE, authored.LOOP_SOURCE,
    "scripts/freeze_native_state_recovery_official_001.py",
    "scripts/native_appworld_scorer.py",
    "scripts/native_state_recovery_gpu_queue_001.py",
    "scripts/native_state_recovery_transition_review_002.py",
    "scripts/native_state_recovery_unit_lifecycle_001.py",
    "scripts/prepare_native_state_recovery_comparison_001.py",
    "scripts/prepare_native_state_recovery_holdout_001.py",
    "scripts/run_native_state_recovery_source_002.py",
    "scripts/serve_qwen38_b3_runtime_relocation_001.py",
}
_require = frozen._require
_hash = frozen._hash
_prior = authored.episode._prior


def _bind(function, **names):
    bound = authored._bind(function, **names)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def load_public(root, sha):
    _require(root in (WORKSPACE, DEPLOYMENT))
    raw = public_read(root, PLAN)
    _require(_hash(raw) == sha)
    public = _decode(raw)
    _require(set(public) == {"id", "split", "status", "freeze_sha256", "source_sha256"}
             and public["id"] == ID and public["split"] == "test_normal" and public["status"] == "registered"
             and set(public["source_sha256"]) == SOURCES
             and all(_public_hash(root, name) == digest for name, digest in public["source_sha256"].items())
             and _public_hash(root, freeze.SUMMARY) == public["freeze_sha256"])
    prior = _prior()
    return {**public, **{key: prior[key] for key in (
        "runtime_relocation", "service_plan_sha256", "model_id", "expected_backend_static",
    )}}


def _load_private(root, sha, *, private=False):
    _require(root == WORKSPACE and private is False and socket.gethostname() == "amax"
             and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    public = load_public(root, sha)
    summary = _decode(public_read(root, freeze.SUMMARY))
    panel = authored.episode._read(freeze.ROSTER, summary["private_frozen_panel_sha256"])
    _require(_same(panel["selection"], {key: value for key, value in summary.items()
                                       if key != "private_frozen_panel_sha256"}))
    draft = panel["draft"]
    freeze.validate_selection(draft, summary)
    prior = _prior()
    _require(all(_same(draft[key], prior[key]) for key in (
        "identity", "model_arm", "model_id", "service_plan_sha256", "gpu_policy_sha256",
    )) and _same(draft["protocol"], {key: value for key, value in prior["utility"].items() if key != "split"}))
    return {**prior, "id": ID, "draft": draft, "expected_pairs": draft["expected_pairs"],
            "batches": draft["batches"], "branch_orders": draft["arm_orders"],
            "utility": {**draft["protocol"], "split": "test_normal"},
            "private_run_root": str(PRIVATE), "public_sha256": sha, "dev_or_test_authorized": True,
            "source_sha256": {**prior["source_sha256"], **public["source_sha256"]}}


_job = _bind(authored._job, ID=ID, MODULE=MODULE)
_queue_template = _bind(authored._queue.__wrapped__, ID=ID, PLAN=PLAN, _job=_job, load_public=load_public)


@cache
def _queue():
    queue = _queue_template()
    registry = SimpleNamespace(load_registration=load_public, _same=_same, PORT=queue.PORT,
                               static_backend=lambda backend: static_backend(source.wire_backend(backend)))
    queue.serve = _bind(service_queue.serve, _core=lambda: queue, registration=registry, ROOT=ROOT, REMOTE=DEPLOYMENT)
    return queue


def _registry():
    prior = _prior()
    return SimpleNamespace(ID=ID, PLAN=PLAN, MODEL_ID=prior["model_id"], MODEL_ARM=prior["model_arm"],
                           PORT=prior["port"], PYTHON=PYTHON, TMPDIR=str(RUNTIME_TMP),
                           PRIVATE_RUN_ROOT=str(PRIVATE), public_read=public_read,
                           _public_hash=_public_hash, load_registration=_load_private)


def _unit_command(kind, unit, arguments, *, root, log_path):
    command = frozen.unit_command(kind, unit, arguments, root=root, log_path=log_path)
    if kind != "scorer":
        module = frozen.__name__
        position = next(index for index, part in enumerate(command) if module in part)
        command[position] = command[position].replace(module, MODULE)
    return command


_services = _bind(frozen._services, ID=ID, PLAN=PLAN, _registry=_registry, _queue=_queue, unit_command=_unit_command)


def _validate_bundle(plan, bundle):
    expected = plan["source"]
    bindings = ("source_plan_sha256", "source_result_sha256", "writer_plan_sha256", "writer_result_sha256",
                "review_manifest_sha256", "experience_bundle_id", "source_scenario_ids", "source_gate_kind")
    _require(bundle["private"] is True and bundle["status"] == "plausible_pending_utility"
             and bundle["source_gate_kind"] == review.SOURCE_GATE_KIND
             and all(_same(bundle[key], expected[key]) for key in bindings)
             and bool(bundle["private_files_sha256"]))
    text = bundle["insertion_text"]
    _require(type(text) is str and 1 <= len(text.encode()) <= 65536
             and bundle["experience_bundle_id"] == review.BUNDLE_PREFIX + _hash(text.encode()))
    return text


def _preflight(root, sha):
    plan = _load_private(root, sha)
    _require(Path.cwd() == root and _same(load_public(root, sha), load_public(DEPLOYMENT, sha))
             and shutil.disk_usage(PRIVATE).free >= 5 * 1024**3)
    for path in (PRIVATE, ASSET_ROOT, RUNTIME_TMP):
        os.close(_private_tree(path))
    bundle = authored._registered_bundle(plan)
    _validate_bundle(plan, bundle)
    return plan, bundle


class Runtime(frozen._Runtime):
    def __init__(self, root):
        self.root = root
        self.service, self.reader = _services(root)
        self.service.unit_status = authored.Runtime._recheck(self.service.unit_status)
        self.verification_audits = []

    def preflight(self, sha):
        self.service.plan, bundle = _preflight(self.root, sha)
        return self.service.plan, bundle

    check_bundle = _bind(authored.Runtime.check_bundle, PRIVATE=PRIVATE)
    binding = _bind(authored.Runtime.binding, _queue=_queue)
    release = _bind(authored.Runtime.release, _queue=_queue)
    preflight_failure = _bind(frozen._Runtime.preflight_failure, _registry=_registry)


def _registered(plan, bundle):
    _require(plan["id"] == ID and plan["status"] == "registered"
             and plan["utility"]["split"] == "test_normal" and plan["dev_or_test_authorized"] is True
             and plan["utility_authorized"] is True
             and all(plan[key] is False for key in ("writer_authorized", "training_authorized", "automatic_retry")))
    return _validate_bundle(plan, bundle)


def _summaries(plan):
    return [{**row, "split": "test_normal"} for row in frozen._summaries(plan)]


def _experiment(index, arm, owner):
    return f"native-official-001-{index:02d}-{arm}-{owner}"


run_episode = _bind(frozen.run_episode, _services=_services, _registry=_registry, _queue=_queue,
                    _experiment=_experiment, _validate_bundle=_validate_bundle)
_score_arm = _bind(frozen._run_arm, _experiment=_experiment)


def _run_arm(runtime, plan, sha, bundle, job, service, index, arm, row, guard, report, expected_metadata):
    try:
        return _score_arm(runtime, plan, sha, bundle, job, service, index, arm, row, guard, report, expected_metadata)
    finally:
        # Closed attempts have validated actor evidence, even when scoring fails.
        report["dev_or_test_calls"] += int(row["closed"])


def _coordinator_unit(sha):
    return authored.assets.parent_unit(PRIVATE, "study-" + sha).replace(
        "wll-authored-parent-", "wll-native-source-official-", 1)


_launch = _bind(authored.launch_study, _preflight=_preflight, _coordinator_unit=_coordinator_unit,
                _unit_command=_unit_command, PRIVATE=PRIVATE)


def run_study(sha, *, runtime=None, arm_runner=_run_arm):
    if runtime is None:
        _require(scorer._main_pid(_coordinator_unit(sha)) == os.getpid())
        runtime = Runtime(WORKSPACE)
    statistics = SimpleNamespace(evaluate_utility=lambda _pairs, rows, **_: authored._summary(runtime.plan, rows))
    loop = _bind(frozen.run_utility, ID=ID, _registered=_registered, _summaries=_summaries,
                 _run_arm=arm_runner, _make_job=_job, utility=statistics)
    return loop(sha, root=WORKSPACE, _runtime=runtime)


def _verify_deployment():
    code = """
from scripts import run_native_state_recovery_official_001 as study
from scripts import serve_qwen38_b3_runtime_relocation_001 as relocated
queue = study._queue()
assert queue._get_core().make_job(13, 'a'*32, 'b'*64) == study._job(13, 'a'*32, 'b'*64)
prior = study._prior()
relocated.preflight(prior['runtime_relocation']['sha256'], prior['service_plan_sha256'], prior['model_id'])
"""
    child = subprocess.run([PYTHON, "-s", "-B", "-c", code], cwd=DEPLOYMENT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                           capture_output=True, timeout=180, check=False)
    _require(child.returncode == 0 and child.stdout == b"")


def register():
    from scripts.register_native_state_recovery_authored_study_001 import _publish

    _require(Path.cwd() == WORKSPACE and socket.gethostname() == "amax"
             and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and shutil.disk_usage(DEPLOYMENT).free >= 1024**3)
    summary = public_read(WORKSPACE, freeze.SUMMARY)
    for name in sorted(SOURCES | {freeze.SUMMARY}):
        _publish(DEPLOYMENT / name, public_read(WORKSPACE, name), 0o644)
    _verify_deployment()
    sources = {name: _public_hash(WORKSPACE, name) for name in sorted(SOURCES)}
    plan = {"id": ID, "split": "test_normal", "status": "registered", "freeze_sha256": _hash(summary),
            "source_sha256": sources}
    raw = _encode(plan) + b"\n"
    sha = _hash(raw)
    _publish(DEPLOYMENT / PLAN, raw, 0o644)
    _publish(WORKSPACE / PLAN, raw, 0o644)
    _preflight(WORKSPACE, sha)
    return sha


def launch(sha):
    _require(scorer._main_pid(AUTHORED_UNIT) == 0)
    fd = _private_tree(AUTHORED_RESULT.parent)
    try:
        result = _decode(scorer._stable_read(fd, AUTHORED_RESULT.name, frozen.MAX_DOCUMENT))
    finally:
        os.close(fd)
    _require(result["plan_sha256"] == AUTHORED_PLAN_SHA and result["id"] == "qwen38-native-state-recovery-authored-003"
             and all(job["release"]["verified"] is True for job in result["costs"]["GPU_jobs"]))
    runtime = Runtime(WORKSPACE)
    try:
        runtime.service.plan = _load_private(WORKSPACE, sha)
        runtime.monitor(empty_port=True)
    finally:
        runtime.close()
    return _launch(sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("register", "serve", "episode", "run", "launch"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--owner")
    parser.add_argument("--config")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    if args.register:
        print(register())
    elif args.serve:
        _queue().serve(_job(args.batch, args.owner, args.plan_sha256))
    elif args.episode:
        return 0 if run_episode(args.config, args.config_sha256, root=WORKSPACE)["status"] == "saved_and_closed" else 1
    elif args.run:
        def interrupted(_signum, _frame):
            raise frozen.NativeExperienceUtilityError("official_panel_interrupted")

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        return 0 if run_study(args.plan_sha256)["status"] == "complete" else 1
    else:
        print(launch(args.plan_sha256))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("official_panel_execution_failed") from None
