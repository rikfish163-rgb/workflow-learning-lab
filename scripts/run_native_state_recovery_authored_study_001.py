"""Coordinate the authored comparison under a separately authorized registration."""

import argparse
import hashlib
import os
import re
import socket
import subprocess
from dataclasses import asdict
from functools import cache
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

from scripts import native_appworld_scorer as scorer
from scripts import native_state_recovery_authored_episode_001 as episode
from scripts import native_state_recovery_authored_runtime_001 as assets
from scripts import native_state_recovery_transition_review_002 as review
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import run_native_experience_utility_001 as frozen
from scripts.native_appworld_worker_client import (
    PYTHON,
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _exclusive,
    _private_tree,
    _same,
)
from scripts.native_source_registration import _public_hash, public_read
from scripts.native_state_recovery_unit_lifecycle_001 import unit_status_with_exit_recheck
from scripts.verification_token_counter import _message_payload

ID = episode.ID
PLAN = "artifacts/qwen38-native-state-recovery-authored-model-plan-001.json"
PRIVATE = RUNTIME_TMP / "authored-panel-model-001"
MODULE = "scripts.run_native_state_recovery_authored_study_001"
SOURCE = MODULE.replace(".", "/") + ".py"
QUEUE_SOURCE = "scripts/native_source_gpu_queue_002.py"
QUEUE_SHA = "ae221cd0ad1a88c2f4833cb605c501e831c5068ad8e3aeca4491f02c31881419"
LOOP_SOURCE = "scripts/run_native_experience_utility_001.py"
LOOP_SHA = "aa5e8184098900f7575c81e3ab1b39218f4e9b8b5db0df62588789c87aa02876"
REQUIRED_SOURCES = {
    SOURCE, episode.EPISODE_SOURCE, QUEUE_SOURCE, LOOP_SOURCE,
    "scripts/native_state_recovery_authored_runtime_001.py",
    "scripts/native_state_recovery_authored_panel_001.py",
    "scripts/probe_native_state_recovery_authored_panel_001.py",
    "scripts/probe_native_state_recovery_authored_world_001.py",
    "scripts/prepare_native_state_recovery_comparison_001.py",
    "scripts/prepare_native_state_recovery_holdout_001.py",
    "scripts/native_state_recovery_unit_lifecycle_001.py",
}


def _require(value):
    if not value:
        raise ValueError("native_authored_study_binding_invalid") from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def load_public(root, sha):
    """GPU service reads this metadata, never the authored task roster or lesson text."""
    _require(root in (WORKSPACE, episode.DEPLOYMENT))
    raw = public_read(root, PLAN)
    _require(_hash(raw) == sha)
    plan = _decode(raw)
    prior = episode._prior()
    _require(set(plan) == {"id", "status", "utility_authorized", "private_sha256",
                          "draft_summary_sha256", "source_sha256"}
             and plan["id"] == ID and plan["status"] == "registered" and plan["utility_authorized"] is True
             and re.fullmatch(r"[a-f0-9]{64}", plan["private_sha256"]) is not None
             and plan["draft_summary_sha256"] == episode.DRAFT_SUMMARY_SHA256)
    sources = plan["source_sha256"]
    _require(set(sources) == REQUIRED_SOURCES and sources[QUEUE_SOURCE] == QUEUE_SHA
             and sources[LOOP_SOURCE] == LOOP_SHA
             and all(_public_hash(root, name) == digest for name, digest in sources.items()))
    return {**plan, "expected_backend_static": prior["expected_backend_static"]}


def _job(batch, owner, sha):
    _require(type(batch) is int and 0 <= batch < 14 and re.fullmatch(r"[a-f0-9]{32}", owner) is not None
             and re.fullmatch(r"[a-f0-9]{64}", sha) is not None)
    return {"id": f"{ID}-comparison-{batch}", "kind": "comparison", "batch_index": batch,
            "owner": owner, "plan_sha256": sha, "max_runtime_seconds": 7200,
            "argv": [PYTHON, "-B", "-m", MODULE, "--serve", "--plan-sha256", sha,
                     "--batch", str(batch), "--owner", owner]}


@cache
def _queue():
    root = Path(__file__).resolve().parents[1]
    raw = public_read(root, QUEUE_SOURCE)
    _require(_hash(raw) == QUEUE_SHA)
    queue = ModuleType("scripts._authored_study_queue")
    queue.__file__ = str(root / QUEUE_SOURCE)
    exec(compile(raw, queue.__file__, "exec", dont_inherit=True), vars(queue))
    _require(queue.REMOTE == episode.DEPLOYMENT and queue.PYTHON == PYTHON
             and queue.BATCH_GPU_SECONDS == 7200 and queue.WAIT_SECONDS == 86400)
    queue.ID, queue.PLAN = ID, PLAN
    queue.GUARD_NAME = "native-state-recovery-transition-utility-003-operations-001.lock"
    queue.make_job, queue.load_registration = _job, load_public
    return queue


def _registered_bundle(prior):
    source = prior["source"]
    return review.load_reviewed_bundle(source["writer_plan_sha256"], source["writer_result_sha256"],
                                       source["review_manifest_sha256"])


def _preflight(root, sha):
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and root == WORKSPACE and Path.cwd() == root)
    public = load_public(root, sha)
    _require(_same(public, load_public(episode.DEPLOYMENT, sha)))
    raw = public_read(root, episode.DRAFT_SUMMARY)
    _require(_hash(raw) == episode.DRAFT_SUMMARY_SHA256)
    selected = _decode(raw)["options"]["authored"]
    registration = episode._read(PRIVATE / episode.REGISTRATION, public["private_sha256"])
    draft = episode._validate_registration(registration, PRIVATE, selected)
    assets.verify_assets(PRIVATE, draft["initial_state_manifest_sha256"])
    for name in ("configs", "runs", "scores"):
        fd = _private_tree(PRIVATE / name)
        os.close(fd)
    _require(all(_public_hash(root, name) == digest for name, digest in draft["source_sha256"].items()))
    prior = episode._prior()
    plan = {**prior, "id": ID, "private_run_root": str(PRIVATE), "draft": draft,
            "utility": draft["protocol"], "expected_pairs": draft["expected_pairs"],
            "batches": draft["batches"], "branch_orders": draft["arm_orders"],
            "public_sha256": sha, "private_sha256": public["private_sha256"]}
    return plan, _registered_bundle(prior)


def _bind(function, **names):
    return FunctionType(function.__code__, {**function.__globals__, **names},
                        function.__name__, function.__defaults__, function.__closure__)


def _unit_command(kind, unit, arguments, *, root, log_path):
    role = "score" if kind == "scorer" else "parent"
    names = re.compile(rf"wll-authored-{role}-[a-f0-9]{{32}}\.service", re.ASCII)
    base_command = _bind(frozen._core().unit_command, _UNIT=names)
    builder = _bind(frozen.unit_command, _core=lambda: SimpleNamespace(unit_command=base_command))
    command = builder(kind, unit, arguments, root=root, log_path=log_path)
    old = "scripts.native_appworld_scorer" if kind == "scorer" else "scripts.run_native_experience_utility_001"
    positions = [i for i, argument in enumerate(command) if old in argument]
    _require(len(positions) == 1 and kind in ("episode", "scorer", "pilot"))
    index = positions[0]
    if kind == "scorer":
        command[index] = command[index].replace(old, assets.__name__)
    else:
        command[index] = command[index].replace(old, MODULE)
    return command


class Runtime(frozen._Runtime):
    def __init__(self, root):
        prior = episode._prior()
        registry = SimpleNamespace(
            ID=ID, PLAN=PLAN, MODEL_ID=prior["model_id"], MODEL_ARM=prior["model_arm"],
            PORT=prior["port"], PYTHON=PYTHON, TMPDIR=str(RUNTIME_TMP), PRIVATE_RUN_ROOT=str(PRIVATE),
            public_read=public_read, _public_hash=_public_hash,
        )
        services = _bind(frozen._services, ID=ID, PLAN=PLAN, _registry=lambda: registry,
                         _queue=_queue, unit_command=_unit_command)
        self.root = Path(root)
        self.service, self.reader = services(self.root)
        self.service.unit_status = self._recheck(self.service.unit_status)
        self.verification_audits = []

    @staticmethod
    def _recheck(original):
        return lambda spec: unit_status_with_exit_recheck(original, spec)

    def preflight(self, sha):
        self.service.plan, bundle = _preflight(self.root, sha)
        return self.service.plan, bundle

    def check_bundle(self, sha, binding, *, gpu_remaining):
        started = self.now()
        _require(sha == self.plan["public_sha256"] and gpu_remaining > 150
                 and binding["path"] == str(PRIVATE / "reviewed-bundle.json"))
        raw = self.read(binding["path"])
        _require(_hash(raw) == binding["sha256"])
        bundle = _registered_bundle(self.plan)
        elapsed = self.now() - started
        _require(_same(bundle, _decode(raw)) and elapsed <= min(180, gpu_remaining - 150))
        self.verification_audits.append({"wall_seconds": elapsed, "returned": True})
        return bundle

    def binding(self, job, service):
        state, _ready, identity = self.reader._queue_binding(
            {"job": job, "expected_backend": service["backend"]}, _queue().queue_operation,
        )
        _require(_same(identity, service["identity"]))
        return state

    def release(self, job, pids):
        result = _queue().fenced_release(self.service, job, pids)
        fresh = self.queue("status", job, pids)
        _require(result["verified"] is True and fresh["owned"] is fresh["monitor_live"] is True
                 and fresh["admission_cancelled"] is True and not fresh["model_pids"]
                 and _same(fresh["job"], result["state"]["job"])
                 and fresh["epoch"] >= result["state"]["epoch"])
        if fresh["job"] is not None:
            _require(fresh["released"] is True)
        return {**result, "state": fresh}

    def preflight_failure(self, report):
        # A missing registration must not create a new private study namespace.
        if PRIVATE.is_dir():
            return super().preflight_failure(report)

    def create_score_directory(self, path):
        _require(path.parent == PRIVATE / "scores")
        fd = _private_tree(path.parent)
        try:
            os.mkdir(path.name, 0o700, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)


def _config(runtime, root, name, value):
    directory = root / "configs"
    fd = _private_tree(directory)
    try:
        with _exclusive(fd, name) as file:
            runtime.service._write(file, value)
        os.fsync(fd)
    finally:
        os.close(fd)
    path = directory / name
    return str(path), _hash(runtime.read(path))


def _partial_cost(runtime, core, run_dir):
    receipt, journal = {}, []
    try:
        receipt = _decode(runtime.read(run_dir / "execution-receipt.json"))
    except Exception:
        pass
    try:
        raw = runtime.read(run_dir / "controller.jsonl")
        journal = core._journal(raw[:raw.rfind(b"\n") + 1])
    except Exception:
        pass
    return core._costs(receipt, journal)


def _run_arm(runtime, plan, sha, _bundle, job, service, index, arm, row, guard, report, expected_metadata):
    root = Path(plan["private_run_root"])
    task, _scene, seed = plan["expected_pairs"][index]
    experiment = f"pair-{index:02d}-{arm}-{job['owner']}"
    run_dir = root / "runs" / experiment
    parent, score_parent = None, None
    started, score_cost = runtime.now(), None
    cost = {"pair_index": index, "arm": arm, "counts_complete": False, "unknown_cost": True}
    core = frozen._core()
    try:
        path, digest = _config(runtime, root, experiment + ".json", {
            "plan_sha256": sha, "pair_index": index, "arm": arm, "job": job, "expected_backend": service["backend"],
        })
        parent, _ = core._unit(runtime, "episode", assets.parent_unit(root, experiment),
            ["--config", path, "--config-sha256", digest, "--plan-sha256", sha], guard=guard)
        evidence = runtime.artifacts(run_dir)
        receipt, journal = evidence["receipt"], evidence["journal"]
        cost.update(core._costs(receipt, journal))
        _require(cost["counts_complete"] is True and receipt["status"] == "saved_and_closed"
                 and receipt["task_id"] == task and receipt["experiment_name"] == experiment
                 and receipt["is_policy_episode"] is True and receipt["worker_exit_code"] == 0
                 and receipt["replay_outcome"] is None)
        intent = core._one(journal, "controller_intent")
        metadata = core._one(journal, "metadata_result")["reply"]["result"]
        metadata_sha = _hash(_encode(metadata))
        _require(intent["seed"] == seed and intent["limits"] == asdict(frozen._limits(plan["utility"]))
                 and intent["review_hint_present"] is (arm == "experience")
                 and (expected_metadata is None or metadata_sha == expected_metadata))
        hint = report["_insertion_text"] if arm == "experience" else None
        messages = frozen.actor.build_actor_messages(metadata, review_hint=hint)
        core.actor_evidence(receipt, journal, expected_context={
            "metadata_sha256": metadata_sha, "messages_sha256": _message_payload(messages)[1],
        }, review_hint=hint)
        _require(core._one(journal, "actor_start")["seed"] == seed
                 and runtime.unit_status(parent)["main_pid"] == 0
                 and runtime.cleanup_worker(run_dir, task, experiment)["verified"] is True)
        row.update(closed=True, is_policy_episode=True, worker_exit_code=0)
        score_dir = root / "scores" / experiment
        runtime.create_score_directory(score_dir)
        score_started = runtime.now()
        score_cost = {"pair_index": index, "arm": arm, "unknown_cost": True}
        binding = evidence["receipt_binding"]
        score_parent, _ = core._unit(runtime, "scorer",
            assets.parent_unit(root, "scorer-" + experiment).replace("-parent-", "-score-"),
            ["--asset-root", str(root), "--initial-sha256", plan["draft"]["initial_state_manifest_sha256"],
             "--receipt", binding["path"], "--receipt-sha256", binding["sha256"],
             "--out-dir", str(score_dir)], guard=guard)
        score = runtime.score_result(score_dir)
        _require(score["status"] == "scored" and score["is_policy_episode"] is True
                 and type(score["success"]) is bool and score["receipt_sha256"] == binding["sha256"]
                 and score["manifest_sha256"] == receipt["snapshot"]["manifest_sha256"])
        score_cost["unknown_cost"] = False
        row.update(status="known", score_status="scored", success=score["success"],
                   outcome_unknown=False, unknown_cost=False)
        report["executions"].append({"pair_index": index, "arm": arm, "receipt_binding": binding,
            "metadata_sha256": metadata_sha, "actor_status": receipt["actor_outcome"]["status"]})
        return metadata_sha
    except BaseException:
        row.update(status="unknown", score_status="unknown", success=None, outcome_unknown=True, unknown_cost=True)
        raise
    finally:
        if not cost["counts_complete"]:
            try:
                cost.update(_partial_cost(runtime, core, run_dir), unknown_cost=True, counts_complete=False)
            except Exception:
                cost["partial_audit_unavailable"] = True
        if score_cost is not None:
            score_cost["wall_seconds"] = runtime.now() - score_started
            report["costs"]["scorers"].append(score_cost)
        try:
            if score_parent is not None:
                _require(runtime.stop_unit(score_parent)["verified"] is True)
        finally:
            try:
                _require(runtime.cleanup_worker(run_dir, task, experiment)["verified"] is True)
            finally:
                if parent is not None:
                    _require(runtime.stop_unit(parent)["verified"] is True)
                cost["parent_wall_seconds"] = runtime.now() - started
                report["costs"]["episodes"].append(cost)


def _summary(plan, rows):
    indexes = {task: index for index, (task, _, _) in enumerate(plan["expected_pairs"])}
    scores = [[indexes[row["task_id"]], row["arm"], row["success"]] for row in rows
              if row["status"] == "known" and row["closed"] is row["is_policy_episode"] is True
              and row["worker_exit_code"] == 0 and row["score_status"] == "scored"
              and row["outcome_unknown"] is row["unknown_cost"] is False]
    return comparison.summarize(plan["draft"], scores)


def _registered(plan, bundle):
    _require(plan["id"] == ID and plan["status"] == "registered" and plan["utility_authorized"] is True
             and plan["dev_or_test_authorized"] is plan["automatic_retry"] is False
             and bundle["private"] is True and bundle["status"] == "plausible_pending_utility"
             and bundle["experience_bundle_id"] == plan["source"]["experience_bundle_id"]
             and _summary(plan, [])["planned_pairs"] == 27)
    return bundle["insertion_text"]


def _summaries(plan):
    return [{**row, "split": "authored"} for row in frozen._summaries(plan)]


def run_study(sha, *, runtime=None, arm_runner=_run_arm):
    _require(_public_hash(WORKSPACE, LOOP_SOURCE) == LOOP_SHA)
    if runtime is None:
        _require(scorer._main_pid(_coordinator_unit(sha)) == os.getpid())
        runtime = Runtime(WORKSPACE)
    statistics = SimpleNamespace(evaluate_utility=lambda _pairs, rows, **_: _summary(runtime.plan, rows))
    loop = _bind(frozen.run_utility, ID=ID, _registered=_registered, _summaries=_summaries,
                 _run_arm=arm_runner, _make_job=_job, utility=statistics)
    return loop(sha, root=WORKSPACE, _runtime=runtime)


def _coordinator_unit(sha):
    _require(type(sha) is str and re.fullmatch(r"[a-f0-9]{64}", sha) is not None)
    return assets.parent_unit(PRIVATE, "study-" + sha)


def launch_study(sha):
    """Start the bounded CPU coordinator only for an authorized, verified plan."""
    _preflight(WORKSPACE, sha)
    unit = _coordinator_unit(sha)
    log = PRIVATE / "utility-parent.log"
    command = _unit_command("pilot", unit, ["--run", "--plan-sha256", sha], root=WORKSPACE, log_path=log)
    fd = _private_tree(PRIVATE)
    try:
        with _exclusive(fd, "parent-launch-intent.json") as intent:
            frozen._core()._Runtime._write(intent, {"unit": unit, "plan_sha256": sha, "argv": command})
        with _exclusive(fd, log.name):
            pass
        os.fsync(fd)
    finally:
        os.close(fd)
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=10, check=False)
    _require(result.returncode == 0)
    return unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--serve", action="store_true")
    modes.add_argument("--episode", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--launch", action="store_true")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--owner")
    parser.add_argument("--config")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    if args.serve:
        _queue().serve(_job(args.batch, args.owner, args.plan_sha256))
        return 0
    if args.launch:
        print(launch_study(args.plan_sha256))
        return 0
    if args.run:
        return 0 if run_study(args.plan_sha256)["status"] == "complete" else 1
    public = load_public(WORKSPACE, args.plan_sha256)
    result = episode.run_episode(PRIVATE, args.config, args.config_sha256, args.plan_sha256,
                                 private_plan_sha=public["private_sha256"], queue=_queue())
    return 0 if result["status"] == "saved_and_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
