"""Run a registered authored arm under an already admitted, bound GPU lease."""

import hashlib
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import native_appworld_controller_006 as controller
from scripts import native_appworld_scorer as scorer
from scripts import native_state_recovery_authored_runtime_001 as assets
from scripts import native_state_recovery_transition_review_002 as review
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import run_native_experience_utility_001 as utility
from scripts.native_appworld_model_client_004 import NativeModel
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _private_tree, _same
from scripts.native_source_registration import _public_hash, public_read, static_backend

ID = "qwen38-native-state-recovery-authored-001"
DRAFT_SUMMARY = "artifacts/native-state-recovery-comparison-drafts-001.json"
DRAFT_SUMMARY_SHA256 = "7db242b9c0fa42bcce6a08ef0c2d432701f6ecb0cd72490d96174dd99202cf86"
DEPLOYMENT = Path("/home/hts/workflow-learning-lab")
REGISTRATION = "study-registration.json"
EPISODE_SOURCE = "scripts/native_state_recovery_authored_episode_001.py"
MAX_CONFIG_BYTES = 1_000_000
_HASH = re.compile(r"[a-f0-9]{64}")
_OWNER = re.compile(r"[a-f0-9]{32}")


def _require(condition):
    if not condition:
        raise ValueError("native_authored_episode_binding_invalid") from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, expected_sha):
    _require(type(expected_sha) is str and _HASH.fullmatch(expected_sha) is not None)
    fd = _private_tree(path.parent)
    try:
        raw = scorer._stable_read(fd, path.name, MAX_CONFIG_BYTES)
    finally:
        os.close(fd)
    _require(_hash(raw) == expected_sha)
    return _decode(raw)


def _validate_registration(plan, root, selected):
    _require(type(plan) is dict and set(plan) == {
        "id", "status", "utility_authorized", "dev_or_test_authorized",
        "automatic_retry", "asset_root", "draft", "episode_source_sha256",
    } and plan["id"] == ID and plan["status"] == "registered"
        and plan["utility_authorized"] is True
        and plan["dev_or_test_authorized"] is plan["automatic_retry"] is False
        and plan["asset_root"] == str(root)
        and plan["episode_source_sha256"] == _public_hash(WORKSPACE, EPISODE_SOURCE))
    draft = plan["draft"]
    _require(_hash(_encode(draft)) == selected["draft_sha256"]
             and draft["split"] == "authored" and draft["scorer"] == "authored_terminal_state"
             and draft["test_set_use_if_run"] is False)
    return draft


def _validate_configuration(config, path, root, draft, plan_sha, queue):
    _require(type(config) is dict and set(config) == {
        "plan_sha256", "pair_index", "arm", "job", "expected_backend",
    } and config["plan_sha256"] == plan_sha and queue.ID == ID and queue.REMOTE == DEPLOYMENT)
    index, arm = config["pair_index"], config["arm"]
    _require(type(index) is int and 0 <= index < len(draft["expected_pairs"]) and arm in comparison.ARMS)
    job = config["job"]
    _require(type(job) is dict and type(job.get("owner")) is str and _OWNER.fullmatch(job["owner"]) is not None)
    batch = index // 2
    _require(index in draft["batches"][batch] and arm in draft["arm_orders"][index]
             and job.get("id") == f"{ID}-comparison-{batch}" and job.get("kind") == "comparison"
             and job.get("max_runtime_seconds") == draft["gpu_seconds_per_batch_ceiling"]
             and _same(job, queue.make_job(batch, job["owner"], plan_sha)))
    experiment = f"pair-{index:02d}-{arm}-{job['owner']}"
    _require(path == root / "configs" / f"{experiment}.json")
    task, _scenario, seed = draft["expected_pairs"][index]
    return task, seed, experiment


def _lease_check(reader, queue, config, *, gpu_seconds, reserve_seconds, launch_seconds, clock=time.time):
    def read():
        state, _ready, identity = reader._queue_binding(config, queue.queue_operation)
        age = clock() - state["job"]["started_epoch"]
        _require(0 <= age < gpu_seconds - reserve_seconds)
        return identity, gpu_seconds - age

    bound, remaining = read()
    _require(remaining >= launch_seconds + reserve_seconds)

    def check():
        identity, _remaining = read()
        _require(_same(identity, bound))

    return check


def _prior():
    raw = public_read(WORKSPACE, comparison.PRIOR_PLAN)
    _require(_hash(raw) == comparison.INPUTS[comparison.PRIOR_PLAN])
    return _decode(raw)


def _bundle(prior, draft):
    source = prior["source"]
    _require(draft["review_manifest_sha256"] == source["review_manifest_sha256"]
             and draft["writer_result_sha256"] == source["writer_result_sha256"])
    bundle = review.load_reviewed_bundle(source["writer_plan_sha256"], source["writer_result_sha256"],
                                         source["review_manifest_sha256"])
    _require(bundle["private"] is True and bundle["status"] == "plausible_pending_utility"
             and bundle["experience_bundle_id"] == source["experience_bundle_id"]
             and bundle["experience_bundle_id"].endswith(":" + _hash(bundle["insertion_text"].encode())))
    return bundle["insertion_text"]


def run_episode(root, config_path, config_sha, plan_sha, *, private_plan_sha, queue, _reader=None,
                _model_factory=None, _controller=None):
    """Entry for the trusted coordinator. This function cannot enqueue or launch a GPU service."""
    _require(Path.cwd() == WORKSPACE and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    root, path = Path(root), Path(config_path)
    _require(path.parent == root / "configs")
    raw = public_read(WORKSPACE, DRAFT_SUMMARY)
    _require(_hash(raw) == DRAFT_SUMMARY_SHA256)
    selected = _decode(raw)["options"]["authored"]
    plan = _read(root / REGISTRATION, private_plan_sha)
    draft = _validate_registration(plan, root, selected)
    _require(all(_public_hash(WORKSPACE, source) == digest
                 for source, digest in draft["source_sha256"].items()))
    root = assets.verify_assets(root, draft["initial_state_manifest_sha256"])
    config = _read(path, config_sha)
    task, seed, experiment = _validate_configuration(config, path, root, draft, plan_sha, queue)
    _require(scorer._main_pid(assets.parent_unit(root, experiment)) == os.getpid())
    prior = _prior()
    _require(draft["model_id"] == prior["model_id"] and draft["model_arm"] == prior["model_arm"])
    policy = draft["protocol"]
    facade = SimpleNamespace(
        MODEL_ARM=prior["model_arm"], static_backend=static_backend,
        public_read=public_read,
    )
    reader = _reader or utility._reader(root, facade, queue)
    reader._backend(prior, config["expected_backend"])
    check = _lease_check(reader, queue, config, gpu_seconds=draft["gpu_seconds_per_batch_ceiling"],
                         reserve_seconds=policy["shutdown_reserve_seconds"],
                         launch_seconds=policy["episode_unit_seconds"] + policy["scorer_seconds"])
    text = _bundle(prior, draft)
    check()
    run_dir = root / "runs" / experiment
    utility._create_episode_directory(reader, root / "runs", run_dir)
    model = (_model_factory or NativeModel)(
        f"http://127.0.0.1:{prior['port']}/v1", draft["model_id"],
        expected_backend=config["expected_backend"], max_tokens=policy["max_new_tokens"],
        context_tokens=policy["context_tokens"], temperature=policy["temperature"],
        timeout=policy["request_window_seconds"], deadline=time.monotonic() + policy["worker_wall_seconds"],
    )
    model = utility._BoundModel(model, check)
    hint = text if config["arm"] == "experience" else None
    result = (_controller or controller.run_task)(
        task, experiment, run_dir, limits=utility._limits(policy), seed=seed,
        model=model, token_counter=model.count_input_tokens, review_hint=hint,
        rpc_timeout=policy["rpc_timeout_default_seconds"], worker_wall_seconds=policy["worker_wall_seconds"],
        setup_seconds=policy["setup_seconds"], cleanup_seconds=policy["cleanup_seconds"],
        is_policy_episode=True, _worker_factory=assets.worker_factory(root, draft["initial_state_manifest_sha256"]),
    )
    check()
    _require(result["task_id"] == task and result["experiment_name"] == experiment)
    return result
