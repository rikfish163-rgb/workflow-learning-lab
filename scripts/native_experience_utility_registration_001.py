"""Future train-only utility registration over a reviewed frozen experience.

The public plan and qualification proof are metadata claims, not evidence of a
completed source, writer or content review. Private preflight must invoke the
source-bound review loader before any utility admission. No plan, proof, private
file, model, task, queue or GPU operation is created by this module.
"""

import copy
import hashlib
import os
import re
import shutil
import socket
from pathlib import Path

from scripts import native_experience_registration_001 as writerreg
from scripts import native_source_registration as public
from scripts import native_source_registration_006 as source6
from scripts.native_appworld_worker_client import _encode

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-experience-utility-001"
PLAN = "artifacts/qwen38-native-experience-utility-plan-001.json"
QUALIFICATION_ID = "native-experience-utility-qualification-001"
QUALIFICATION_PROOF = "artifacts/native-experience-utility-qualification-001.json"
SOURCE_PLAN = "artifacts/qwen38-native-source-pilot-plan-006.json"
SOURCE_PLAN_SHA = "da4b4c4a4c8040a961ef0604ed19152bc5a3ea21790618afd19e694651c63ad3"
SOURCE_GATE_KIND = "ordinary_control_recovery_gate"
WRITER_ID = "qwen38-native-experience-writer-001"
WRITER_PLAN = "artifacts/qwen38-native-experience-writer-plan-001.json"
PORT = 18991
SERVICE_PLAN = "artifacts/qwen38-b3-service-plan-001.json"
SERVICE_SHA = "d7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5"
MODEL_ARM = "B3"
MODEL_ID = "qwen38-b3-matched-b3-001"
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
PRIVATE_STATE_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-experience-utility-001"
TMPDIR = PRIVATE_STATE_ROOT + "/runtime-tmp"
SOURCE_RESULT_PATH = PRIVATE_STATE_ROOT + "/native-source-pilot-006/study-result.json"
WRITER_RESULT_PATH = PRIVATE_STATE_ROOT + "/native-experience-writer-001/study-result.json"
REVIEW_MANIFEST_PATH = PRIVATE_STATE_ROOT + "/native-experience-writer-001/review-manifest.json"
MINIMUM_FREE_BYTES = 5 * 1024**3
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200
N_BATCHES = 14
TOTAL_GPU_SECONDS = N_BATCHES * BATCH_GPU_SECONDS
REQUIRED_NEW_SOURCES = frozenset({
    "scripts/native_experience_utility_registration_001.py", "tests/test_native_experience_utility_registration_001.py",
    "scripts/native_experience_utility_gpu_queue_001.py", "tests/test_native_experience_utility_gpu_queue_001.py",
    "scripts/run_native_experience_utility_001.py", "tests/test_run_native_experience_utility_001.py",
    "scripts/native_experience_utility_001.py", "tests/test_native_experience_utility_001.py",
    "scripts/exact_cluster_resampling.py", "tests/test_exact_resampling.py",
    "scripts/native_experience_content_001.py", "tests/test_native_experience_content_001.py",
    "scripts/native_experience_review_001.py", "tests/test_native_experience_review_001.py",
    "scripts/native_appworld_controller_006.py", "tests/test_native_appworld_controller_006.py",
    "scripts/native_appworld_worker_client_006.py", "tests/test_native_appworld_worker_client_006.py",
})
_ERROR = "native_experience_utility_registration_invalid"
_PLAN_FIELDS = frozenset({
    "id", "status", "source", "utility", "expected_pairs", "batches", "branch_orders", "identity",
    "model_arm", "model_id", "expected_backend_static", "port", "service_plan", "service_plan_sha256",
    "python", "gpu_policy", "gpu_policy_sha256", "source_sha256", "input_sha256", "runtime_sha256",
    "tmp_dir", "private_run_root", "gpu_seconds", "total_gpu_seconds", "admission_wait_seconds",
    "automatic_retry", "writer_authorized", "training_authorized", "utility_authorized", "dev_or_test_authorized",
})
_PROOF_FIELDS = frozenset({
    "id", "status", "source_plan_sha256", "source_result_sha256", "writer_plan_sha256",
    "writer_result_sha256", "review_manifest_sha256", "experience_bundle_id", "source_scenario_ids", "source_gate_kind",
})
_PRIVATE_FIELDS = frozenset({
    "private", "status", "source_plan_sha256", "source_result_sha256", "writer_plan_sha256",
    "writer_result_sha256", "review_manifest_sha256", "experience_bundle_id", "source_scenario_ids",
    "insertion_text", "private_files_sha256", "source_gate_kind",
})
public_read = public.public_read
_public_hash = public._public_hash
_absolute_hash = public._absolute_hash
_private_directory = public._private_directory


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def utility_policy():
    return {"comparison": "ordinary_experience_vs_no_experience", "split": "train", "pairs": 27, "scenarios": 9,
            "max_pairs_per_job": 2, "serial_gpu_jobs": 14, "environment_seed": 100, "generation_seed": 17,
            "max_generations": 60, "max_actions": 60, "max_api_requests": 60,
            "max_input_tokens": 15360, "max_new_tokens": 1024, "context_tokens": 16384, "temperature": 0.7,
            "max_prompt_tokens_total": 921600, "max_completion_tokens_total": 61440,
            "actor_seconds": 600, "setup_seconds": 120, "cleanup_seconds": 60,
            "metadata_rpc_timeout": "full_remaining_setup_120", "rpc_timeout_default_seconds": 60,
            "explicit_dispatch_and_replay_timeout": "existing_phase_and_worker_deadlines",
            "save_close_timeout": "existing_min_default60_remaining",
            "worker_wall_seconds": 780, "episode_unit_seconds": 900, "scorer_seconds": 120,
            "model_ready_seconds": 600, "source_verification_seconds": 180,
            "source_verification_scope": "per_pass", "release_seconds": 90, "shutdown_reserve_seconds": 150,
            "memory_placement": "user_message_index_2", "fresh_world_per_arm": True, "replay": False,
            "control_memory": False, "experience_memory": "frozen_reviewed_bundle",
            "same_model_backend_seed_budget": True, "bundle_frozen_before_utility_feedback": True,
            "score_feedback_to_actor": False, "separate_scorer_after_actor_exit": True,
            "require_complete_planned_pairs": True,
            "allowed_physical_gpus": [8, 9], "continuous_idle_seconds": 180,
            "fresh_idle_ticket_required": True, "project_lock_required": True, "allocator_cap_gib": 20,
            "maximum_simultaneous_project_gpus": 1, "request_window_seconds": 120,
            "intent_journal_reserve_seconds": 5,
            "parent_tasks_max": 256, "parent_memory_bytes": 6 * 1024**3, "parent_cpu_quota_percent": 400,
            "bootstrap_samples": 2000, "bootstrap_seed": 0, "minimum_scenarios": 8,
            "maximum_harm_rate": 0.05, "both_lower_bounds_positive": True,
            "known_budget_stop_with_saved_scored_receipt": "evaluable_boolean_score",
            "unknown_or_incomplete_evidence": "not_evaluable"}


def _source_plan(root):
    _require(writerreg.SOURCE_PLAN == SOURCE_PLAN and writerreg.SOURCE_PLAN_SHA == SOURCE_PLAN_SHA
             and writerreg.SOURCE_GATE_KIND == SOURCE_GATE_KIND
             and source6.ID == "qwen38-native-source-pilot-006" and source6.PLAN == SOURCE_PLAN)
    return writerreg._source_plan(root)


def _writer_plan(root, sha):
    _require(writerreg.ID == WRITER_ID and writerreg.PLAN == WRITER_PLAN
             and public._sha(sha) and hashlib.sha256(public_read(root, WRITER_PLAN)).hexdigest() == sha)
    return writerreg.load_registration(root, sha, private=False)


def _cohort(previous):
    cohort = previous["design"]["cohort"]
    _require(type(cohort) is dict and cohort.get("utility_scenarios") == 9
             and cohort.get("official_split") == "train" and cohort.get("project_unseen") is False
             and cohort.get("all_train_roles_already_exposed") is True)
    source_tasks, tasks = previous["task_ids"], cohort.get("utility_task_ids_reserved")
    _require(type(source_tasks) is list and len(source_tasks) == 12
             and all(type(task) is str and re.fullmatch(r"[a-f0-9]{7}_2", task) for task in source_tasks))
    origins = sorted({task[:-2] for task in source_tasks})
    _require(len(origins) == 12 and type(tasks) is list and len(tasks) == len(set(tasks)) == 27
             and all(type(task) is str and re.fullmatch(r"[a-f0-9]{7}_[123]", task) for task in tasks))
    scenarios = list(dict.fromkeys(task[:-2] for task in tasks))
    _require(len(scenarios) == 9 and tasks == [scenario + "_" + str(v) for scenario in scenarios for v in (1, 2, 3)]
             and not set(origins) & set(scenarios))
    pairs = [[task, task[:-2], 17] for task in tasks]
    batches = [list(range(i, min(i + 2, len(pairs)))) for i in range(0, len(pairs), 2)]
    orders = [["control", "experience"] if i % 2 == 0 else ["experience", "control"]
              for i in range(len(pairs))]
    return origins, pairs, batches, orders


def _qualification(root, expected_sha, origins):
    _require(public._sha(expected_sha))
    raw = public_read(root, QUALIFICATION_PROOF)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha)
    proof = public._json(raw)
    _require(type(proof) is dict and set(proof) == _PROOF_FIELDS and proof["id"] == QUALIFICATION_ID
             and proof["source_gate_kind"] == SOURCE_GATE_KIND
             and proof["status"] == "plausible_pending_utility" and proof["source_plan_sha256"] == SOURCE_PLAN_SHA
             and all(public._sha(proof[k]) for k in ("source_result_sha256", "writer_plan_sha256",
                                                      "writer_result_sha256", "review_manifest_sha256"))
             and type(proof["experience_bundle_id"]) is str
             and re.fullmatch(r"native-experience-001:[a-f0-9]{64}", proof["experience_bundle_id"])
             and public._same(proof["source_scenario_ids"], origins))
    return proof


def _expected_source(proof, proof_sha):
    return {"source_plan": SOURCE_PLAN, "source_plan_sha256": SOURCE_PLAN_SHA,
            "source_gate_kind": SOURCE_GATE_KIND,
            "source_result_sha256": proof["source_result_sha256"], "writer_plan": WRITER_PLAN,
            "writer_plan_sha256": proof["writer_plan_sha256"], "writer_result_sha256": proof["writer_result_sha256"],
            "review_manifest_sha256": proof["review_manifest_sha256"],
            "experience_bundle_id": proof["experience_bundle_id"],
            "source_scenario_ids": copy.deepcopy(proof["source_scenario_ids"]),
            "qualification_proof": {"path": QUALIFICATION_PROOF, "sha256": proof_sha}}


def _identity(previous, writer):
    actor_sha = writer["source_sha256"].get("scripts/native_appworld_actor_004.py")
    controller_sha = writer["source_sha256"].get("scripts/native_appworld_controller_006.py")
    client_sha = writer["source_sha256"].get("scripts/native_appworld_worker_client_006.py")
    _require(all(public._sha(digest) for digest in (actor_sha, controller_sha, client_sha)))
    scaffold = {"actor_source_sha256": actor_sha, "memory_placement": "user_message_index_2",
                "controller_source_sha256": controller_sha, "worker_client_source_sha256": client_sha,
                "metadata_rpc_timeout": "full_remaining_setup_120", "rpc_timeout_default_seconds": 60,
                "explicit_dispatch_and_replay_timeout": "existing_phase_and_worker_deadlines",
                "control_has_memory": False, "full_metadata": True}
    return {"model_id": MODEL_ID, "scaffold_id": hashlib.sha256(_encode(scaffold)).hexdigest(),
            "backend_id": hashlib.sha256(_encode(previous["expected_backend_static"])).hexdigest(),
            "budget_id": hashlib.sha256(_encode(utility_policy())).hexdigest()}


def _private_assets():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    for path in (PRIVATE_STATE_ROOT, TMPDIR, PRIVATE_RUN_ROOT):
        _private_directory(path)
    _require(shutil.disk_usage(TMPDIR).free >= MINIMUM_FREE_BYTES
             and shutil.disk_usage(PRIVATE_RUN_ROOT).free >= MINIMUM_FREE_BYTES)


def _load_reviewed_bundle(root, writer_plan_sha, writer_result_sha, review_manifest_sha):
    from scripts.native_experience_review_001 import load_reviewed_bundle

    return load_reviewed_bundle(root, writer_plan_sha, writer_result_sha, review_manifest_sha)


def _private_qualification(root, plan, proof):
    _private_assets()
    _require(all(_absolute_hash(name) == digest for name, digest in plan["runtime_sha256"].items()))
    reviewed = _load_reviewed_bundle(root, proof["writer_plan_sha256"], proof["writer_result_sha256"],
                                     proof["review_manifest_sha256"])
    _require(type(reviewed) is dict and set(reviewed) == _PRIVATE_FIELDS and reviewed["private"] is True
             and reviewed["status"] == "plausible_pending_utility" and reviewed["source_gate_kind"] == SOURCE_GATE_KIND
             and all(public._same(reviewed[k], proof[k]) for k in (
                 "source_gate_kind", "source_plan_sha256", "source_result_sha256", "writer_plan_sha256", "writer_result_sha256",
                 "review_manifest_sha256", "experience_bundle_id", "source_scenario_ids")))
    text = reviewed["insertion_text"]
    _require(type(text) is str and bool(text.strip()) and 1 <= len(text.encode("utf-8")) <= 65536
             and reviewed["experience_bundle_id"] == "native-experience-001:" + hashlib.sha256(text.encode()).hexdigest())
    files = reviewed["private_files_sha256"]
    _require(type(files) is dict and bool(files) and files.get(SOURCE_RESULT_PATH) == proof["source_result_sha256"]
             and files.get(WRITER_RESULT_PATH) == proof["writer_result_sha256"]
             and files.get(REVIEW_MANIFEST_PATH) == proof["review_manifest_sha256"])
    for name, digest in files.items():
        _require(type(name) is str and Path(name).is_absolute() and os.path.normpath(name) == name
                 and Path(PRIVATE_STATE_ROOT) in Path(name).parents and ".." not in Path(name).parts
                 and public._sha(digest))
    return reviewed


def _validate_previous(writer, previous, proof):
    _require(type(writer) is dict and writer.get("id") == WRITER_ID and writer.get("status") == "registered"
             and writer.get("writer_authorized") is True
             and all(writer.get(k) is False for k in ("training_authorized", "utility_authorized", "dev_or_test_authorized"))
             and writer.get("model_arm") == MODEL_ARM and writer.get("model_id") == MODEL_ID
             and writer.get("port") == PORT and writer.get("service_plan") == SERVICE_PLAN
             and writer.get("service_plan_sha256") == SERVICE_SHA and writer.get("python") == PYTHON
             and writer.get("tmp_dir") == TMPDIR and type(writer.get("source")) is dict
             and writer["source"].get("plan_sha256") == SOURCE_PLAN_SHA
             and writer["source"].get("source_gate_kind") == SOURCE_GATE_KIND
             and writer["source"].get("result_sha256") == proof["source_result_sha256"]
             and public._same(writer.get("expected_backend_static"), previous["expected_backend_static"]))


def validate_registration(root, plan, *, private=False):
    try:
        _require(type(private) is bool and type(plan) is dict and set(plan) == _PLAN_FIELDS)
        public._values(plan)
        previous = _source_plan(root)
        origins, pairs, batches, orders = _cohort(previous)
        inputs = plan["input_sha256"]
        public._binding_map(inputs)
        _require(public._sha(inputs.get(QUALIFICATION_PROOF)))
        proof = _qualification(root, inputs[QUALIFICATION_PROOF], origins)
        writer = _writer_plan(root, proof["writer_plan_sha256"])
        _validate_previous(writer, previous, proof)
        _require(plan["id"] == ID and plan["status"] == "registered"
                 and public._same(plan["source"], _expected_source(proof, inputs[QUALIFICATION_PROOF]))
                 and public._same(plan["utility"], utility_policy()) and public._same(plan["expected_pairs"], pairs)
                 and public._same(plan["batches"], batches) and public._same(plan["branch_orders"], orders)
                 and public._same(plan["identity"], _identity(previous, writer))
                 and plan["model_arm"] == MODEL_ARM and plan["model_id"] == MODEL_ID
                 and type(plan["port"]) is int and plan["port"] == PORT
                 and plan["service_plan"] == SERVICE_PLAN and plan["service_plan_sha256"] == SERVICE_SHA
                 and plan["python"] == PYTHON and plan["tmp_dir"] == TMPDIR
                 and plan["private_run_root"] == PRIVATE_RUN_ROOT
                 and type(plan["gpu_seconds"]) is int and plan["gpu_seconds"] == BATCH_GPU_SECONDS
                 and type(plan["total_gpu_seconds"]) is int and plan["total_gpu_seconds"] == TOTAL_GPU_SECONDS
                 and type(plan["admission_wait_seconds"]) is int and plan["admission_wait_seconds"] == WAIT_SECONDS
                 and plan["automatic_retry"] is False and plan["utility_authorized"] is True
                 and all(plan[k] is False for k in ("writer_authorized", "training_authorized", "dev_or_test_authorized"))
                 and public._same(plan["gpu_policy"], writer["gpu_policy"])
                 and plan["gpu_policy_sha256"] == writer["gpu_policy_sha256"]
                 and public._same(plan["expected_backend_static"], previous["expected_backend_static"]))
        sources, runtime = plan["source_sha256"], plan["runtime_sha256"]
        public._binding_map(sources)
        public._binding_map(runtime, runtime=True)
        for field, current in (("source_sha256", sources), ("input_sha256", inputs)):
            _require(type(writer[field]) is dict
                     and all(current.get(name) == digest for name, digest in writer[field].items()))
        _require(public._same(runtime, writer["runtime_sha256"]) and REQUIRED_NEW_SOURCES <= sources.keys()
                 and inputs.get(WRITER_PLAN) == proof["writer_plan_sha256"])
        for name, digest in {**sources, **inputs}.items():
            _require(name not in sources or name not in inputs or sources[name] == inputs[name])
            _require(_public_hash(root, name) == digest)
        if private:
            _private_qualification(root, plan, proof)
        return copy.deepcopy(plan)
    except Exception:
        raise ValueError(_ERROR) from None


def build_registration(root, *, source_sha256, input_sha256, runtime_sha256,
                       qualification_proof_sha256, private=False):
    """Build a plan in memory; the caller owns any later exclusive writing."""
    try:
        previous = _source_plan(root)
        origins, pairs, batches, orders = _cohort(previous)
        proof = _qualification(root, qualification_proof_sha256, origins)
        writer = _writer_plan(root, proof["writer_plan_sha256"])
        plan = {"id": ID, "status": "registered", "source": _expected_source(proof, qualification_proof_sha256),
                "utility": utility_policy(), "expected_pairs": pairs, "batches": batches, "branch_orders": orders,
                "identity": _identity(previous, writer), "model_arm": MODEL_ARM, "model_id": MODEL_ID,
                "expected_backend_static": copy.deepcopy(previous["expected_backend_static"]),
                "port": PORT, "service_plan": SERVICE_PLAN, "service_plan_sha256": SERVICE_SHA, "python": PYTHON,
                "gpu_policy": copy.deepcopy(writer["gpu_policy"]), "gpu_policy_sha256": writer["gpu_policy_sha256"],
                "source_sha256": copy.deepcopy(source_sha256), "input_sha256": copy.deepcopy(input_sha256),
                "runtime_sha256": copy.deepcopy(runtime_sha256), "tmp_dir": TMPDIR, "private_run_root": PRIVATE_RUN_ROOT,
                "gpu_seconds": BATCH_GPU_SECONDS, "total_gpu_seconds": TOTAL_GPU_SECONDS,
                "admission_wait_seconds": WAIT_SECONDS, "automatic_retry": False,
                "writer_authorized": False, "training_authorized": False,
                "utility_authorized": True, "dev_or_test_authorized": False}
        return validate_registration(root, plan, private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def load_registration(root, sha, *, private=False):
    try:
        _require(public._sha(sha))
        raw = public_read(root, PLAN)
        _require(hashlib.sha256(raw).hexdigest() == sha)
        return validate_registration(root, public._json(raw), private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def private_preflight(root, sha):
    """Return the frozen plan and private reviewed bundle after one verifier pass."""
    try:
        plan = load_registration(root, sha, private=False)
        previous = _source_plan(root)
        origins, _pairs, _batches, _orders = _cohort(previous)
        proof = _qualification(root, plan["source"]["qualification_proof"]["sha256"], origins)
        bundle = _private_qualification(root, plan, proof)
        _require(public._same(load_registration(root, sha, private=False), plan))
        return plan, bundle
    except Exception:
        raise ValueError(_ERROR) from None
