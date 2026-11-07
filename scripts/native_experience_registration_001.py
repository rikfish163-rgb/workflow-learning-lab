"""Read-only builder and validator for a future native experience writer.

Public validation authenticates source and metadata bindings only; it cannot
attest that the public qualification claim is true. A coordinator must run
``private=True`` before admission, then bind its actual writer inputs to that
fresh private recomputation. This module writes no plan, proof or private data.
"""

import copy
import hashlib
import os
import shutil
import socket
from pathlib import Path

from scripts import native_source_registration as public
from scripts import native_source_registration_006 as source6
from scripts.native_appworld_worker_client import _decode, _encode
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-experience-writer-001"
PLAN = "artifacts/qwen38-native-experience-writer-plan-001.json"
SOURCE_PLAN = "artifacts/qwen38-native-source-pilot-plan-006.json"
SOURCE_PLAN_SHA = "da4b4c4a4c8040a961ef0604ed19152bc5a3ea21790618afd19e694651c63ad3"
SOURCE_GATE_KIND = "ordinary_control_recovery_gate"
SOURCE_RESULT_PATH = source6.PRIVATE_RUN_ROOT + "/study-result.json"
QUALIFICATION_ID = "native-experience-writer-qualification-001"
QUALIFICATION_PROOF = "artifacts/native-experience-writer-qualification-001.json"
PORT = 18991
SERVICE_PLAN = "artifacts/qwen38-b3-service-plan-001.json"
SERVICE_SHA = "d7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5"
MODEL_ARM = "B3"
MODEL_ID = "qwen38-b3-matched-b3-001"
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
PRIVATE_STATE_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-experience-writer-001"
TMPDIR = PRIVATE_STATE_ROOT + "/runtime-tmp"
MINIMUM_FREE_BYTES = 5 * 1024**3
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200
REQUIRED_NEW_SOURCES = frozenset({
    "scripts/native_experience_registration_001.py", "tests/test_native_experience_registration_001.py",
    "scripts/native_experience_sources_001.py", "tests/test_native_experience_sources_001.py",
    "scripts/native_experience_sources_005.py", "tests/test_native_experience_sources_005.py",
    "scripts/native_control_recovery_sources_006.py", "tests/test_native_control_recovery_sources_006.py",
    "scripts/native_appworld_controller_006.py", "tests/test_native_appworld_controller_006.py",
    "scripts/native_appworld_worker_client_006.py", "tests/test_native_appworld_worker_client_006.py",
    "scripts/native_experience_writer_001.py", "tests/test_native_experience_writer_001.py",
    "scripts/native_experience_gpu_queue_001.py", "tests/test_native_experience_gpu_queue_001.py",
    "scripts/run_native_experience_writer_001.py", "tests/test_run_native_experience_writer_001.py",
    "scripts/native_experience_content_001.py", "tests/test_native_experience_content_001.py",
    "scripts/native_experience_review_001.py", "tests/test_native_experience_review_001.py",
    "tests/test_native_experience_review_schedule_001.py",
})
_ERROR = "native_experience_registration_invalid"
_PLAN_FIELDS = frozenset({
    "id", "status", "source", "writer", "model_arm", "model_id", "expected_backend_static",
    "port", "service_plan", "service_plan_sha256", "python", "gpu_policy", "gpu_policy_sha256",
    "source_sha256", "input_sha256", "runtime_sha256", "tmp_dir", "private_run_root",
    "gpu_seconds", "total_gpu_seconds", "admission_wait_seconds", "automatic_retry",
    "writer_authorized", "training_authorized", "utility_authorized", "dev_or_test_authorized",
})
public_read = public.public_read
_public_hash = public._public_hash
_absolute_hash = public._absolute_hash
_private_directory = public._private_directory


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def writer_policy():
    return {"max_sources": 12, "max_calls_per_source": 1, "source_order": "ascending_source_index",
            "max_input_tokens": 15360, "max_output_tokens": 128, "temperature": 0.7,
            "seed_base": 17, "seed_rule": "17 + source_index", "request_window_seconds": 120,
            "per_call_seconds": 300, "model_ready_seconds": 600, "source_verification_seconds": 180,
            "content_review": {"kind": "fallible_local_model_review_pending_utility",
                "max_calls_per_candidate": 1, "max_input_tokens": 15360, "max_output_tokens": 256,
                "per_call_seconds": 180, "temperature": 0.7, "seed_base": 117,
                "seed_rule": "117 + source_index", "deterministic_screen_required": True,
                "accepted": False, "content_certified": False, "training_authorized": False},
            "release_seconds": 90, "shutdown_reserve_seconds": 150,
            "parent_tasks_max": 256, "parent_memory_bytes": 6 * 1024**3, "parent_cpu_quota_percent": 400}


def _source_plan(root):
    """Authenticate source006 registration without reading its private data."""
    raw = public_read(root, SOURCE_PLAN)
    _require(hashlib.sha256(raw).hexdigest() == SOURCE_PLAN_SHA)
    return source6.load_registration(root, SOURCE_PLAN_SHA, private=False)


def _qualification(root, expected_sha, previous):
    _require(public._sha(expected_sha))
    raw = public_read(root, QUALIFICATION_PROOF)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha)
    proof = public._json(raw)
    _require(type(proof) is dict and set(proof) == {
        "id", "status", "source_plan_sha256", "source_result_sha256", "private_input_manifest_sha256",
        "source_gate_kind", "gate", "writer_inputs"} and proof["id"] == QUALIFICATION_ID and proof["status"] == "passed"
        and proof["source_gate_kind"] == SOURCE_GATE_KIND
        and proof["source_plan_sha256"] == SOURCE_PLAN_SHA and public._sha(proof["source_result_sha256"])
        and public._sha(proof["private_input_manifest_sha256"]))
    gate = proof["gate"]
    _require(type(gate) is dict and set(gate) == {
        "status", "source_denominator", "complete_recovery_scenarios", "distinct_student_targets"}
        and gate["status"] == "passed" and type(gate["source_denominator"]) is int
        and gate["source_denominator"] == 12 and type(gate["complete_recovery_scenarios"]) is int
        and 4 <= gate["complete_recovery_scenarios"] <= 12 and type(gate["distinct_student_targets"]) is int
        and 20 <= gate["distinct_student_targets"] <= 24 * gate["complete_recovery_scenarios"])
    tasks, rows = previous["task_ids"], proof["writer_inputs"]
    _require(type(tasks) is list and len(tasks) == 12 and type(rows) is list
             and len(rows) == gate["complete_recovery_scenarios"])
    indices, scenarios = [], []
    for row in rows:
        _require(type(row) is dict and set(row) == {"source_index", "source_scenario", "messages_sha256"}
                 and type(row["source_index"]) is int and 0 <= row["source_index"] < 12
                 and type(row["source_scenario"]) is str and public._sha(row["messages_sha256"]))
        task = tasks[row["source_index"]]
        _require(type(task) is str and task.endswith("_2") and row["source_scenario"] == task[:-2])
        indices.append(row["source_index"])
        scenarios.append(row["source_scenario"])
    _require(indices == sorted(set(indices)) and len(scenarios) == len(set(scenarios)))
    return proof


def _private_assets():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    for path in (PRIVATE_STATE_ROOT, TMPDIR, PRIVATE_RUN_ROOT):
        _private_directory(path)
    _require(shutil.disk_usage(TMPDIR).free >= MINIMUM_FREE_BYTES
             and shutil.disk_usage(PRIVATE_RUN_ROOT).free >= MINIMUM_FREE_BYTES)


def _load_private_sources(root, plan_sha, result_sha):
    from scripts.native_control_recovery_sources_006 import load_qualified_sources

    return load_qualified_sources(root, plan_sha, result_sha)


def _private_qualification(root, proof, runtime):
    _private_assets()
    _require(all(_absolute_hash(name) == digest for name, digest in runtime.items()))
    loaded = _load_private_sources(root, SOURCE_PLAN_SHA, proof["source_result_sha256"])
    _require(type(loaded) is dict and set(loaded) == {
        "private", "source_gate_kind", "gate", "writer_inputs", "input_manifest", "input_manifest_sha256"}
        and loaded["private"] is True and loaded["source_gate_kind"] == SOURCE_GATE_KIND
        and public._same(loaded["gate"], proof["gate"])
        and loaded["input_manifest_sha256"] == proof["private_input_manifest_sha256"])
    manifest, inputs = loaded["input_manifest"], loaded["writer_inputs"]
    _require(type(manifest) is dict and set(manifest) == {
        "private", "source_plan_sha256", "source_result_sha256", "private_files_sha256",
        "public_context_binding", "writer_messages_sha256", "scope", "source_gate_kind",
        "primary_data_feasibility_gate", "primary_candidate_student_target_sha256", "ordinary_control_targets",
        "ordinary_control_target_sha256", "ordinary_targets_scope"}
        and manifest["private"] is True and manifest["source_plan_sha256"] == SOURCE_PLAN_SHA
        and manifest["source_gate_kind"] == SOURCE_GATE_KIND
        and manifest["source_result_sha256"] == proof["source_result_sha256"]
        and hashlib.sha256(_encode(manifest)).hexdigest() == proof["private_input_manifest_sha256"]
        and type(inputs) is list and len(inputs) == len(proof["writer_inputs"]))
    files = manifest["private_files_sha256"]
    _require(type(files) is dict and files.get(SOURCE_RESULT_PATH) == proof["source_result_sha256"]
             and public._sha(files.get(source6.PRIVATE_RUN_ROOT + "/candidate-student-targets.json")))
    for name, digest in files.items():
        _require(type(name) is str and Path(name).is_absolute() and os.path.normpath(name) == name
                 and Path(source6.PRIVATE_RUN_ROOT) in Path(name).parents and ".." not in Path(name).parts
                 and public._sha(digest))
    targets, hashes = manifest["ordinary_control_targets"], manifest["ordinary_control_target_sha256"]
    _require(type(targets) is list and len(targets) == proof["gate"]["distinct_student_targets"]
             and type(hashes) is list and hashes == sorted(set(hashes)) and len(hashes) == len(targets)
             and all(public._sha(digest) for digest in hashes))
    target_hashes = []
    scenarios = {row["source_scenario"] for row in proof["writer_inputs"]}
    for target in targets:
        _require(type(target) is dict and set(target) == {"task_id", "target_sha256", "action", "student_messages",
            "source_messages_sha256", "student_messages_sha256", "new_generation_index"}
            and type(target["task_id"]) is str and target["task_id"].endswith("_2")
            and target["task_id"][:-2] in scenarios and type(target["new_generation_index"]) is int
            and 0 <= target["new_generation_index"] < 24 and type(target["action"]) is str)
        _require(hashlib.sha256(_encode(_decode(target["action"].encode("utf-8")))).hexdigest() == target["target_sha256"]
                 and _message_payload(target["student_messages"])[1] == target["source_messages_sha256"]
                 == target["student_messages_sha256"])
        target_hashes.append(target["target_sha256"])
    _require(sorted(target_hashes) == hashes)
    projections, hashes = [], []
    for row in inputs:
        _require(type(row) is dict and set(row) == {"source_index", "source_scenario", "messages", "messages_sha256"}
                 and public._sha(row["messages_sha256"])
                 and _message_payload(row["messages"])[1] == row["messages_sha256"])
        projections.append({key: row[key] for key in ("source_index", "source_scenario", "messages_sha256")})
        hashes.append(row["messages_sha256"])
    _require(public._same(projections, proof["writer_inputs"])
             and public._same(manifest["writer_messages_sha256"], hashes))
    return loaded


def validate_registration(root, plan, *, private=False):
    """Validate writer metadata; private=True additionally recomputes the source gate."""
    try:
        _require(type(private) is bool and type(plan) is dict and set(plan) == _PLAN_FIELDS)
        public._values(plan)
        previous = _source_plan(root)
        _require(previous["id"] == "qwen38-native-source-pilot-006" and previous["status"] == "registered"
                 and all(previous[key] is False for key in (
                     "writer_authorized", "training_authorized", "utility_authorized", "dev_or_test_authorized"))
                 and previous["model_arm"] == MODEL_ARM and previous["model_id"] == MODEL_ID
                 and type(previous["port"]) is int and previous["port"] == PORT
                 and previous["service_plan"] == SERVICE_PLAN and previous["service_plan_sha256"] == SERVICE_SHA
                 and previous["python"] == PYTHON and previous["tmp_dir"] == TMPDIR)
        _require(plan["id"] == ID and plan["status"] == "registered"
                 and plan["model_arm"] == MODEL_ARM and plan["model_id"] == MODEL_ID
                 and type(plan["port"]) is int and plan["port"] == PORT
                 and plan["service_plan"] == SERVICE_PLAN and plan["service_plan_sha256"] == SERVICE_SHA
                 and plan["python"] == PYTHON and plan["tmp_dir"] == TMPDIR
                 and plan["private_run_root"] == PRIVATE_RUN_ROOT
                 and type(plan["gpu_seconds"]) is int and plan["gpu_seconds"] == BATCH_GPU_SECONDS
                 and type(plan["total_gpu_seconds"]) is int and plan["total_gpu_seconds"] == BATCH_GPU_SECONDS
                 and type(plan["admission_wait_seconds"]) is int and plan["admission_wait_seconds"] == WAIT_SECONDS
                 and plan["automatic_retry"] is False and plan["writer_authorized"] is True
                 and all(plan[key] is False for key in (
                     "training_authorized", "utility_authorized", "dev_or_test_authorized"))
                 and public._same(plan["writer"], writer_policy())
                 and public._same(plan["gpu_policy"], previous["gpu_policy"])
                 and plan["gpu_policy_sha256"] == previous["gpu_policy_sha256"]
                 and public._same(plan["expected_backend_static"], previous["expected_backend_static"]))
        sources, inputs, runtime = (plan[key] for key in ("source_sha256", "input_sha256", "runtime_sha256"))
        public._binding_map(sources)
        public._binding_map(inputs)
        public._binding_map(runtime, runtime=True)
        for field, current in (("source_sha256", sources), ("input_sha256", inputs)):
            _require(type(previous[field]) is dict
                     and all(current.get(name) == digest for name, digest in previous[field].items()))
        _require(public._same(runtime, previous["runtime_sha256"]) and REQUIRED_NEW_SOURCES <= sources.keys()
                 and inputs.get(SOURCE_PLAN) == SOURCE_PLAN_SHA and public._sha(inputs.get(QUALIFICATION_PROOF)))
        for name, digest in {**sources, **inputs}.items():
            _require(name not in sources or name not in inputs or sources[name] == inputs[name])
            _require(_public_hash(root, name) == digest)
        proof = _qualification(root, inputs[QUALIFICATION_PROOF], previous)
        expected_source = {"plan": SOURCE_PLAN, "plan_sha256": SOURCE_PLAN_SHA,
            "source_gate_kind": SOURCE_GATE_KIND,
            "result_sha256": proof["source_result_sha256"],
            "private_input_manifest_sha256": proof["private_input_manifest_sha256"],
            "qualification_proof": {"path": QUALIFICATION_PROOF, "sha256": inputs[QUALIFICATION_PROOF]},
            "writer_inputs": proof["writer_inputs"]}
        _require(public._same(plan["source"], expected_source))
        if private:
            _private_qualification(root, proof, runtime)
        return copy.deepcopy(plan)
    except Exception:
        raise ValueError(_ERROR) from None


def build_registration(root, *, source_sha256, input_sha256, runtime_sha256,
                       qualification_proof_sha256, private=False):
    """Build and validate only; caller owns qualification preparation and writing."""
    try:
        previous = _source_plan(root)
        proof = _qualification(root, qualification_proof_sha256, previous)
        plan = {"id": ID, "status": "registered", "source": {
                    "plan": SOURCE_PLAN, "plan_sha256": SOURCE_PLAN_SHA,
                    "source_gate_kind": SOURCE_GATE_KIND,
                    "result_sha256": proof["source_result_sha256"],
                    "private_input_manifest_sha256": proof["private_input_manifest_sha256"],
                    "qualification_proof": {"path": QUALIFICATION_PROOF, "sha256": qualification_proof_sha256},
                    "writer_inputs": copy.deepcopy(proof["writer_inputs"])},
                "writer": writer_policy(), "model_arm": MODEL_ARM, "model_id": MODEL_ID,
                "expected_backend_static": copy.deepcopy(previous["expected_backend_static"]),
                "port": PORT, "service_plan": SERVICE_PLAN, "service_plan_sha256": SERVICE_SHA,
                "python": PYTHON, "gpu_policy": copy.deepcopy(previous["gpu_policy"]),
                "gpu_policy_sha256": previous["gpu_policy_sha256"],
                "source_sha256": copy.deepcopy(source_sha256), "input_sha256": copy.deepcopy(input_sha256),
                "runtime_sha256": copy.deepcopy(runtime_sha256), "tmp_dir": TMPDIR,
                "private_run_root": PRIVATE_RUN_ROOT, "gpu_seconds": BATCH_GPU_SECONDS,
                "total_gpu_seconds": BATCH_GPU_SECONDS, "admission_wait_seconds": WAIT_SECONDS,
                "automatic_retry": False, "writer_authorized": True, "training_authorized": False,
                "utility_authorized": False, "dev_or_test_authorized": False}
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
    """Return one fully verified plan and private bundle after one semantic load.

    The bundle contains protected writer messages and must remain private. The
    public plan and pins are rechecked after the private recomputation; future
    lightweight freshness checks remain the coordinator's responsibility.
    """
    try:
        plan = load_registration(root, sha, private=False)
        previous = _source_plan(root)
        proof = _qualification(root, plan["source"]["qualification_proof"]["sha256"], previous)
        bundle = _private_qualification(root, proof, plan["runtime_sha256"])
        _require(public._same(load_registration(root, sha, private=False), plan))
        return plan, bundle
    except Exception:
        raise ValueError(_ERROR) from None
