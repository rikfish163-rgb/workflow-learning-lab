"""Prospective singleton state-recovery cohort registration and validation."""

import hashlib
import os
import shutil
import socket
from pathlib import Path

from scripts import native_source_registration as public
from scripts import native_state_recovery_episode_001 as episode
from scripts import serve_qwen38_b3_runtime_relocation_001 as relocated
from scripts.native_appworld_worker_client import WORKSPACE
from scripts.native_source_registration import (  # Re-export for the bound episode and pilot runtime.
    MODEL_ARM,
    MODEL_ID,
    PYTHON,
    SERVICE_PLAN,
    SERVICE_SHA,
    TMPDIR,
    _public_hash,
    _same,
    public_read,
    static_backend,
)

ID, PLAN, TASK_IDS = episode.ID, episode.PLAN, episode.TASK_IDS
PRIVATE_RUN_ROOT = public.PRIVATE_STATE_ROOT + "/native-state-recovery-source-001"
PORT = 18999
BATCH_TASK_IDS = tuple((task,) for task in TASK_IDS)
BATCH_GPU_SECONDS = 7200
WAIT_SECONDS = 86400
REFERENCE = "artifacts/qwen38-native-source-pilot-plan-006.json"
CONTEXT = "artifacts/native-state-recovery-source-context-CPU-001.json"
WORKER_PROOF = "artifacts/native-state-refusal-runtime-relocated-root-CPU-001.json"
FIXED_INPUTS = {
    REFERENCE: "da4b4c4a4c8040a961ef0604ed19152bc5a3ea21790618afd19e694651c63ad3",
    CONTEXT: "d61fe9697f3b3cf62bb5bc1b30e1eaf076754b63c9e8dc9c701d91bbae6bdfb4",
    WORKER_PROOF: "7432443f6a36bd99bfe4dee48597483eb24b00e37dbe7b5cbf62557567aeaa66",
    relocated.RELOCATION: "be1a7cab95cd9f1e0e6e1db7f8b719b3bbda64ca411378abaaee72eddc79875e",
    episode.DESIGN: episode.DESIGN_SHA256,
}
NEW_SOURCES = set(episode.SELF_SOURCES) | {
    "scripts/native_state_recovery_registration_001.py", "scripts/native_state_recovery_gpu_queue_001.py",
    "scripts/run_native_state_recovery_source_001.py", "scripts/serve_qwen38_b3_runtime_relocation_001.py",
    "tests/test_native_state_recovery_source_001.py",
}


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_registration_invalid")


def _binding(path):
    return {"path": path, "sha256": FIXED_INPUTS[path]}


def build(root):
    """Build the only accepted plan from fixed, pre-outcome CPU evidence."""
    docs = {}
    for name, expected in FIXED_INPUTS.items():
        raw = public_read(root, name)
        _require(hashlib.sha256(raw).hexdigest() == expected)
        docs[name] = public._json(raw)
    reference, context, worker = (docs[name] for name in (REFERENCE, CONTEXT, WORKER_PROOF))
    _require(context["status"] == "measured_CPU_initial_contexts" and context["task_ids"] == list(TASK_IDS)
             and [row["task_id"] for row in context["rows"]] == list(TASK_IDS)
             and all(row["initial_fits"] is True and type(row["input_tokens"]) is int
                     and 1 <= row["input_tokens"] <= 15360 and public._sha(row["metadata_sha256"])
                     and public._sha(row["messages_sha256"]) for row in context["rows"])
             and context["official_train_membership_verified"] is True and context["utility_scenarios_disjoint"] is True
             and worker["status"] == "passed" and public._sha(worker["snapshot_manifest_sha256"])
             and all(type(doc[key]) is int and doc[key] == 0 for doc in (context, worker)
                     for key in ("GPU_jobs", "model_calls")))
    sources = {**reference["source_sha256"], **context["source_sha256"],
               **{name: _public_hash(root, name) for name in NEW_SOURCES}}
    inputs = {**reference["input_sha256"], **FIXED_INPUTS}
    for bindings in (sources, inputs):
        _require(all(_public_hash(root, name) == sha for name, sha in bindings.items()))
    _require(all(sources.get(name) == sha for name, sha in episode.SOURCE_PINS.items())
             and all(sources.get(name) == sha for name, sha in relocated.SOURCE_PINS.items()))
    migration = docs[relocated.RELOCATION]
    runtime = dict(reference["runtime_sha256"])
    _require(runtime.pop("/usr/bin/python3.12") == migration["interpreter_sha256"]
             and runtime[str(relocated.CONFIG)] == migration["historical_config_sha256"])
    runtime.update({str(relocated.CONFIG): migration["effective_config_sha256"],
                    str(relocated.OLD_CONFIG): migration["historical_config_sha256"],
                    str(relocated.INTERPRETER): migration["interpreter_sha256"],
                    str(relocated.FALLBACK): migration["fallback_pth_sha256"]})
    backend = dict(reference["expected_backend_static"][MODEL_ARM],
                   runtime_relocation_sha256=FIXED_INPUTS[relocated.RELOCATION])
    return {"id": ID, "status": "registered", "design": docs[episode.DESIGN], "task_ids": list(TASK_IDS),
            "batch_task_ids": [list(batch) for batch in BATCH_TASK_IDS], "automatic_retry": False,
            "writer_authorized": False, "training_authorized": False, "utility_authorized": False,
            "dev_or_test_authorized": False, "model_arm": MODEL_ARM, "model_id": MODEL_ID,
            "service_plan": SERVICE_PLAN, "service_plan_sha256": SERVICE_SHA, "port": PORT, "python": PYTHON,
            "expected_backend_static": {MODEL_ARM: backend}, "private_run_root": PRIVATE_RUN_ROOT, "tmp_dir": TMPDIR,
            "admission_wait_seconds": WAIT_SECONDS, "gpu_seconds": BATCH_GPU_SECONDS,
            "total_gpu_seconds": BATCH_GPU_SECONDS * len(TASK_IDS), "gpu_policy": reference["gpu_policy"],
            "gpu_policy_sha256": reference["gpu_policy_sha256"], "asset_manifest": public.ASSET_MANIFEST,
            "asset_manifest_sha256": public.ASSET_MANIFEST_SHA256, "source_sha256": sources,
            "input_sha256": inputs, "runtime_sha256": runtime, "context_proof": _binding(CONTEXT),
            "worker_CPU_proof": _binding(WORKER_PROOF), "runtime_relocation": _binding(relocated.RELOCATION)}


def load_registration(root, sha, *, private=False):
    _require(public._sha(sha) and type(private) is bool)
    raw = public_read(root, PLAN)
    _require(hashlib.sha256(raw).hexdigest() == sha)
    plan = public._json(raw)
    _require(_same(plan, build(root)) and public_read(root, PLAN) == raw)
    if private:
        _require(Path(root) == WORKSPACE and socket.gethostname() == "amax"
                 and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("TMPDIR") == TMPDIR)
        public._private_directory(PRIVATE_RUN_ROOT)
        _require(shutil.disk_usage(PRIVATE_RUN_ROOT).free >= public.MINIMUM_FREE_BYTES)
        public._private_assets()
        _require(all(public._absolute_hash(path) == expected for path, expected in plan["runtime_sha256"].items())
                 and _same(relocated.runtime_binding(), public._read(root, relocated.RELOCATION)))
    return plan


__all__ = ["_public_hash", "_same", "public_read", "static_backend"]
