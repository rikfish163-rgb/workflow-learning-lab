"""Bind the ten registered-source inputs using only metadata and CPU tokenization."""

import hashlib
import os
from pathlib import Path

from scripts import native_appworld_actor_004 as actor
from scripts import native_source_registration as public
from scripts import native_state_recovery_episode_001 as episode
from scripts import native_verification_token_counter as counter
from scripts import prepare_native_source_contexts as base
from scripts.native_appworld_worker_client import _decode, _encode
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
ID = "native-state-recovery-source-context-CPU-001"
PROOF = ROOT / "artifacts" / (ID + ".json")
PREPARER = "scripts/prepare_native_state_recovery_contexts_001.py"
SOURCES = base.PROOF_SOURCES | episode.SOURCE_PINS.keys() | {PREPARER, "scripts/native_state_recovery_episode_001.py"}
TRAIN = Path(public.ASSET_ROOT) / "data/datasets/train.txt"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare():
    design_raw = public.public_read(ROOT, episode.DESIGN)
    base._require(digest(design_raw) == episode.DESIGN_SHA256)
    design = _decode(design_raw)
    tasks = list(episode.TASK_IDS)
    roles, _ = public._cohort(ROOT)
    train_raw = TRAIN.read_bytes()
    train = train_raw.decode("ascii").splitlines()
    base._require(len(train) == len(set(train)) == 90 and set(train) == set(sum(roles.values(), [])))
    base._require(tasks == design["source_task_ids"] and len(tasks) == len(set(tasks)) == 10
                  and all(task.endswith("_1") and task in roles["origin"] for task in tasks)
                  and {task[:-2] for task in tasks}.isdisjoint(design["utility_scenarios"])
                  and set(design["utility_scenarios"]) == {task[:-2] for task in roles["utility"]})
    sources = {name: digest(public.public_read(ROOT, name)) for name in sorted(SOURCES)}
    base._require(all(sources[name] == pin for name, pin in episode.SOURCE_PINS.items()))
    runtime = {"config.json": public.MODEL_CONFIG_SHA256, **public.TOKENIZER_SHA256}
    base._require(base._cuda_uninitialized() and base._runtime_hashes() == runtime)
    progress = {"ground_truth_load_attempts": 0}
    rows = []
    with base._native_task_classes() as (task_type, ground_truth):
        with base._forbid_ground_truth(ground_truth, progress):
            for task in tasks:
                metadata = base._task_metadata(task_type.load(task_id=task, load_ground_truth=False), task)
                metadata_raw = _encode(metadata)
                messages = actor.build_actor_messages(metadata, prefix_records=None, review_hint=None)
                _, messages_sha = _message_payload(messages)
                count = counter.count_input_tokens(messages)
                base._require(type(count) is int and 1 <= count <= base.MAX_INPUT_TOKENS
                              and _encode(metadata) == metadata_raw and _message_payload(messages)[1] == messages_sha)
                rows.append({"task_id": task, "metadata_sha256": digest(metadata_raw), "messages_sha256": messages_sha,
                             "input_tokens": count, "initial_fits": True})
    base._require(progress["ground_truth_load_attempts"] == 0 and base._cuda_uninitialized()
                  and base._runtime_hashes() == runtime and TRAIN.read_bytes() == train_raw
                  and all(digest(public.public_read(ROOT, name)) == sha for name, sha in sources.items()))
    return {"id": ID, "status": "measured_CPU_initial_contexts", "task_ids": tasks, "rows": rows,
            "design_sha256": episode.DESIGN_SHA256, "official_train_ids_sha256": digest(train_raw),
            "official_train_membership_verified": True, "utility_scenarios_disjoint": True,
            "source_sha256": sources, "preparation": {"path": PREPARER, "sha256": sources[PREPARER]},
            "model_path": public.MODEL_PATH, "model_config_sha256": public.MODEL_CONFIG_SHA256,
            "tokenizer_files_sha256": public.TOKENIZER_SHA256, "CUDA_VISIBLE_DEVICES": "", "cuda_initialized": False,
            "model_weights_loaded": False, "GPU_jobs": 0, "model_calls": 0, "task_worlds_started": 0,
            "task_metadata_loads": len(rows), "ground_truth_load_attempts": 0}


def main():
    base._host_preflight()
    os.umask(0o077)
    private = Path(public.PRIVATE_STATE_ROOT) / (ID + "-runtime-restored-003")
    private.mkdir(mode=0o700)
    fd = public._directory(private)
    try:
        with base._exclusive_file(fd, "diagnostics.log") as diagnostics:
            os.dup2(diagnostics.fileno(), 1)
            os.dup2(diagnostics.fileno(), 2)
            raw = _encode(prepare()) + b"\n"
            with base._exclusive_file(fd, "context-proof.json") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(fd)
            artifact_fd = public._directory(PROOF.parent)
            try:
                os.link("context-proof.json", PROOF.name, src_dir_fd=fd, dst_dir_fd=artifact_fd, follow_symlinks=False)
                os.unlink("context-proof.json", dir_fd=fd)
                os.fsync(fd)
                os.fsync(artifact_fd)
            finally:
                os.close(artifact_fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
