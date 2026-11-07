"""One private, registered native JSON source-pilot episode; never a scorer.

The owning launcher supplies a hashed private configuration and an already
admitted B3 service. This entrypoint cannot enqueue or retry a GPU job, choose
a task or hint dynamically, load ground truth, or change the fixed budgets.
"""

import argparse
import hashlib
import math
import os
import re
import socket
import sys
import time
from pathlib import Path

from scripts import native_appworld_actor as actor
from scripts import native_source_gpu_queue as queue
from scripts import native_source_registration as registration
from scripts.native_appworld_controller import run_task
from scripts.native_appworld_model_client import NativeModel
from scripts.native_appworld_replay import _inputs, _metadata
from scripts.native_appworld_worker_client import (
    WORKSPACE,
    NativeWorkerClient,
    _decode,
    _encode,
    _exclusive,
    _open_directory,
    _private,
    _same,
)
from scripts.verification_token_counter import _message_payload
from wll.protocol import fingerprint

PRIVATE_RUN_ROOT = Path(registration.PRIVATE_RUN_ROOT)
REMOTE = queue.REMOTE
_ERROR = "native_source_episode_invalid"
_CONFIG_KEYS = {"plan_sha256", "source_index", "phase", "run_dir", "job", "expected_backend",
                "prefix_binding", "initial_context"}
_CONTEXT_KEYS = {"metadata_sha256", "messages_sha256", "input_tokens"}
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_SOURCE_BUDGET = {"max_generations": 60, "max_action_attempts": 60,
                  "max_API_invocations": 60, "max_prompt_tokens": 921600,
                  "max_completion_tokens": 61440}
_SUFFIX_BUDGET = {"max_generations": 6, "max_action_attempts_including_replay": 60,
                  "max_API_invocations_including_replay": 60, "max_prompt_tokens": 92160,
                  "max_completion_tokens": 6144}


class NativeSourceEpisodeError(RuntimeError):
    """Content-free failure; raw execution diagnostics belong to private files."""


def _require(value):
    if not value:
        raise NativeSourceEpisodeError(_ERROR)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _absolute(value):
    raw = os.fspath(value)
    path = Path(raw)
    _require(path.is_absolute() and raw == str(path) and ".." not in path.parts)
    return path


def _private_directory(path):
    path = _absolute(path)
    base = _absolute(PRIVATE_RUN_ROOT)
    _require(path == base or base in path.parents)
    descriptor = _open_directory(base)
    try:
        _private(descriptor, directory=True)
        for component in path.relative_to(base).parts:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _private(descriptor, directory=True)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_private(path, *, maximum):
    path = _absolute(path)
    _require(PRIVATE_RUN_ROOT in path.parents)
    directory = _private_directory(path.parent)
    descriptor = None
    try:
        descriptor = os.open(path.name, _FILE_FLAGS, dir_fd=directory)
        before = _private(descriptor, directory=False)
        _require(0 <= before.st_size <= maximum)
        chunks, size = [], 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not block:
                break
            size += len(block)
            _require(size <= maximum)
            chunks.append(block)
        _require(size == before.st_size and _stamp(before) == _stamp(os.fstat(descriptor))
                 == _stamp(os.stat(path.name, dir_fd=directory, follow_symlinks=False)))
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _context_row(root, plan, index):
    binding = plan["context_proof"]
    _require(type(binding) is dict and set(binding) == {"path", "sha256"}
             and _sha(binding["sha256"])
             and plan["input_sha256"].get(binding["path"]) == binding["sha256"])
    raw = registration.public_read(root, binding["path"])
    _require(_hash(raw) == binding["sha256"])
    proof = _decode(raw)
    _require(type(proof) is dict and type(proof.get("rows")) is list
             and len(proof["rows"]) == len(registration.TASK_IDS))
    row = proof["rows"][index]
    _require(type(row) is dict and set(row) == {"task_id", "initial_fits", *_CONTEXT_KEYS}
             and row["task_id"] == registration.TASK_IDS[index] and row["initial_fits"] is True
             and _sha(row["metadata_sha256"]) and _sha(row["messages_sha256"])
             and type(row["input_tokens"]) is int and 1 <= row["input_tokens"] <= 15360)
    return row


def _fixed_design(plan, phase):
    design = plan["design"]
    budgets, sampling = design["budgets"], design["sampling"]
    _require(_same(budgets["source"], _SOURCE_BUDGET)
             and _same(budgets["suffix"], _SUFFIX_BUDGET)
             and all(type(budgets.get(key)) is int and budgets[key] == value for key, value in {
                 "max_input_tokens": 15360, "max_new_tokens": 1024,
                 "actor_generation_seconds": 600, "setup_seconds": 120, "cleanup_seconds": 60,
                 "worker_wall_seconds": 780, "parent_unit_seconds": 900}.items())
             and budgets["prefix_replay_charged_to_setup"] is True
             and type(design["schedule"]["generation_seed_base"]) is int
             and design["schedule"]["generation_seed_base"] == 17
             and type(sampling["context_tokens"]) is int and sampling["context_tokens"] == 16384
             and type(sampling["temperature"]) is float and sampling["temperature"] == 0.7
             and type(design["review"]["hint"]) is str and bool(design["review"]["hint"])
             and type(design["boundary"]["prefix_min"]) is int
             and design["boundary"]["prefix_min"] == 1
             and type(design["boundary"]["prefix_max"]) is int
             and design["boundary"]["prefix_max"] == 54)
    branch = budgets["source" if phase == "first_attempt" else "suffix"]
    return actor.ActorLimits(
        max_generations=branch["max_generations"], max_actions=60, max_api_requests=60,
        max_input_tokens=budgets["max_input_tokens"],
        max_prompt_tokens_total=branch["max_prompt_tokens"],
        max_completion_tokens_total=branch["max_completion_tokens"],
        max_new_tokens=budgets["max_new_tokens"], wall_seconds=budgets["actor_generation_seconds"])


def _backend(plan, backend):
    _require(type(backend) is dict and set(plan["expected_backend_static"]) == {registration.MODEL_ARM}
             and _same(registration.static_backend(backend),
                       plan["expected_backend_static"][registration.MODEL_ARM]))
    static_keys = set(plan["expected_backend_static"][registration.MODEL_ARM])
    _require(set(backend) == static_keys | {"gpu_policy", "loading_peak_reserved_bytes"})
    policy = backend["gpu_policy"]
    _require(type(policy) is dict and set(policy) == {"hostname", "physical_gpu_index", "gpu_uuid",
                                                     "visible_device_count", "policy_sha256"})
    index = policy["physical_gpu_index"]
    _require(type(index) is int and index in (8, 9)
             and policy["hostname"] == plan["gpu_policy"]["hostname"] == "amax"
             and policy["gpu_uuid"] == plan["gpu_policy"]["allowed_gpus"][str(index)]
             and policy["gpu_uuid"] == {
                 8: "GPU-ecc8cd45-fdaa-b5ee-e54a-bf2ef7df8f82",
                 9: "GPU-4fc78635-e0b9-1acf-774f-9debcfc7fa47"}[index]
             and type(policy["visible_device_count"]) is int and policy["visible_device_count"] == 1
             and _sha(policy["policy_sha256"])
             and policy["policy_sha256"] == plan["gpu_policy_sha256"]
             and type(backend["loading_peak_reserved_bytes"]) is int
             and 0 < backend["loading_peak_reserved_bytes"] <= 20 * 1024**3)


def _prefix(config, row, limits, hint):
    binding = config["prefix_binding"]
    if config["phase"] == "first_attempt":
        _require(binding is None)
        return None, None
    _require(type(binding) is dict and set(binding) == {"path", "sha256"} and _sha(binding["sha256"]))
    raw = _read_private(binding["path"], maximum=32_000_000)
    _require(_hash(raw) == binding["sha256"])
    prefix = _decode(raw)
    _require(type(prefix) is dict and set(prefix) == {"metadata", "prefix_records"})
    metadata, records = _inputs(prefix["metadata"], prefix["prefix_records"], lambda _: None,
                                 time.monotonic() + 120, limits.max_actions,
                                 limits.max_api_requests, time.monotonic)
    _require(_hash(_encode(metadata)) == row["metadata_sha256"] and 1 <= len(records) <= 54)
    for index, item in enumerate(records):
        observation = item["observation"]
        confirmed_error = observation.get("error_type") == "ObservedAPIError"
        _require(confirmed_error == (index == len(records) - 1)
                 and not (observation["ok"] and actor._is_completion(item["action"])))
    messages = actor.build_actor_messages(metadata, prefix_records=records, review_hint=hint)
    _require(_message_payload(messages)[1] == config["initial_context"]["messages_sha256"])
    return metadata, records


def _configuration(config, root, loader):
    _require(type(config) is dict and set(config) == _CONFIG_KEYS and _sha(config["plan_sha256"])
             and type(config["source_index"]) is int and 0 <= config["source_index"] < 12
             and type(config["phase"]) is str and config["phase"] in {"first_attempt", "control", "review"})
    index, phase = config["source_index"], config["phase"]
    plan = loader(root, config["plan_sha256"], private=False)
    _require(type(plan) is dict and plan["id"] == registration.ID and plan["status"] == "registered"
             and _same(plan["task_ids"], list(registration.TASK_IDS))
             and plan["model_arm"] == registration.MODEL_ARM and plan["model_id"] == registration.MODEL_ID
             and type(plan["port"]) is int and plan["port"] == registration.PORT
             and plan["private_run_root"] == str(PRIVATE_RUN_ROOT))
    limits = _fixed_design(plan, phase)
    job = config["job"]
    _require(type(job) is dict
             and _same(job, queue.make_job(index // 4, job.get("owner"), config["plan_sha256"])))
    path = _absolute(config["run_dir"])
    _require(path.parent == PRIVATE_RUN_ROOT
             and path.name == f"unit-{index:02d}-{phase}-{job['owner']}")
    _backend(plan, config["expected_backend"])
    context = config["initial_context"]
    _require(type(context) is dict and set(context) == _CONTEXT_KEYS
             and _sha(context["metadata_sha256"]) and _sha(context["messages_sha256"])
             and type(context["input_tokens"]) is int and 1 <= context["input_tokens"] <= 15360)
    row = _context_row(root, plan, index)
    _require(context["metadata_sha256"] == row["metadata_sha256"])
    if phase == "first_attempt":
        _require(all(_same(context[key], row[key]) for key in _CONTEXT_KEYS))
    hint = plan["design"]["review"]["hint"] if phase == "review" else None
    metadata, records = _prefix(config, row, limits, hint)
    return plan, limits, path, metadata, records, hint


def _queue_binding(config, status_function):
    job, backend = config["job"], config["expected_backend"]
    state = status_function("status", job, root=REMOTE)
    _require(type(state) is dict and state.get("owned") is True
             and state.get("admission_cancelled") is False and state.get("released") is False
             and state.get("monitor_live") is True and _finite(state.get("epoch"))
             and 0 <= time.time() - state["epoch"] < 5)
    queue_raw = registration.public_read(REMOTE, "artifacts/gpu-idle-queue.json")
    _require(_same(_decode(queue_raw), job))
    info, processes = state.get("job"), state.get("model_processes")
    _require(type(info) is dict and info.get("job_id") == job["id"] and info.get("status") == "running"
             and info.get("queue_sha256") == _hash(queue_raw)
             and type(info.get("process_group")) is int and info["process_group"] > 0
             and type(info.get("start_ticks")) is str and info["start_ticks"].isdigit()
             and type(info.get("gpu_index")) is int
             and info["gpu_index"] == backend["gpu_policy"]["physical_gpu_index"]
             and _finite(info.get("started_epoch")) and info["started_epoch"] > 0
             and type(processes) is list and len(processes) == 1)
    process = processes[0]
    _require(type(process) is dict and type(process.get("pid")) is int and process["pid"] > 0
             and type(process.get("uid")) is int and process["uid"] == os.geteuid()
             and type(process.get("start_ticks")) is str and process["start_ticks"].isdigit()
             and process.get("process_group") == info["process_group"]
             and process.get("cwd") == str(REMOTE)
             and process.get("cuda_visible_devices") == backend["gpu_policy"]["gpu_uuid"]
             and _same(state.get("model_pids"), [process["pid"]]))
    ready_path = queue.ready_path(REMOTE / "artifacts", job)
    ready = _decode(registration.public_read(REMOTE, str(ready_path.relative_to(REMOTE))))
    _require(type(ready) is dict and set(ready) == {"job_sha256", "ready_epoch", "ready_monotonic",
                                                  "backend_sha256", "pid"}
             and ready["job_sha256"] == fingerprint(job)
             and ready["backend_sha256"] == fingerprint(backend)
             and type(ready["pid"]) is int and ready["pid"] == process["pid"]
             and _finite(ready["ready_epoch"]) and info["started_epoch"] <= ready["ready_epoch"] <= time.time()
             and _finite(ready["ready_monotonic"]) and 0 < ready["ready_monotonic"] <= time.monotonic())
    identity = (process["pid"], process["start_ticks"], process["process_group"], info["start_ticks"],
                fingerprint(ready))
    return state, ready, identity


def _write(file, value):
    view = memoryview(_encode(value) + b"\n")
    while view:
        count = file.write(view)
        _require(type(count) is int and count > 0)
        view = view[count:]
    file.flush()
    os.fsync(file.fileno())


class _MetadataGuard:
    def __init__(self, worker, expected_context, state):
        self._worker, self._context, self._state = worker, expected_context, state

    def request(self, request, **kwargs):
        reply = self._worker.request(request, **kwargs)
        if request == {"op": "metadata"}:
            _require(type(reply) is dict and reply.get("ok") is True)
            metadata = reply.get("result")
            _metadata(metadata)
            _require(_hash(_encode(metadata)) == self._context["metadata_sha256"])
            self._state["metadata_verified"] = True
        return reply

    def __getattr__(self, name):
        return getattr(self._worker, name)


class _ModelGuard:
    def __init__(self, delegate, config, state, check_queue, record):
        self._delegate, self._config, self._state = delegate, config, state
        self._check_queue, self._record = check_queue, record
        self._initial_count_verified = self._health_verified = False
        self.max_tokens = delegate.max_tokens

    @property
    def deadline(self):
        return self._delegate.deadline

    @deadline.setter
    def deadline(self, value):
        self._delegate.deadline = value

    def check_health(self):
        self._health_verified = False
        self._check_queue()
        reply = self._delegate.check_health()
        _require(type(reply) is dict and reply.get("model") == registration.MODEL_ID
                 and reply.get("ready") is True and reply.get("busy") is False
                 and type(reply.get("max_context")) is int and reply["max_context"] == 16384
                 and type(reply.get("max_output_tokens")) is int and reply["max_output_tokens"] == 1024
                 and _same(reply.get("backend"), self._config["expected_backend"]))
        self._health_verified = True
        return reply

    def count_input_tokens(self, messages):
        _require(self._state["metadata_verified"])
        digest = _message_payload(messages)[1]
        if not self._initial_count_verified:
            _require(digest == self._config["initial_context"]["messages_sha256"])
        count = self._delegate.count_input_tokens(messages)
        if not self._initial_count_verified:
            _require(type(count) is int and count == self._config["initial_context"]["input_tokens"])
            self._record({"event": "initial_context_verified", "messages_sha256": digest,
                          "input_tokens": count})
            self._initial_count_verified = True
        return count

    def generate(self, messages, seed):
        _require(self._state["metadata_verified"] and self._health_verified
                 and self._initial_count_verified)
        self._check_queue()
        return self._delegate.generate(messages, seed=seed)


def run_episode(config_path, expected_config_sha256, *, root=WORKSPACE,
                _load_plan=None, _model_factory=None, _run_task=None, _queue_status=None):
    """Run exactly once using registered tasks, limits, service, and private inputs."""
    run_fd, journal = None, None
    try:
        _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                 and _sha(expected_config_sha256))
        raw = _read_private(config_path, maximum=4_000_000)
        _require(_hash(raw) == expected_config_sha256)
        config = _decode(raw)
        plan, limits, path, metadata, records, hint = _configuration(
            config, root, _load_plan or registration.load_registration)
        status_function = _queue_status or queue.queue_operation
        initial_state, initial_ready, bound_identity = _queue_binding(config, status_function)
        parent_fd = _private_directory(PRIVATE_RUN_ROOT)
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            run_fd = os.open(path.name, _DIR_FLAGS, dir_fd=parent_fd)
            _private(run_fd, directory=True)
        finally:
            os.close(parent_fd)
        journal = _exclusive(run_fd, "episode.jsonl")

        def record(event):
            _write(journal, event)

        record({"event": "registered_episode_intent", "config_sha256": expected_config_sha256,
                "config": config, "new_GPU_jobs_started": 0, "official_scores_computed": 0})
        record({"event": "queue_binding", "state": initial_state, "ready": initial_ready})

        def check_queue():
            state, ready, identity = _queue_binding(config, status_function)
            _require(identity == bound_identity)
            record({"event": "queue_binding", "state": state, "ready": ready})

        model = (_model_factory or NativeModel)(
            f"http://127.0.0.1:{registration.PORT}/v1", registration.MODEL_ID,
            expected_backend=config["expected_backend"], max_tokens=1024, context_tokens=16384,
            temperature=0.7, timeout=600, deadline=time.monotonic() + 780)
        state = {"metadata_verified": False}
        guarded_model = _ModelGuard(model, config, state, check_queue, record)
        task_id = registration.TASK_IDS[config["source_index"]]
        experiment = f"native-source-001-{config['source_index']:02d}-{config['phase']}-{config['job']['owner']}"

        def worker_factory(worker_task, worker_experiment, worker_path, **kwargs):
            _require(worker_task == task_id and worker_experiment == experiment
                     and Path(worker_path) == path / "worker"
                     and _same(kwargs, {"max_actions": 60, "max_api_requests": 60,
                                        "rpc_timeout": 60, "wall_seconds": 780}))
            return _MetadataGuard(NativeWorkerClient(worker_task, worker_experiment, worker_path, **kwargs),
                                  config["initial_context"], state)

        receipt = (_run_task or run_task)(
            task_id, experiment, path, limits=limits,
            seed=17 + (len(records) if records is not None else 0), model=guarded_model,
            token_counter=guarded_model.count_input_tokens, rpc_timeout=60, worker_wall_seconds=780,
            setup_seconds=120, cleanup_seconds=60, is_policy_episode=True,
            source_metadata=metadata, prefix_records=records, review_hint=hint,
            _worker_factory=worker_factory)
        _require(type(receipt) is dict and receipt.get("task_id") == task_id
                 and receipt.get("status") in {"saved_and_closed", "unknown", "observation_mismatch"})
        if receipt["status"] == "saved_and_closed":
            _require(state["metadata_verified"] and guarded_model._health_verified)
        record({"event": "registered_episode_stop", "receipt": receipt})
        os.fsync(run_fd)
        return receipt
    except Exception as error:
        if run_fd is not None:
            try:
                failure = {"error_type": type(error).__name__, "message": str(error)}
                with _exclusive(run_fd, "episode-failure.json") as file:
                    _write(file, failure)
                os.fsync(run_fd)
            except Exception:
                pass
        raise NativeSourceEpisodeError(_ERROR) from None
    finally:
        if journal is not None:
            journal.close()
        if run_fd is not None:
            os.close(run_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_episode(args.config, args.config_sha256)
        if receipt["status"] == "saved_and_closed":
            return 0
    except Exception:
        sys.stderr.write("native_source_episode_failed\n")
        return 2
    sys.stderr.write("native_source_episode_unknown\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
