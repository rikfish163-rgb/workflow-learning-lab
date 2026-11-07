"""Private remote orchestration for the native JSON actor and task worker.

This library does not launch GPUs or authorize a scientific study. Its trusted
caller owns registration/admission, supplies the model and exact token counter,
and launches this parent in a bounded durable user unit. Scoring is separate.
"""

import json
import math
import os
import re
import socket
import subprocess
import time
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from scripts.native_appworld_actor import ActorLimits, _validate_review_hint, run_actor
from scripts.native_appworld_worker_client import (
    PRIVATE_STATE_ROOT,
    NativeWorkerClient,
    _exclusive,
    _private_tree,
)


class NativeControllerError(RuntimeError):
    """Content-free controller configuration or private persistence failure."""


def _flush(file):
    file.flush()
    os.fsync(file.fileno())


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("ascii")


def _write(file, data):
    view = memoryview(data)
    while view:
        count = file.write(view)
        if not count:
            raise OSError("Private write failed.")
        view = view[count:]
    _flush(file)


def _duration_seconds(value):
    if type(value) is not str or not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?(?:us|ms|min|s|h|d|w) ?)+", value):
        raise NativeControllerError("native_controller_worker_profile")
    units = {"us": Decimal("0.000001"), "ms": Decimal("0.001"), "s": Decimal(1),
             "min": Decimal(60), "h": Decimal(3600), "d": Decimal(86400), "w": Decimal(604800)}
    return sum((Decimal(n) * units[u] for n, u in re.findall(
        r"([0-9]+(?:\.[0-9]+)?)(us|ms|min|s|h|d|w)", value)), Decimal(0))


def inspect_worker(unit_name, *, expected_runtime_seconds):
    """Inspect this live unit and its actual PID/cgroup without reading logs."""
    if type(unit_name) is not str or not re.fullmatch(
            r"wll-native-worker-[0-9a-f]{16}-[0-9a-f]{32}\.service", unit_name):
        raise NativeControllerError("native_controller_worker_identity")
    if type(expected_runtime_seconds) is not int or expected_runtime_seconds <= 0:
        raise NativeControllerError("native_controller_worker_profile")
    properties = ["MainPID", "Description", "MemoryMax", "TasksMax", "NoNewPrivileges",
                  "RestrictAddressFamilies", "ControlGroup", "ActiveState", "RuntimeMaxUSec", "KillMode"]
    result = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit_name,
                             *["--property=" + key for key in properties]],
                            capture_output=True, timeout=5, check=False)
    if result.returncode or len(result.stdout) > 16384:
        raise NativeControllerError("native_controller_worker_identity")
    values = dict(line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line)
    token = unit_name.rsplit("-", 1)[1].removesuffix(".service")
    if (values.get("Description") != "wll-native-worker:" + token
            or values.get("MemoryMax") != "3221225472" or values.get("TasksMax") != "128"
            or values.get("NoNewPrivileges") != "yes"
            or values.get("RestrictAddressFamilies") != "AF_UNIX"
            or values.get("ActiveState") != "active" or values.get("KillMode") != "control-group"
            or _duration_seconds(values.get("RuntimeMaxUSec")) != expected_runtime_seconds):
        raise NativeControllerError("native_controller_worker_profile")
    pid = int(values["MainPID"])
    if pid <= 0:
        raise NativeControllerError("native_controller_worker_identity")
    proc = Path("/proc") / str(pid)
    before = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
    cgroup = (proc / "cgroup").read_text().strip()
    command = (proc / "cmdline").read_bytes().split(b"\0")
    environment = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
    if (command[:5] != [b"/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python",
                       b"-s", b"-P", b"-B", b"-c"]
            or environment.get(b"PYTHONHASHSEED") != b"17"
            or any(key in environment for key in (b"PYTHONPATH", b"PYTHONHOME", b"PYTHONUSERBASE",
                                                  b"PYTHONSTARTUP", b"PYTHONINSPECT"))):
        raise NativeControllerError("native_controller_worker_python_environment")
    group = values["ControlGroup"]
    if (not group.startswith("/") or ".." in Path(group).parts or cgroup != "0::" + group
            or status.get("NoNewPrivs", "").strip() != "1"
            or int(status.get("CapEff", "1").strip(), 16) != 0):
        raise NativeControllerError("native_controller_worker_profile")
    cg = Path("/sys/fs/cgroup") / group.lstrip("/")
    memory, pids = (cg / "memory.max").read_text().strip(), (cg / "pids.max").read_text().strip()
    cpu = (cg / "cpu.max").read_text().strip()
    quota, period = map(int, cpu.split())
    if memory != "3221225472" or pids != "128" or quota != 2 * period:
        raise NativeControllerError("native_controller_worker_profile")
    after = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    if before != after:
        raise NativeControllerError("native_controller_worker_identity")
    return {"unit_name": unit_name, "pid": pid, "start_ticks": int(before),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "memory_max_bytes": int(memory), "pids_max": int(pids), "cpu_max": cpu,
            "no_new_privileges": True, "effective_capabilities_zero": True,
            "restrict_address_families": values["RestrictAddressFamilies"],
            "runtime_max": values["RuntimeMaxUSec"],
            "runtime_max_seconds": expected_runtime_seconds, "kill_mode": values["KillMode"],
            "python_hash_seed": "17", "safe_path_and_no_user_site_flags": True,
            "unsafe_PYTHON_environment_absent": True,
            "scope": "Live unit/PID/cgroup inspection; no socket creation or exhaustion stress test."}


def run_task(task_id, experiment_name, run_dir, *, limits, seed, model,
             token_counter, rpc_timeout, worker_wall_seconds, setup_seconds,
             cleanup_seconds, is_policy_episode, _worker_factory=None,
             _actor=None, _inspect=inspect_worker, _clock=time.monotonic,
             source_metadata=None, prefix_records=None, _replay=None, review_hint=None):
    """Run one private fresh task; return a private execution receipt.

    Setup, actor and cleanup have separate reserves. Unknown results are never
    saved for scoring. A normal budget stop may be saved/scored as incomplete.
    The trusted caller must distinguish a scripted CPU probe from a registered
    policy episode and keep every returned snapshot/receipt private.
    """
    try:
        if socket.gethostname() != "amax" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("host")
        _validate_review_hint(review_hint)
        if (not isinstance(limits, ActorLimits) or type(worker_wall_seconds) is not int
                or type(is_policy_episode) is not bool
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                       for x in (rpc_timeout, setup_seconds, cleanup_seconds, limits.wall_seconds))
                or worker_wall_seconds < setup_seconds + limits.wall_seconds + cleanup_seconds
                or (source_metadata is None) != (prefix_records is None)):
            raise ValueError("unreserved_lifetime")
        path = Path(run_dir)
        if PRIVATE_STATE_ROOT not in path.parents or os.fspath(run_dir) != str(path):
            raise ValueError("path")
        run_fd = _private_tree(path)
    except Exception:
        raise NativeControllerError("native_controller_invalid_configuration") from None
    files, worker, released = [], None, False
    try:
        journal = _exclusive(run_fd, "controller.jsonl")
        files.append(journal)
        receipt_file = _exclusive(run_fd, "execution-receipt.json")
        files.append(receipt_file)
        os.mkdir("worker", mode=0o700, dir_fd=run_fd)
        os.fsync(run_fd)

        def record(event):
            _write(journal, _json(event) + b"\n")

        started = _clock()
        setup_deadline, deadline = started + setup_seconds, started + worker_wall_seconds

        def remaining(end=deadline):
            value = end - _clock()
            if not math.isfinite(value) or value <= 0:
                raise NativeControllerError("native_controller_deadline")
            return min(rpc_timeout, value)

        original_model_deadline = getattr(model, "deadline", None)
        record({"event": "controller_intent", "task_id": task_id,
                "experiment_name": experiment_name, "limits": asdict(limits), "seed": seed,
                "setup_seconds": setup_seconds, "cleanup_seconds": cleanup_seconds,
                "worker_wall_seconds": worker_wall_seconds, "is_policy_episode": is_policy_episode,
                "review_hint_present": review_hint is not None,
                "pid": os.getpid(), "started_monotonic": started,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        outcome, snapshot, profile, worker_exit, replay_outcome = None, None, None, None, None
        status = "unknown"
        try:
            if original_model_deadline is not None:
                model.deadline = min(original_model_deadline, setup_deadline)
            record({"event": "health_intent"})
            health = model.check_health()
            record({"event": "health_result", "health": health})
            remaining(setup_deadline)
            worker = (_worker_factory or NativeWorkerClient)(
                task_id, experiment_name, path / "worker", max_actions=limits.max_actions,
                max_api_requests=limits.max_api_requests, rpc_timeout=rpc_timeout,
                wall_seconds=worker_wall_seconds)
            record({"event": "metadata_intent", "unit_name": worker.unit_name})
            metadata = worker.request({"op": "metadata"}, timeout=remaining(setup_deadline))
            record({"event": "metadata_result", "reply": metadata})
            if (type(metadata) is not dict or metadata.get("ok") is not True
                    or type(metadata.get("cost")) is not dict
                    or set(metadata["cost"]) != {"action_attempts", "api_invocations", "upstream_requests"}
                    or any(type(v) is not int or v != 0 for v in metadata["cost"].values())):
                raise NativeControllerError("native_controller_nonfresh_worker")
            profile = _inspect(worker.unit_name, expected_runtime_seconds=worker_wall_seconds)
            record({"event": "worker_effective_profile", "profile": profile})
            remaining(setup_deadline)
            actor_metadata, verified_prefix, initial_cost = metadata["result"], None, None
            if prefix_records is not None:
                from scripts.native_appworld_replay import replay_prefix

                replayed = (_replay or replay_prefix)(worker, source_metadata=source_metadata,
                    prefix_records=prefix_records, record=record, deadline=setup_deadline,
                    max_actions=limits.max_actions, max_api_requests=limits.max_api_requests,
                    clock=_clock)
                record({"event": "prefix_replay_result", "result": replayed})
                replay_outcome = {key: value for key, value in replayed.items()
                                  if key not in {"metadata", "prefix_records"}}
                if replayed["status"] != "matched":
                    if replayed["status"] == "observation_mismatch":
                        status = "observation_mismatch"
                    raise NativeControllerError("native_controller_replay_not_matched")
                actor_metadata, verified_prefix = replayed["metadata"], replayed["prefix_records"]
                initial_cost = replayed["worker_cost"]
            actor_seconds = min(limits.wall_seconds, deadline - _clock() - cleanup_seconds)
            if actor_seconds <= 0:
                raise NativeControllerError("native_controller_no_actor_reserve")
            if original_model_deadline is not None:
                model.deadline = min(original_model_deadline, _clock() + actor_seconds)
            outcome = (_actor or run_actor)(worker, model, token_counter,
                metadata=actor_metadata, limits=replace(limits, wall_seconds=actor_seconds),
                seed=seed, record=record, clock=_clock, prefix_records=verified_prefix,
                initial_worker_cost=initial_cost, review_hint=review_hint)
            record({"event": "actor_result", "outcome": outcome})
            if (outcome["status"] not in {"incomplete", "model_reported_complete"}
                    or outcome["outcome_unknown"] is not False or outcome["unknown_cost"] is not False):
                raise NativeControllerError("native_controller_unknown_actor")
            record({"event": "save_intent"})
            saved = worker.request({"op": "save"}, timeout=remaining())
            record({"event": "save_result", "reply": saved})
            record({"event": "close_intent"})
            closed = worker.request({"op": "close"}, timeout=remaining())
            record({"event": "close_result", "reply": closed})
            worker_exit = worker.wait(timeout=remaining())
            if type(worker_exit) is not int or worker_exit != 0:
                raise NativeControllerError("native_controller_worker_exit")
            snapshot = worker.snapshot(timeout=remaining())
            remaining()
            if (snapshot["unit_name"] != worker.unit_name
                    or snapshot["cost"] != {k: outcome["cost"][k] for k in (
                        "action_attempts", "api_invocations", "upstream_requests")}):
                snapshot = None
                raise NativeControllerError("native_controller_snapshot_cost")
            record({"event": "execution_snapshot", "snapshot": snapshot})
            worker.release()
            released = True
            status = "saved_and_closed"
        except Exception as error:
            snapshot = None
            try:
                message = str(error)
            except Exception:
                message = "Exception message unavailable."
            record({"event": "controller_failure", "error_type": type(error).__name__, "message": message})
        finally:
            if worker is not None and not released:
                worker.abort()
        result = {"status": status, "task_id": task_id, "experiment_name": experiment_name,
                  "is_policy_episode": is_policy_episode, "actor_outcome": outcome,
                  "snapshot": snapshot, "worker_exit_code": worker_exit,
                  "replay_outcome": replay_outcome,
                  "worker_profile": profile, "new_GPU_jobs_started_by_controller": 0,
                  "official_scores_computed": 0, "elapsed_seconds": _clock() - started}
        record({"event": "controller_stop", "result": result})
        _write(receipt_file, _json(result) + b"\n")
        os.fsync(run_fd)
        return result
    except Exception:
        if worker is not None and not released:
            worker.abort()
        raise NativeControllerError("native_controller_private_persistence_or_lifecycle_failed") from None
    finally:
        for file in files:
            file.close()
        os.close(run_fd)
