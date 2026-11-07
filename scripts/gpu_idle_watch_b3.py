"""Frozen idle monitor flow plus one source-bound, registered six-hour B3 exception.

This separate monitor must be deployed by its owner after release and a fresh
idle window. It does not modify the original watcher or the GPU entry guard.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

from scripts.probe_chunked_prefill import PYTHON
from wll.gpu_idle import IdleTracker, parse_snapshot
from wll.gpu_policy import query_smi
from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
B3_ID = 'qwen38-b3-sft-001'
B3_PLAN = 'artifacts/qwen38-b3-sft-plan-001.json'
B3_RUNTIME = 21600
B3_SOURCE = 'scripts/gpu_idle_watch_b3.py'
_WATCHER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _b3_preflight(sha):
    # Normal, lazy import: no dynamic code or inherited monitor globals.
    from scripts.train_qwen38_b3 import preflight

    preflight(sha)


def validate_queued_job(job, *, root=None):
    """Authorize before any ticket write; all ordinary jobs retain the 7200s cap."""
    if (not isinstance(job, dict) or not isinstance(job.get('id'), str)
            or re.fullmatch('[a-z0-9_-]+', job['id']) is None
            or not isinstance(job.get('argv'), list) or not job['argv']
            or any(not isinstance(value, str) for value in job['argv'])
            or type(job.get('max_runtime_seconds')) is not int):
        raise ValueError('invalid queued command')
    if job['id'] != B3_ID:
        if not 1 <= job['max_runtime_seconds'] <= 7200:
            raise ValueError('invalid queued command')
        return
    sha = job.get('plan_sha256')
    if (job['max_runtime_seconds'] != B3_RUNTIME or not isinstance(sha, str)
            or re.fullmatch('[a-f0-9]{64}', sha) is None
            or job['argv'] != [PYTHON, '-B', '-m', 'scripts.train_qwen38_b3', '--plan-sha256', sha]):
        raise ValueError('invalid registered B3 command/runtime/hash')
    root = ROOT if root is None else Path(root)
    path, source = root / B3_PLAN, root / B3_SOURCE
    if path.is_symlink() or not path.is_file() or source.is_symlink() or not source.is_file():
        raise ValueError('B3 plan and executing watcher source must be regular bound files')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('registered B3 plan hash differs')
    plan = json.loads(raw)
    if (not isinstance(plan, dict) or plan.get('id') != B3_ID or plan.get('state') != 'registered'
            or type(plan.get('max_gpu_job_seconds')) is not int or plan['max_gpu_job_seconds'] != B3_RUNTIME
            or not isinstance(plan.get('source_sha256'), dict)
            or plan['source_sha256'].get(B3_SOURCE) != _WATCHER_SOURCE_SHA256
            or hashlib.sha256(source.read_bytes()).hexdigest() != _WATCHER_SOURCE_SHA256):
        raise ValueError('B3 registration/runtime/executing source binding differs')
    _b3_preflight(sha)


def read_snapshot(policy):
    return parse_snapshot(query_smi("index,uuid,memory.used,memory.free,utilization.gpu"),
                          query_smi("gpu_uuid,pid"), policy)


def fresh_heartbeat(path):
    try:
        return 0 <= time.time() - path.stat().st_mtime <= 60
    except FileNotFoundError:
        return False


def stop_own_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    policy_bytes = (ROOT / "config/gpu-policy.json").read_bytes()
    policy = json.loads(policy_bytes)
    if socket.gethostname() != policy["hostname"] or policy["idle_seconds"] < 180:
        raise ValueError("unauthorized monitor host or idle threshold")
    digest = hashlib.sha256(policy_bytes).hexdigest()
    if args.probe:
        print(json.dumps(read_snapshot(policy)))
        return
    artifacts = ROOT / "artifacts"
    state_path = artifacts / "gpu-idle-watch-status.json"
    queue_path = artifacts / "gpu-idle-queue.json"
    tracker = IdleTracker(policy)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    with (artifacts / "gpu-idle-watch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            now = time.monotonic()
            state = {"pid": os.getpid(), "checked_epoch": time.time(),
                     "policy_sha256": digest, "phase": "monitoring", "poll_seconds": policy["poll_seconds"],
                     "required_idle_seconds": policy["idle_seconds"]}
            try:
                if (ROOT / "config/gpu-policy.json").read_bytes() != policy_bytes:
                    raise RuntimeError("policy changed; restart monitor to reset the idle window")
                snapshot = read_snapshot(policy)
                now = time.monotonic()
                eligible = tracker.observe(snapshot, now)
                state.update(gpus=snapshot, continuous_idle_seconds=tracker.durations(now))
                if queue_path.exists():
                    job_bytes = queue_path.read_bytes()
                    job = json.loads(job_bytes)
                    job_id = job["id"]
                    if not job_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in job_id):
                        raise ValueError("invalid job ID")
                    result_path = artifacts / f"gpu-idle-job-{job_id}.json"
                    heartbeat = artifacts / f"gpu-idle-heartbeat-{job_id}"
                    state.update(queued_job=job_id, client_ready=fresh_heartbeat(heartbeat),
                                 already_attempted=result_path.exists())
                    if eligible and fresh_heartbeat(heartbeat) and not result_path.exists():
                        # A second snapshot at dispatch, then another check in the CUDA entrypoint.
                        selected = eligible[0]
                        if not read_snapshot(policy)[selected]["idle"]:
                            tracker.reset()
                            continue
                        ticket = {"gpu_index": selected, "gpu_uuid": policy["allowed_gpus"][str(selected)],
                                  "policy_sha256": digest, "boot_id": boot_id,
                                  "qualified_at_monotonic": time.monotonic(),
                                  "idle_seconds": tracker.durations(now)[selected],
                                  "sample_count": tracker.counts[selected],
                                  "max_gap_seconds": tracker.max_gaps[selected], "job_id": job_id}
                        ticket_path = artifacts / f"gpu-idle-ticket-{job_id}.json"
                        if ticket_path.exists() or ticket_path.with_suffix(".consumed.json").exists():
                            raise RuntimeError("job ticket already exists; no automatic repeat")
                        validate_queued_job(job)
                        atomic_json(ticket_path, ticket)
                        result = {"job_id": job_id, "status": "dispatching", "gpu_index": selected,
                                  "queue_sha256": hashlib.sha256(job_bytes).hexdigest(),
                                  "started_epoch": time.time(), "ticket": ticket}
                        atomic_json(result_path, result)
                        env = dict(os.environ, PYTHONPATH=str(ROOT),
                                   CUDA_VISIBLE_DEVICES=ticket["gpu_uuid"], WLL_GPU9_SHARED="0",
                                   WLL_GPU_IDLE_TICKET=str(ticket_path))
                        child = None
                        try:
                            with (artifacts / f"gpu-idle-job-{job_id}.log").open("x") as log:
                                # The hard cap survives this monitor dying or disconnecting.
                                command = ["timeout", "--signal=TERM", "--kill-after=15s",
                                           f"{job['max_runtime_seconds']}s", *job["argv"]]
                                child = subprocess.Popen(command, cwd=ROOT, env=env,
                                                         stdout=log, stderr=subprocess.STDOUT,
                                                         start_new_session=True)
                                result.update(status="running", process_group=child.pid,
                                              start_ticks=Path(f"/proc/{child.pid}/stat").read_text().split()[21])
                                atomic_json(result_path, result)
                                started = time.monotonic()
                                while child.poll() is None:
                                    state.update(phase="job_running", job=job_id, physical_gpu=selected,
                                                 already_attempted=True, continuous_idle_seconds=None,
                                                 admission_idle_seconds=ticket["idle_seconds"],
                                                 checked_epoch=time.time())
                                    try:
                                        state["gpus"] = read_snapshot(policy)
                                        state["gpu_observed_epoch"] = time.time()
                                        state.pop("gpu_observation_error", None)
                                    except Exception as exc:
                                        # Missing telemetry is not proof that the running job died.
                                        state["gpus"] = None
                                        state["gpu_observation_error"] = type(exc).__name__
                                    atomic_json(state_path, state)
                                    if not fresh_heartbeat(heartbeat):
                                        result["stop_reason"] = "client_heartbeat_expired"
                                        break
                                    if time.monotonic() - started > job["max_runtime_seconds"]:
                                        result["stop_reason"] = "runtime_limit"
                                        break
                                    time.sleep(5)
                        finally:
                            if child is not None:
                                stop_own_child(child)
                            result.update(status="finished", exit_code=child.returncode if child else None,
                                          finished_epoch=time.time())
                            atomic_json(result_path, result)
                            tracker.reset()
                        state["phase"] = "monitoring_job_finished"
            except Exception as exc:
                tracker.reset()
                state.update(phase="observation_error", error=f"{type(exc).__name__}: {exc}")
            atomic_json(state_path, state)
            time.sleep(policy["poll_seconds"])


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
