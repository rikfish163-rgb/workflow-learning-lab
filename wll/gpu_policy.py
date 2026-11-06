"""Fail-closed resource guard for this project's inference and training entrypoints."""
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .gpu_idle import parse_snapshot, validate_ticket
from .storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
SHARED_LIMIT_MIB = 20480
SHARED_HEADROOM_MIB = 2048


def query_smi(fields):
    kind = "gpu" if fields.startswith("index,") else "compute-apps"
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )


@contextmanager
def acquire_gpu(*, policy_path=ROOT / "config/gpu-policy.json",
                lock_path=ROOT / "artifacts/gpu9.lock"):
    # Validate before importing torch: CUDA visibility must not be cached first.
    for key, allowed in [("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"),
                         ("RANK", "0"), ("LOCAL_RANK", "0")]:
        if os.environ.get(key, allowed) != allowed:
            raise RuntimeError(f"single GPU policy forbids {key}={os.environ[key]}")
    if "torch" in sys.modules:
        raise RuntimeError("GPU policy must be applied before importing torch")
    sharing = os.environ.get("WLL_GPU9_SHARED", "0")
    if sharing not in {"0", "1"}:
        raise ValueError("WLL_GPU9_SHARED must be 0 or 1")
    raw = Path(policy_path).read_bytes()
    policy = json.loads(raw)
    if socket.gethostname() != policy["hostname"]:
        raise RuntimeError("GPU host is not authorized by the project policy")
    index, uuid = policy["gpu_index"], policy["gpu_uuid"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    idle_policy = "idle_seconds" in policy
    ticket_path = None
    if idle_policy:
        if sharing != "0":
            raise RuntimeError("idle-only policy forbids sharing")
        candidates = [(int(i), u) for i, u in policy["allowed_gpus"].items()
                      if visible in {i, u}]
        if visible is not None:
            if len(candidates) != 1:
                raise RuntimeError("single GPU policy rejects conflicting CUDA_VISIBLE_DEVICES")
            index, uuid = candidates[0]
        if not os.environ.get("WLL_GPU_IDLE_TICKET"):
            raise RuntimeError(f"a fresh idle ticket covering {policy['idle_seconds']} seconds is required")
        ticket_path = Path(os.environ["WLL_GPU_IDLE_TICKET"])
        ticket = json.loads(ticket_path.read_text())
        validate_ticket(ticket, policy, index, hashlib.sha256(raw).hexdigest(),
                        Path("/proc/sys/kernel/random/boot_id").read_text().strip(), time.monotonic())
    if not isinstance(index, int) or index < 0 or not uuid.startswith("GPU-"):
        raise ValueError("invalid GPU policy")
    if visible is not None and visible not in {str(index), uuid}:
        raise RuntimeError("single GPU policy rejects conflicting CUDA_VISIBLE_DEVICES")
    rows = [tuple(x.strip() for x in line.split(","))
            for line in query_smi("index,uuid").splitlines() if line.strip()]
    if (str(index), uuid) not in rows:
        raise RuntimeError("GPU identity changed; no automatic fallback is permitted")

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a project GPU job is already running") from exc
        try:
            busy = [line for line in query_smi("gpu_uuid,pid").splitlines()
                    if line.split(",")[0].strip() == uuid]
            admission = None
            if idle_policy:
                snapshot = parse_snapshot(
                    query_smi("index,uuid,memory.used,memory.free,utilization.gpu"),
                    query_smi("gpu_uuid,pid"), policy)
                if not snapshot[index]["idle"]:
                    raise RuntimeError("authorized GPU became busy after idle qualification")
                # One-use ticket is consumed under the same project lock as the GPU job.
                consumed = ticket_path.with_suffix(".consumed.json")
                if consumed.exists():
                    raise RuntimeError("idle ticket was already consumed")
                ticket_path.rename(consumed)
                os.environ["WLL_GPU_ALLOCATOR_LIMIT_MIB"] = str(SHARED_LIMIT_MIB)
                admission = {"mode": "continuous_idle_only", "ticket": ticket,
                             "required_idle_seconds": policy["idle_seconds"],
                             "gpu_uuid": uuid, "physical_gpu_index": index,
                             "pid": os.getpid(), "snapshot": snapshot[index],
                             "allocator_limit_mib": SHARED_LIMIT_MIB,
                             "checked_utc": datetime.now(UTC).isoformat()}
            elif sharing == "1":
                memory_rows = [tuple(x.strip() for x in line.split(",")) for line in
                               query_smi("index,uuid,memory.free,memory.total").splitlines()]
                matches = [row for row in memory_rows if len(row) == 4 and row[:2] == (str(index), uuid)]
                if len(matches) != 1:
                    raise RuntimeError("cannot verify authorized GPU memory headroom")
                free_mib, total_mib = map(int, matches[0][2:])
                if not SHARED_LIMIT_MIB + SHARED_HEADROOM_MIB <= free_mib <= total_mib:
                    raise RuntimeError("insufficient GPU memory headroom for bounded sharing")
                admission = {"mode": "shared_with_memory_limit", "gpu_uuid": uuid,
                    "physical_gpu_index": index, "pid": os.getpid(),
                    "checked_utc": datetime.now(UTC).isoformat(),
                    "free_mib_before_cuda": free_mib, "total_mib": total_mib,
                    "allocator_limit_mib": SHARED_LIMIT_MIB,
                    "required_headroom_mib": SHARED_HEADROOM_MIB,
                    "existing_compute_pids": [int(line.split(",")[1]) for line in busy],
                    "gpu_guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "scope": "PyTorch caching allocator limit, not an OS-wide GPU memory reservation"}
            elif busy:
                raise RuntimeError("authorized GPU is busy; refusing to start or kill other jobs")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            if admission:
                atomic_json(lock_path.parent / f"gpu9-admission-{os.getpid()}.json", admission)
            yield {"hostname": policy["hostname"], "physical_gpu_index": index,
                   "gpu_uuid": uuid, "visible_device_count": 1,
                   "policy_sha256": hashlib.sha256(raw).hexdigest()}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_cuda(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("single GPU policy requires exactly one visible CUDA device")
    if (os.environ.get("WLL_GPU9_SHARED", "0") == "1"
            or os.environ.get("WLL_GPU_ALLOCATOR_LIMIT_MIB") == str(SHARED_LIMIT_MIB)):
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        if free_bytes < (SHARED_LIMIT_MIB + SHARED_HEADROOM_MIB) * 1024**2:
            raise RuntimeError("GPU memory headroom decreased before model loading")
        torch.cuda.set_per_process_memory_fraction(SHARED_LIMIT_MIB * 1024**2 / total_bytes, 0)
