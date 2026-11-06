"""Docker isolation for generated code; no arbitrary host execution fallback."""
import json
import selectors
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path

from .protocol import Task

APPWORLD_REV = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"


class StderrTail:
    """Drain child diagnostics without an unbounded file on the host root disk."""
    def __init__(self, stream):
        self.chunks = deque(maxlen=16)
        self.lock = threading.Lock()

        def drain():
            while chunk := stream.read1(4096):
                with self.lock:
                    self.chunks.append(chunk)

        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()

    def tail(self):
        with self.lock:
            return b"".join(self.chunks)


class AppWorldWorker:
    def __init__(self, data_dir, image="wll-appworld:local", deps_dir=None, timeout=120):
        data_dir = Path(data_dir).resolve(strict=True)
        if "," in str(data_dir):
            raise ValueError("comma is not supported in Docker mount paths")
        self.timeout = timeout
        self.name = "wll-" + uuid.uuid4().hex[:16]
        command = ["docker", "run", "--rm", "-i", "--name", self.name,
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "128",
                   "--memory", "3g", "--cpus", "2", "--user", "10001:10001",
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m,uid=10001,gid=10001",
                   "--tmpfs", "/workspace/experiments:rw,nosuid,nodev,size=1g,uid=10001,gid=10001",
                   "--mount", f"type=bind,src={data_dir},dst=/workspace/data,readonly"]
        if deps_dir:
            deps_dir = Path(deps_dir).resolve(strict=True)
            if "," in str(deps_dir):
                raise ValueError("comma is not supported in Docker mount paths")
            command.extend(["--mount", f"type=bind,src={deps_dir},dst=/opt/deps,readonly"])
        command.append(image)
        # No shell, host home, Docker socket, credentials or writable host mount.
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.error_log = StderrTail(self.proc.stderr)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def call(self, op, **kwargs):
        if self.proc.poll() is not None:
            raise RuntimeError(f"AppWorld worker exited with {self.proc.returncode}")
        self.proc.stdin.write((json.dumps({"op": op, **kwargs}) + "\n").encode())
        self.proc.stdin.flush()
        if not self.selector.select(self.timeout):
            self.close()
            raise TimeoutError(f"AppWorld controller operation timed out: {op}")
        line = self.proc.stdout.readline(4_000_000)
        if not line.endswith(b"\n"):
            self.close()
            raise RuntimeError("truncated or oversized AppWorld protocol response")
        reply = json.loads(line)
        if not reply["ok"]:
            detail = self.error_log.tail()[-8000:].decode(errors="replace") if op == "start" else ""
            raise RuntimeError(f"AppWorld {op} failed: {reply['error']}\n{detail}")
        return reply["result"]

    def close(self):
        # Killing the CLI alone can leave the container running.
        subprocess.run(["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15, check=False)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.error_log.thread.join(timeout=2)
        self.proc.stderr.close()


class AppWorldEnvironment:
    def __init__(self, task_id, data_dir, max_steps, image="wll-appworld:local", deps_dir=None):
        self.worker = AppWorldWorker(data_dir, image, deps_dir)
        self.evaluated = False
        try:
            self.worker.call("start", task_id=task_id, max_steps=max_steps)
        except BaseException:
            self.worker.close()
            raise

    def execute(self, code):
        if self.evaluated:
            raise RuntimeError("evaluation is terminal; do not resume after verifier access")
        return self.worker.call("execute", code=code)

    def completed(self):
        return self.worker.call("completed")

    def evaluate(self):
        if self.evaluated:
            raise RuntimeError("evaluation already consumed")
        self.evaluated = True
        result = self.worker.call("evaluate")
        if not isinstance(result, bool):
            raise TypeError("official success must be a boolean")
        return result

    def close(self):
        self.worker.close()


def load_tasks(data_dir, split, limit=None, image="wll-appworld:local", deps_dir=None,
               exclude_scenarios=(), one_per_scenario=False, include_task_ids=None):
    if split not in {"train", "dev", "test_normal", "test_challenge"}:
        raise ValueError("unknown official split")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    worker = AppWorldWorker(data_dir, image, deps_dir)
    try:
        ids = worker.call("list", split=split)
        if include_task_ids is not None:
            if len(include_task_ids) != len(set(include_task_ids)):
                raise ValueError("duplicate requested task IDs")
            if not include_task_ids or not set(include_task_ids) <= set(ids):
                raise ValueError("requested tasks are not a nonempty subset of the official split")
            ids = [task_id for task_id in ids if task_id in set(include_task_ids)]
        tasks = []
        seen = set(exclude_scenarios)
        for task_id in ids:
            scenario = task_id.rsplit("_", 1)[0]
            if scenario in seen:
                continue
            info = worker.call("describe", task_id=task_id)
            tasks.append(Task(task_id, scenario, split, **info))
            if one_per_scenario:
                seen.add(scenario)
            if limit is not None and len(tasks) >= limit:
                break
        return tasks
    finally:
        worker.close()
