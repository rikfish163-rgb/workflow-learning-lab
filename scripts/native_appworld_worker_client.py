"""Trusted CPU parent for the fixed native worker; all payloads remain private.

Only the action string in an exact dispatch envelope may come from a policy.
This module does not run a model, expose a scorer, or load ground truth. The
systemd user service bounds a surviving worker if this parent dies; it is not
a claim of reboot durability or complete host/cgroup isolation.
"""

import hashlib
import json
import math
import os
import re
import selectors
import socket
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

PRIVATE_STATE_ROOT = Path("/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state")
ASSET_ROOT = PRIVATE_STATE_ROOT / "native-cpu-assets-001"
WORKSPACE = Path("/home/data4/hts/workflow-learning-lab/runtime-remote-001/workspace")
RUNTIME_TMP = PRIVATE_STATE_ROOT / "runtime-tmp"
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
MAX_RPC_BYTES = 6 * 65536 + 4096
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_COST_KEYS = {"action_attempts", "api_invocations", "upstream_requests"}
_FRESH_KEYS = {
    "last_successful_save_through_attempt", "current_state_saved", "score_eligible",
    "outcome_unknown", "unknown_cost",
}
_TASK_ID = re.compile(r"[0-9a-f]{7}_[1-9][0-9]*", re.ASCII)
_EXPERIMENT = re.compile(r"[A-Za-z0-9_-]{1,128}", re.ASCII)


class NativeWorkerClientError(RuntimeError):
    """Content-free trusted parent failure; inspect only private state."""


def _fail() -> None:
    raise NativeWorkerClientError("Native worker transport or saved-state validation failed.") from None


def _positive(value: Any, *, integer: bool = False) -> None:
    if type(value) not in ((int,) if integer else (int, float)):
        _fail()
    if value <= 0 or not math.isfinite(value):
        _fail()


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("ascii")


def _same(left: Any, right: Any) -> bool:
    """JSON equality with exact types (Python otherwise equates True and 1)."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    _fail()


def _values(value: Any, depth: int = 0) -> None:
    if depth > 32:
        _fail()
    if type(value) is str:
        value.encode("utf-8", errors="strict")
    elif type(value) is float:
        if not math.isfinite(value):
            _fail()
    elif type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail()
            _values(key, depth + 1)
            _values(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _values(child, depth + 1)
    elif value is not None and type(value) not in (int, bool):
        _fail()


def _decode(data: bytes) -> Any:
    try:
        value = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=_pairs,
                           parse_constant=_invalid_constant)
        _values(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        _fail()


def _open_directory(path: Path | str) -> int:
    raw, parsed = os.fspath(path), Path(path)
    if not parsed.is_absolute() or raw != str(parsed) or ".." in parsed.parts:
        _fail()
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parsed.parts[1:]:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _private(fd: int, *, directory: bool) -> os.stat_result:
    info = os.fstat(fd)
    mode = 0o700 if directory else 0o600
    if (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != mode
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        _fail()
    return info


def _private_tree(path: Path) -> int:
    if PRIVATE_STATE_ROOT not in path.parents:
        _fail()
    fd = _open_directory(PRIVATE_STATE_ROOT)
    try:
        _private(fd, directory=True)
        for part in path.relative_to(PRIVATE_STATE_ROOT).parts:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
            _private(fd, directory=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _exclusive(fd: int, name: str) -> Any:
    child = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600, dir_fd=fd)
    try:
        os.fchmod(child, 0o600)
        return os.fdopen(child, "wb", buffering=0)
    except BaseException:
        os.close(child)
        raise


def _wait_fd(fd: int, event: int, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, event)
        if not selector.select(remaining):
            _fail()


def _write_all(fd: int, data: bytes, deadline: float) -> None:
    view = memoryview(data)
    while view:
        _wait_fd(fd, selectors.EVENT_WRITE, deadline)
        try:
            count = os.write(fd, view)
        except BlockingIOError:
            continue
        if count <= 0:
            _fail()
        view = view[count:]


def _read_line(fd: int, deadline: float, limit: int = MAX_RPC_BYTES) -> bytes:
    data = bytearray()
    while True:
        _wait_fd(fd, selectors.EVENT_READ, deadline)
        try:
            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
        except BlockingIOError:
            continue
        if not chunk:
            _fail()
        data.extend(chunk)
        if len(data) > limit:
            _fail()
        if b"\n" in chunk:
            if data.find(b"\n") != len(data) - 1:
                _fail()
            return bytes(data)


class NativeWorkerClient:
    def __init__(self, task_id: str, experiment_name: str, run_dir: str | Path, *,
                 max_actions: int, max_api_requests: int, rpc_timeout: float,
                 wall_seconds: int) -> None:
        self._process = self._journal = self._stderr = None
        self._run_fd = None
        self._aborted = self._failed = self._closed = False
        self._waited = False
        self._exit_code = None
        self._lock = threading.Lock()
        self._worker_records: list[dict[str, Any]] = []
        self._cost = dict.fromkeys(sorted(_COST_KEYS), 0)
        self._freshness = dict.fromkeys(sorted(_FRESH_KEYS), False)
        self._freshness["last_successful_save_through_attempt"] = None
        self._task_id, self._experiment_name = task_id, experiment_name
        self._max_actions, self._max_api_requests = max_actions, max_api_requests
        self._rpc_timeout, self._wall_seconds = rpc_timeout, wall_seconds
        try:
            if (type(task_id) is not str or not _TASK_ID.fullmatch(task_id)
                    or type(experiment_name) is not str or not _EXPERIMENT.fullmatch(experiment_name)
                    or socket.gethostname() != "amax"):
                _fail()
            for value in (max_actions, max_api_requests, wall_seconds):
                _positive(value, integer=True)
            _positive(rpc_timeout)
            raw = os.fspath(run_dir)
            self.run_dir = Path(raw)
            if raw != str(self.run_dir) or ".." in self.run_dir.parts:
                _fail()
            self._run_fd = _private_tree(self.run_dir)
            tmp_fd = _private_tree(RUNTIME_TMP)
            os.close(tmp_fd)
            self._journal = _exclusive(self._run_fd, "transport.jsonl")
            self._stderr = _exclusive(self._run_fd, "worker.stderr.log")
            os.fsync(self._run_fd)
            token = uuid.uuid4().hex
            digest = hashlib.sha256((task_id + experiment_name + str(self.run_dir)).encode()).hexdigest()[:16]
            self._unit_name = f"wll-native-worker-{digest}-{token}.service"
            self._description = f"wll-native-worker:{token}"
            env = {key: os.environ[key] for key in (
                "PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
            ) if key in os.environ}
            env.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(RUNTIME_TMP),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="17", LANG="C.UTF-8")
            self._env = env
            bootstrap = (
                "import os,sys;assert sys.flags.no_user_site and sys.flags.safe_path;"
                "assert not sys.flags.ignore_environment and os.environ.get('PYTHONHASHSEED')=='17';"
                "sys.path.insert(0," + repr(str(WORKSPACE)) + ");"
                "from scripts.native_appworld_worker import main;raise SystemExit(main())"
            )
            args = [SYSTEMD_RUN, "--user", "--pipe", "--wait", "--collect", "--quiet",
                    f"--unit={self.unit_name}", f"--description={self._description}",
                    f"--working-directory={WORKSPACE}"]
            args += [f"--property={value}" for value in (
                "MemoryMax=3G", "CPUQuota=200%", "TasksMax=128", "NoNewPrivileges=yes",
                "RestrictAddressFamilies=AF_UNIX", f"RuntimeMaxSec={wall_seconds}",
                "KillMode=control-group",
            )]
            args += [f"--setenv={key}={env[key]}" for key in (
                "CUDA_VISIBLE_DEVICES", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "LANG",
            )]
            # -I implies -E and ignores PYTHONHASHSEED. Clear the service
            # environment explicitly, disable user-site/unsafe path insertion,
            # and retain the fixed seed before this interpreter starts.
            args += ["--", "/usr/bin/env", "-i", *[f"{key}={value}" for key, value in sorted(env.items())],
                     PYTHON, "-s", "-P", "-B", "-c", bootstrap,
                     "--task-id", task_id, "--experiment-name", experiment_name,
                     "--run-dir", str(self.run_dir), "--max-actions", str(max_actions),
                     "--max-api-requests", str(max_api_requests)]
            self._record({"event": "launch_intent", "unit": self.unit_name, "argv": args})
            self._deadline = time.monotonic() + wall_seconds
            self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=self._stderr, cwd=str(WORKSPACE), env=env,
                                             bufsize=0, close_fds=True)
            for pipe in (self._process.stdin, self._process.stdout):
                os.set_blocking(pipe.fileno(), False)
            self._record({"event": "launch_result", "pid": self._process.pid})
        except Exception:
            self.abort()
            _fail()

    def _record(self, record: dict[str, Any]) -> None:
        data = _encode(record) + b"\n"
        # The journal is an exclusive regular private file, never a pipe.
        view = memoryview(data)
        while view:
            written = self._journal.write(view)
            if not written:
                _fail()
            view = view[written:]
        os.fsync(self._journal.fileno())

    def _bounded_deadline(self, timeout: float | None) -> float:
        value = self._rpc_timeout if timeout is None else timeout
        _positive(value)
        return min(self._deadline, time.monotonic() + value)

    def _no_pending(self, *, eof_ok: bool = False) -> None:
        try:
            value = os.read(self._process.stdout.fileno(), 1)
        except BlockingIOError:
            return
        if value or not eof_ok:
            _fail()

    def _validate_reply(self, reply: Any, op: str) -> None:
        if type(reply) is not dict or type(reply.get("ok")) is not bool:
            _fail()
        required = {"ok", "cost", "freshness"}
        required |= {"result"} if reply["ok"] else {"error_type", "message"}
        if not reply["ok"] and reply.get("error_type") == "ObservedAPIError":
            required.add("status_code")
            if type(reply.get("status_code")) is not int or not 400 <= reply["status_code"] < 500:
                _fail()
        if set(reply) != required:
            _fail()
        cost, fresh = reply["cost"], reply["freshness"]
        if type(cost) is not dict or set(cost) != _COST_KEYS:
            _fail()
        for key, value in cost.items():
            if type(value) is not int or value < self._cost[key]:
                _fail()
        expected_attempts = self._cost["action_attempts"] + (op == "dispatch")
        if (cost["action_attempts"] != expected_attempts
                or cost["action_attempts"] > self._max_actions
                or cost["api_invocations"] > cost["action_attempts"]
                or cost["upstream_requests"] > self._max_api_requests
                or cost["api_invocations"] > self._max_api_requests):
            _fail()
        if op != "dispatch" and cost != self._cost:
            _fail()
        if op == "dispatch" and cost["api_invocations"] - self._cost["api_invocations"] not in (0, 1):
            _fail()
        if op == "dispatch":
            invocations = cost["api_invocations"] - self._cost["api_invocations"]
            requests = cost["upstream_requests"] - self._cost["upstream_requests"]
            expected_invocations = 0 if reply.get("error_type") == "ToolDispatchError" else 1
            if (invocations != expected_invocations or (invocations == 0 and requests != 0)
                    or (reply.get("error_type") == "ObservedAPIError" and requests == 0)):
                _fail()
        if type(fresh) is not dict or set(fresh) != _FRESH_KEYS:
            _fail()
        for key in _FRESH_KEYS - {"last_successful_save_through_attempt"}:
            if type(fresh[key]) is not bool:
                _fail()
        saved_at = fresh["last_successful_save_through_attempt"]
        if saved_at is not None and (type(saved_at) is not int or not 0 <= saved_at <= expected_attempts):
            _fail()
        if fresh["outcome_unknown"] or fresh["unknown_cost"]:
            _fail()
        previous_save = self._freshness["last_successful_save_through_attempt"]
        expected_save = expected_attempts if op == "save" and reply["ok"] else previous_save
        expected_saved = op == "save" or (op != "dispatch" and self._freshness["current_state_saved"])
        if (saved_at != expected_save or fresh["current_state_saved"] != expected_saved
                or fresh["score_eligible"] != (op == "close" and expected_saved)):
            _fail()
        if not reply["ok"]:
            if (op != "dispatch" or type(reply["message"]) is not str
                    or reply["error_type"] not in ("ToolDispatchError", "ObservedAPIError")):
                _fail()
        elif op == "metadata":
            result = reply["result"]
            if (type(result) is not dict or set(result) != {"instruction", "supervisor", "app_descriptions"}
                    or type(result["instruction"]) is not str or type(result["supervisor"]) is not dict
                    or type(result["app_descriptions"]) is not dict):
                _fail()
        elif op == "save" and reply["result"] != {"db_path": str(self._db_path)}:
            _fail()
        elif op == "close" and reply["result"] is not None:
            _fail()

    @property
    def unit_name(self) -> str:
        return self._unit_name

    @property
    def _db_path(self) -> Path:
        return ASSET_ROOT / "experiments" / "outputs" / self._experiment_name / "tasks" / self._task_id / "dbs"

    def request(self, request: dict, *, timeout: float | None = None) -> dict:
        if type(request) is not dict or type(request.get("op")) is not str:
            _fail()
        op = request["op"]
        if (set(request) != ({"op", "action"} if op == "dispatch" else {"op"})
                or op not in ("metadata", "dispatch", "save", "close")
                or (op == "dispatch" and type(request["action"]) is not str)):
            _fail()
        try:
            _values(request)
            data = _encode(request) + b"\n"
        except (ValueError, UnicodeError, RecursionError):
            _fail()
        if len(data) > MAX_RPC_BYTES or self._failed or self._aborted or self._closed or self._waited:
            _fail()
        if op == "dispatch" and (self._cost["action_attempts"] >= self._max_actions
                                 or self._cost["upstream_requests"] >= self._max_api_requests
                                 or self._cost["api_invocations"] >= self._max_api_requests):
            _fail()
        deadline = self._bounded_deadline(timeout)
        if not self._lock.acquire(blocking=False):
            _fail()
        try:
            self._no_pending()
            self._record({"event": "request_intent", "request": request})
            _write_all(self._process.stdin.fileno(), data, deadline)
            response = _decode(_read_line(self._process.stdout.fileno(), deadline))
            self._validate_reply(response, op)
            self._record({"event": "request_result", "reply": response})
            self._cost, self._freshness = response["cost"].copy(), response["freshness"].copy()
            if op == "dispatch":
                attempt = self._cost["action_attempts"]
                self._worker_records += [
                    {"event": "attempt", "attempt": attempt, "action": request["action"]},
                    {"event": "result", "attempt": attempt, "reply": _decode(_encode(response))},
                ]
            elif op == "save":
                self._worker_records += [
                    {"event": "save_attempt"},
                    {"event": "save_result", "success": True,
                     "saved_through_attempt": self._cost["action_attempts"]},
                ]
            elif op == "close":
                self._closed = True
                self._process.stdin.close()
            return response
        except Exception:
            self._failed = True
            self.abort()
            _fail()
        finally:
            self._lock.release()

    def _control(self, args: list[str], *, capture: bool = False) -> bytes:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                   stderr=self._stderr, cwd=str(WORKSPACE), env=self._env,
                                   bufsize=0, close_fds=True)
        deadline = time.monotonic() + 1.0
        try:
            result = b""
            if capture:
                os.set_blocking(process.stdout.fileno(), False)
                result = _read_line(process.stdout.fileno(), deadline, limit=256)
            if process.wait(timeout=max(0.001, deadline - time.monotonic())) != 0:
                _fail()
            if capture and os.read(process.stdout.fileno(), 1):
                _fail()
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
            if process.stdout is not None:
                process.stdout.close()

    def abort(self) -> None:
        """Best-effort bounded cleanup of this launch only; never retry an RPC."""
        if self._aborted:
            return
        self._aborted = self._failed = True
        try:
            if self._journal is not None:
                self._record({"event": "abort_intent"})
        except Exception:
            pass
        process = self._process
        if process is not None:
            try:
                description = self._control([
                    SYSTEMCTL, "--user", "show", "--property=Description", "--value", self.unit_name,
                ], capture=True)
                if description == (self._description + "\n").encode("ascii"):
                    self._record({"event": "stop_owned_unit_intent", "unit": self.unit_name})
                    self._control([SYSTEMCTL, "--user", "stop", self.unit_name])
                    self._record({"event": "stop_owned_unit_result", "unit": self.unit_name})
            except Exception:
                pass
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=0.25)
            except Exception:
                pass
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        self._close_files()

    def _close_files(self) -> None:
        for attr in ("_journal", "_stderr"):
            file = getattr(self, attr)
            if file is not None:
                try:
                    file.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._run_fd is not None:
            try:
                os.close(self._run_fd)
            except OSError:
                pass
            self._run_fd = None

    def wait(self, *, timeout: float | None = None) -> int:
        if self._waited:
            return self._exit_code
        try:
            deadline = self._bounded_deadline(timeout)
            code = self._process.wait(timeout=max(0.001, deadline - time.monotonic()))
            self._no_pending(eof_ok=True)
            self._record({"event": "exit_result", "returncode": code})
            self._waited, self._exit_code = True, code
            if self._process.stdout is not None:
                self._process.stdout.close()
            return code
        except Exception:
            self.abort()
            _fail()

    def release(self) -> None:
        """Close parent descriptors after wait observed exit, without stopping a unit."""
        if not self._waited or self._process is None or self._process.poll() is None:
            _fail()
        try:
            for pipe in (self._process.stdin, self._process.stdout):
                if pipe is not None:
                    pipe.close()
        except Exception:
            _fail()
        finally:
            self._close_files()

    def _receipt(self, name: str) -> Any:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=self._run_fd)
        try:
            _private(fd, directory=False)
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    def snapshot(self, *, timeout: float | None = None) -> dict:
        """Return a private manifest only after normal close, exit, and receipts.

        Hash the manifest before adding manifest_sha256: sorted JSON object
        keys, ASCII escapes, no nonfinite values/whitespace, ASCII bytes. File
        entries contain path/size/sha256 and are sorted by relative path.
        """
        if timeout is not None:
            self._deadline = self._bounded_deadline(timeout)
        if (not self._waited or self._exit_code != 0 or not self._closed
                or self._failed or self._aborted or not self._freshness["score_eligible"]):
            _fail()
        try:
            run_fd = _private_tree(self.run_dir)
            try:
                opened, retained = os.fstat(run_fd), os.fstat(self._run_fd)
                if (opened.st_dev, opened.st_ino) != (retained.st_dev, retained.st_ino):
                    _fail()
            finally:
                os.close(run_fd)
            with self._receipt("status.json") as file:
                data = file.read(MAX_RPC_BYTES + 1)
                if len(data) > MAX_RPC_BYTES or not data.endswith(b"\n"):
                    _fail()
                status = _decode(data)
            expected_status = {
                "state": "closed", "task_id": self._task_id, "experiment_name": self._experiment_name,
                "max_actions": self._max_actions, "max_api_requests": self._max_api_requests,
                "cost": self._cost, **self._freshness,
            }
            if not _same(status, expected_status):
                _fail()
            if (self._freshness["last_successful_save_through_attempt"] != self._cost["action_attempts"]
                    or not self._worker_records or self._worker_records[-1] != {
                        "event": "save_result", "success": True,
                        "saved_through_attempt": self._cost["action_attempts"],
                    }):
                _fail()
            with self._receipt("actions.jsonl") as file:
                for expected in self._worker_records:
                    line = file.readline(2 * MAX_RPC_BYTES + 1024)
                    if (not line.endswith(b"\n") or len(line) >= 2 * MAX_RPC_BYTES + 1024
                            or not _same(_decode(line), expected)):
                        _fail()
                if file.read(1):
                    _fail()
            db_fd = _open_directory(self._db_path)
            try:
                files: list[dict[str, Any]] = []
                self._hash_tree(db_fd, "", files)
                current_db_fd = _open_directory(self._db_path)
                try:
                    opened, current = os.fstat(db_fd), os.fstat(current_db_fd)
                    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                        _fail()
                finally:
                    os.close(current_db_fd)
            finally:
                os.close(db_fd)
            if not files:
                _fail()
            files.sort(key=lambda item: item["path"])
            manifest = {"unit_name": self.unit_name, "db_path": str(self._db_path),
                        "files": files, "cost": self._cost.copy()}
            manifest["manifest_sha256"] = hashlib.sha256(_encode(manifest)).hexdigest()
            self._record({"event": "snapshot_result", "manifest": manifest})
            return manifest
        except Exception:
            self._failed = True
            _fail()

    def _hash_tree(self, fd: int, prefix: str, files: list[dict[str, Any]]) -> None:
        if prefix.count("/") > 16 or len(files) > 4096 or time.monotonic() > self._deadline:
            _fail()
        def names_at_fd() -> list[str]:
            names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > 4096:
                        _fail()
            return sorted(names)

        names = names_at_fd()
        for name in names:
            if time.monotonic() > self._deadline:
                _fail()
            name.encode("utf-8", errors="strict")
            relative = prefix + name
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(before.st_mode):
                child = os.open(name, _DIR_FLAGS, dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                        _fail()
                    self._hash_tree(child, relative + "/", files)
                    named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
                        _fail()
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                dir_fd=fd)
                try:
                    before = os.fstat(child)
                    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                            or before.st_nlink != 1):
                        _fail()
                    digest, size = hashlib.sha256(), 0
                    while True:
                        if time.monotonic() > self._deadline:
                            _fail()
                        data = os.read(child, 1024 * 1024)
                        if not data:
                            break
                        size += len(data)
                        digest.update(data)
                    after = os.fstat(child)
                    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink",
                              "st_size", "st_mtime_ns", "st_ctime_ns")
                    named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if (size != before.st_size or any(
                        getattr(before, key) != getattr(after, key)
                        or getattr(after, key) != getattr(named, key) for key in fields
                    )):
                        _fail()
                    files.append({"path": relative, "size": size, "sha256": digest.hexdigest()})
                    if len(files) > 4096:
                        _fail()
                finally:
                    os.close(child)
            else:
                _fail()
        if names != names_at_fd():
            _fail()
