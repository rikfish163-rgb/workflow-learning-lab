"""One-shot serial coordinator for the independently registered native source pilot.

All task metadata, raw requests/replies, prefixes, targets, scores and diagnostics
stay in the registered private directory. The injected runtime is for CPU tests;
production uses the same fixed episode/scorer commands and the existing queue.
Import and help perform no registration, task, model, service or GPU operation.
"""

import argparse
import copy
import hashlib
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from scripts import native_appworld_actor as actor
from scripts import native_source_episode as episode
from scripts import native_source_gpu_queue as queue
from scripts import native_source_registration as registration
from scripts.native_appworld_controller import _duration_seconds
from scripts.native_appworld_model_client import _HTTP_CHILD
from scripts.native_appworld_replay import _inputs, _metadata, _observation
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _exclusive
from scripts.native_verification_token_counter import count_input_tokens
from scripts.verification_token_counter import _message_payload

PULSE_SECONDS = 5
RELEASE_SECONDS = 60
SHUTDOWN_RESERVE_SECONDS = 120
_ERROR = "native_source_pilot_failed"
_COST_KEYS = ("generations", "prompt_tokens", "completion_tokens", "action_attempts",
              "api_invocations", "upstream_requests")
_UNIT = re.compile(r"wll-native-source-[A-Za-z0-9_-]{1,120}-[a-f0-9]{32}\.service")


class NativeSourcePilotError(RuntimeError):
    """Content-free coordinator failure; inspect only its private evidence."""


class _BudgetStop(Exception):
    pass


def _require(value):
    if not value:
        raise NativeSourcePilotError(_ERROR)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def initial_slots():
    return [{"source_index": index, "task_id": task, "pair_class": "unstarted",
             **{phase: {"status": "unstarted"} for phase in ("first_attempt", "control", "review")}}
            for index, task in enumerate(registration.TASK_IDS)]


def unit_command(kind, unit, arguments, *, root=WORKSPACE, log_path):
    """Construct only a trusted episode/scorer command with explicit limits."""
    _require(kind in {"episode", "scorer", "pilot"} and type(unit) is str and _UNIT.fullmatch(unit)
             and type(arguments) is list and all(type(value) is str and "\0" not in value for value in arguments)
             and Path(root).is_absolute() and Path(log_path).is_absolute())
    module = {"episode": "scripts.native_source_episode", "scorer": "scripts.native_appworld_scorer",
              "pilot": "scripts.run_native_source_pilot"}[kind]
    duration, memory, cpu, tasks, families = (
        (120, "3G", "200%", 128, "AF_UNIX") if kind == "scorer" else
        (900 if kind == "episode" else 282600, "6G", "400%", 256, "AF_UNIX AF_INET AF_INET6"))
    environment = {key: os.environ[key] for key in (
        "PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS") if key in os.environ}
    environment.update(CUDA_VISIBLE_DEVICES="", TMPDIR=registration.TMPDIR,
                       PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="17", LANG="C.UTF-8",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")
    bootstrap = ("import os,runpy,sys;assert sys.flags.no_user_site and sys.flags.safe_path;"
                 "assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';"
                 "sys.path.insert(0," + repr(str(root)) + ");"
                 "runpy.run_module(" + repr(module) + ",run_name='__main__')")
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--service-type=exec",
            "--unit=" + unit, "--description=wll-native-source:" + unit,
            "--working-directory=" + str(root),
            *["--property=" + value for value in (
                f"RuntimeMaxSec={duration}", "MemoryMax=" + memory, "CPUQuota=" + cpu,
                f"TasksMax={tasks}", "NoNewPrivileges=yes", "KillMode=control-group", "RemainAfterExit=yes",
                "RestrictAddressFamilies=" + families, "UMask=0077",
                "StandardOutput=append:" + str(log_path), "StandardError=append:" + str(log_path))],
            "--", "/usr/bin/env", "-i", *[key + "=" + value for key, value in sorted(environment.items())],
            registration.PYTHON, "-s", "-P", "-B", "-c", bootstrap, *arguments]


def _journal(raw):
    _require(type(raw) is bytes and bool(raw) and raw.endswith(b"\n") and len(raw) <= 64_000_000)
    values = [_decode(line) for line in raw.splitlines()]
    _require(len(values) <= 100000 and all(type(row) is dict and type(row.get("event")) is str for row in values))
    return values


def _one(rows, name):
    values = [row for row in rows if row["event"] == name]
    _require(len(values) == 1)
    return values[0]


def actor_evidence(receipt, rows, *, expected_context, prefix_records=None, review_hint=None):
    """Bind private new-generation/action pairs without accessing score feedback."""
    metadata_reply = _one(rows, "metadata_result")["reply"]
    _require(type(metadata_reply) is dict and metadata_reply.get("ok") is True)
    metadata = metadata_reply["result"]
    _metadata(metadata)
    _require(_hash(_encode(metadata)) == expected_context["metadata_sha256"])
    expected_messages = actor.build_actor_messages(metadata, prefix_records=prefix_records,
                                                   review_hint=review_hint)
    start = _one(rows, "actor_start")
    _require(registration._same(start["messages"], expected_messages)
             and _message_payload(expected_messages)[1] == expected_context["messages_sha256"]
             and start.get("review_hint_index") == (2 + 2 * len(prefix_records or [])
                                                    if review_hint is not None else None))
    current, pending, records, generations = copy.deepcopy(expected_messages), {}, [], []
    for row in rows:
        event = row["event"]
        if event == "generation_intent":
            _require(not pending and registration._same(row["messages"], current))
            pending = {"intent": row}
        elif event == "generation_result":
            _require(set(pending) == {"intent"} and type(row.get("response")) is dict)
            pending["response"] = row["response"]
        elif event == "dispatch_intent":
            _require(set(pending) == {"intent", "response"}
                     and type(row.get("action")) is str and row["action"] == pending["response"].get("text"))
            pending["action"] = row["action"]
        elif event == "dispatch_result":
            _require(set(pending) == {"intent", "response", "action"} and type(row.get("reply")) is dict)
            reply = row["reply"]
            observation = {key: value for key, value in reply.items() if key not in {"cost", "freshness"}}
            _observation(observation)
            record = {"action": pending["action"], "observation": observation}
            records.append(record)
            generations.append({**copy.deepcopy(pending), "reply": copy.deepcopy(reply)})
            current.extend([{"role": "assistant", "content": record["action"]},
                            {"role": "user", "content": _encode(observation).decode("ascii")}])
            pending = {}
    outcome = receipt.get("actor_outcome")
    _require(type(outcome) is dict and outcome.get("outcome_unknown") is False
             and outcome.get("unknown_cost") is False
             and outcome.get("status") in {"incomplete", "model_reported_complete"}
             and outcome["cost"]["action_attempts"] == len(records) + len(prefix_records or [])
             and outcome["cost"]["generations"] == sum(row["event"] == "generation_result" for row in rows)
             and registration._same(_one(rows, "actor_stop")["outcome"], outcome))
    return metadata, records, generations


def select_prefix(metadata, records):
    """First confirmed actor-visible 4xx only; terminal scores are not inputs."""
    for index, record in enumerate(records):
        observation = record["observation"]
        _observation(observation)
        if observation.get("ok") is True and actor._is_completion(record["action"]):
            return None
        if observation.get("error_type") == "ObservedAPIError":
            if not 1 <= index + 1 <= 54:
                return None
            prefix = records[:index + 1]
            _inputs(metadata, prefix, lambda _event: None, time.monotonic() + 120,
                    60, 60, time.monotonic)
            return {"metadata": copy.deepcopy(metadata), "prefix_records": copy.deepcopy(prefix)}
    return None


def student_targets(generations, *, prefix_length, hint):
    """Preserve full histories, deleting only the one fixed review message."""
    targets = {}
    hint_index = 2 + 2 * prefix_length
    for index, generation in enumerate(generations):
        response, action, reply = generation["response"], generation["action"], generation["reply"]
        try:
            raw = response["raw_reply"]
            choice = raw["choices"][0]
            messages = copy.deepcopy(generation["intent"]["messages"])
            _require(response.get("format_error") is None and response.get("text") == action
                     and reply.get("ok") is True and type(raw["choices"]) is list and len(raw["choices"]) == 1
                     and choice.get("finish_reason") == "stop"
                     and choice["message"] == {"role": "assistant", "content": action}
                     and len(messages) > hint_index
                     and messages[hint_index] == {"role": "user", "content": hint})
            parsed = _decode(action.encode("utf-8"))
            _require(type(parsed) is dict and set(parsed) == {"name", "arguments"}
                     and type(parsed["name"]) is str and bool(parsed["name"])
                     and type(parsed["arguments"]) is dict)
            teacher_sha = _message_payload(messages)[1]
            del messages[hint_index]
            student_sha = _message_payload(messages)[1]
            digest = _hash(_encode(parsed))
            targets.setdefault(digest, {"target_sha256": digest, "action": action,
                                       "student_messages": messages, "teacher_messages_sha256": teacher_sha,
                                       "student_messages_sha256": student_sha, "new_generation_index": index})
        except (KeyError, IndexError, TypeError, ValueError, NativeSourcePilotError):
            continue
    return list(targets.values())


class _Heartbeat:
    def __init__(self, runtime, job, *, interval=PULSE_SECONDS):
        self.runtime, self.job, self.interval = runtime, job, interval
        self.event, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.event.wait(self.interval):
            try:
                state = self.runtime.queue("pulse", self.job)
                _queue_state(state, self.runtime)
                self.runtime.monitor()
                self.runtime.record({"event": "GPU_heartbeat", "state": state})
            except BaseException as error:
                self.error = error
                self.event.set()

    def start(self):
        self.thread.start()
        return self

    def check(self):
        _require(self.error is None)

    def stop(self):
        self.event.set()
        self.thread.join(10)
        _require(not self.thread.is_alive())
        self.check()


def _queue_state(state, runtime, *, stopping=False):
    _require(type(state) is dict and state.get("owned") is True
             and state.get("monitor_live") is True and _finite(state.get("epoch"))
             and 0 <= runtime.epoch() - state["epoch"] < 5
             and (stopping or state.get("admission_cancelled") is False))


class _Runtime:
    """Fixed trusted OS operations; no generated code is accepted by this layer."""
    def __init__(self, root):
        self.root, self.private, self.plan = Path(root), None, None
        self.journal, self.fd, self.lock = None, None, threading.Lock()
        self.units = {}

    def now(self):
        return time.monotonic()

    def epoch(self):
        return time.time()

    def wait(self, seconds):
        _require(_finite(seconds) and 0 < seconds <= PULSE_SECONDS)
        time.sleep(seconds)

    def owner(self):
        return uuid.uuid4().hex

    def load_plan(self, sha):
        _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
        self.plan = registration.load_registration(self.root, sha, private=True)
        registration.load_registration(queue.REMOTE, sha, private=False)
        return self.plan

    def begin(self, plan, sha, slots):
        self.private = Path(plan["private_run_root"])
        self.fd = episode._private_directory(self.private)
        with _exclusive(self.fd, "study-intent.json") as file:
            self._write(file, {"plan_sha256": sha, "task_ids": list(registration.TASK_IDS),
                               "slots": slots, "automatic_retry": False, "pid": os.getpid(),
                               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        self.journal = _exclusive(self.fd, "study-journal.jsonl")
        os.fsync(self.fd)

    @staticmethod
    def _write(file, value):
        data = memoryview(_encode(value) + b"\n")
        while data:
            written = file.write(data)
            _require(type(written) is int and written > 0)
            data = data[written:]
        file.flush()
        os.fsync(file.fileno())

    def record(self, value):
        with self.lock:
            self._write(self.journal, value)

    def write(self, name, value):
        _require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", name))
        with _exclusive(self.fd, name) as file:
            self._write(file, value)
        os.fsync(self.fd)
        raw = self.read(self.private / name)
        return {"path": str(self.private / name), "sha256": _hash(raw)}

    def read(self, path):
        return episode._read_private(path, maximum=64_000_000)

    def context_rows(self, plan):
        binding = plan["context_proof"]
        raw = registration.public_read(self.root, binding["path"])
        _require(_hash(raw) == binding["sha256"])
        return _decode(raw)["rows"]

    def queue(self, operation, job, known_pids=()):
        return queue.queue_operation(operation, job, root=queue.REMOTE, known_pids=known_pids)

    def heartbeat(self, job):
        return _Heartbeat(self, job).start()

    def monitor(self, *, empty_port=False):
        _require(self.plan is not None)
        watched = _decode(registration.public_read(queue.REMOTE, "artifacts/gpu-idle-watch-status.json"))
        saved = _decode(registration.public_read(queue.REMOTE, "artifacts/gpu-idle-watch-process.json"))
        pid = saved.get("pid")
        _require(type(pid) is int and pid > 0 and watched.get("pid") == pid
                 and _finite(watched.get("checked_epoch")) and 0 <= self.epoch() - watched["checked_epoch"] < 35
                 and watched.get("policy_sha256") == self.plan["gpu_policy_sha256"])
        proc = Path("/proc") / str(pid)
        before = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        expected = [registration.PYTHON.encode(), b"-B", b"-m", b"scripts.gpu_idle_watch_b3"]
        _require(proc.stat().st_uid == os.geteuid() and str((proc / "cwd").resolve()) == str(queue.REMOTE)
                 and before == saved.get("start_ticks") and argv in (expected, expected[:1] + [b"-u"] + expected[1:])
                 and registration._public_hash(queue.REMOTE, "scripts/gpu_idle_watch_b3.py")
                 == self.plan["source_sha256"]["scripts/gpu_idle_watch_b3.py"]
                 and before == (proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
        if empty_port:
            _require(not self.port_owners())
        return {"pid": pid, "start_ticks": before, "checked_epoch": watched["checked_epoch"]}

    def port_owners(self):
        inodes = set()
        for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
            for line in path.read_text().splitlines()[1:]:
                columns = line.split()
                if columns[3] == "0A" and int(columns[1].rsplit(":", 1)[1], 16) == registration.PORT:
                    inodes.add("socket:[" + columns[9] + "]")
        owners = set()
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                for fd in (proc / "fd").iterdir():
                    if os.readlink(fd) in inodes:
                        owners.add(int(proc.name))
            except (OSError, PermissionError):
                continue
        _require(not inodes or bool(owners))
        return owners

    def ready(self, job, state):
        info = state.get("job")
        if type(info) is not dict or info.get("status") != "running":
            return None
        ready_path = queue.ready_path(queue.REMOTE / "artifacts", job)
        if not ready_path.exists():
            return None
        spec = _encode({"url": f"http://127.0.0.1:{registration.PORT}/health", "body": None, "timeout": 5})
        response = subprocess.run([registration.PYTHON, "-I", "-B", "-c", _HTTP_CHILD], input=spec,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6, check=False,
                                  env={"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                                       "TMPDIR": registration.TMPDIR, "LANG": "C.UTF-8"})
        _require(response.returncode == 0 and len(response.stdout) <= 4_000_000)
        health = _decode(response.stdout)
        _require(health.get("model") == registration.MODEL_ID and health.get("ready") is True
                 and health.get("busy") is False and type(health.get("max_context")) is int
                 and health["max_context"] == 16384 and type(health.get("max_output_tokens")) is int
                 and health["max_output_tokens"] == 1024)
        episode._backend(self.plan, health["backend"])
        config = {"job": job, "expected_backend": health["backend"]}
        fresh, _ready, identity = episode._queue_binding(config, queue.queue_operation)
        _require(self.port_owners() == set(fresh["model_pids"]))
        return {"backend": health["backend"], "identity": identity}

    def launch(self, kind, unit, arguments, *, log_name):
        _require(unit not in self.units)
        with _exclusive(self.fd, log_name):
            pass
        command = unit_command(kind, unit, arguments, root=self.root, log_path=self.private / log_name)
        spec = {"kind": kind, "unit": unit, "command": command}
        self.units[unit] = spec
        self.record({"event": "unit_launch_intent", **spec})
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=10, check=False)
            _require(result.returncode == 0)
            self.record({"event": "unit_launch_return", "unit": unit, "exit_code": 0})
        except BaseException:
            self.stop_unit(spec)
            raise
        return spec

    def unit_status(self, spec):
        unit = spec["unit"]
        _require(unit in self.units and self.units[unit] == spec)
        result = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
            *["--property=" + name for name in ("MainPID", "ActiveState", "SubState", "LoadState", "Result", "ExecMainStatus",
                "ExecMainCode", "Description", "MemoryMax", "CPUQuotaPerSecUSec", "TasksMax",
                "NoNewPrivileges", "KillMode", "RestrictAddressFamilies", "RuntimeMaxUSec", "ControlGroup")]],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False)
        _require(result.returncode == 0 and len(result.stdout) <= 32768)
        values = dict(line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line)
        scorer = spec["kind"] == "scorer"
        _require(values["Description"] == "wll-native-source:" + unit
                 and values["MemoryMax"] == str((3 if scorer else 6) * 1024**3)
                 and _duration_seconds(values["CPUQuotaPerSecUSec"]) == (2 if scorer else 4)
                 and values["TasksMax"] == ("128" if scorer else "256")
                 and values["NoNewPrivileges"] == "yes" and values["KillMode"] == "control-group"
                 and set(values["RestrictAddressFamilies"].split()) ==
                 ({"AF_UNIX"} if scorer else {"AF_UNIX", "AF_INET", "AF_INET6"})
                 and _duration_seconds(values["RuntimeMaxUSec"]) == (120 if scorer else 900))
        pid = int(values["MainPID"])
        group = values["ControlGroup"]
        _require(not group or (group.startswith("/") and ".." not in Path(group).parts))
        if group:
            spec.setdefault("cgroup", group)
            _require(spec["cgroup"] == group)
        running = pid > 0 or values["ActiveState"] in {"activating", "deactivating"} or (
            values["ActiveState"] == "active" and values["SubState"] != "exited")
        if pid > 0:
            proc = Path("/proc") / str(pid)
            ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
            command = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            expected = spec["command"][spec["command"].index(registration.PYTHON):]
            environment = dict(item.split(b"=", 1) for item in
                               (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
            _require(proc.stat().st_uid == os.geteuid() and command == [value.encode() for value in expected]
                     and (proc / "cgroup").read_text().strip() == "0::" + group
                     and environment.get(b"CUDA_VISIBLE_DEVICES") == b""
                     and environment.get(b"TMPDIR") == registration.TMPDIR.encode()
                     and environment.get(b"PYTHONHASHSEED") == b"17"
                     and not any(key in environment for key in (b"PYTHONPATH", b"PYTHONHOME", b"PYTHONUSERBASE",
                                                               b"PYTHONSTARTUP", b"PYTHONINSPECT"))
                     and status.get("NoNewPrivs", "").strip() == "1"
                     and int(status.get("CapEff", "1").strip(), 16) == 0)
            cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
            quota, period = map(int, (cgroup / "cpu.max").read_text().split())
            _require((cgroup / "memory.max").read_text().strip() == values["MemoryMax"]
                     and (cgroup / "pids.max").read_text().strip() == values["TasksMax"]
                     and quota == (2 if scorer else 4) * period
                     and ticks == (proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
            identity = {"pid": pid, "start_ticks": ticks, "cgroup": group,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            spec.setdefault("process_identity", identity)
            _require(spec["process_identity"] == identity)
        if not running:
            cgroup = Path("/sys/fs/cgroup") / group.lstrip("/") if group else None
            _require(cgroup is None or not cgroup.exists() or not (cgroup / "cgroup.procs").read_text().strip())
        return {"running": running, "verified": True, "main_pid": pid,
                "exit_code": None if running else int(values["ExecMainStatus"]),
                "success": values.get("Result") == "success" and values.get("ExecMainCode") == "1"}

    def stop_unit(self, spec):
        _require(self.unit_status(spec)["verified"])
        result = subprocess.run(["/usr/bin/systemctl", "--user", "stop", spec["unit"]],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10, check=False)
        _require(result.returncode == 0)
        checked = subprocess.run(["/usr/bin/systemctl", "--user", "show", spec["unit"],
                                  "--property=MainPID", "--property=LoadState", "--property=Description"],
                                 stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False)
        _require(checked.returncode == 0 and len(checked.stdout) <= 8192)
        values = dict(line.split("=", 1) for line in checked.stdout.decode().splitlines() if "=" in line)
        _require(values.get("MainPID") == "0" and (values.get("LoadState") == "not-found"
                 or values.get("Description") == "wll-native-source:" + spec["unit"]))
        group = spec.get("cgroup")
        directory = Path("/sys/fs/cgroup") / group.lstrip("/") if group else None
        _require(directory is None or not directory.exists() or not (directory / "cgroup.procs").read_text().strip())
        return {"verified": True, "unit": spec["unit"]}

    def artifacts(self, run_dir):
        raw = self.read(Path(run_dir) / "execution-receipt.json")
        return {"receipt": _decode(raw), "receipt_binding": {
            "path": str(Path(run_dir) / "execution-receipt.json"), "sha256": _hash(raw)},
            "journal": _journal(self.read(Path(run_dir) / "controller.jsonl"))}

    def create_score_directory(self, path):
        _require(Path(path).parent == self.private)
        os.mkdir(Path(path).name, mode=0o700, dir_fd=self.fd)
        os.fsync(self.fd)

    def score_result(self, path):
        return _decode(self.read(Path(path) / "scoring-result.json"))

    def count(self, messages):
        return count_input_tokens(messages, timeout_seconds=55)

    def cleanup_worker(self, run_dir, task_id, experiment):
        path = Path(run_dir) / "worker/transport.jsonl"
        if not path.exists():
            return {"verified": True, "launch_intent_absent": True}
        rows = _journal(self.read(path))
        first = rows[0]
        unit = first.get("unit")
        digest = _hash((task_id + experiment + str(Path(run_dir) / "worker")).encode())[:16]
        match = re.fullmatch(r"wll-native-worker-" + digest + r"-([a-f0-9]{32})\.service", unit or "")
        _require(first.get("event") == "launch_intent" and match is not None
                 and type(first.get("argv")) is list
                 and "--unit=" + unit in first["argv"]
                 and "--description=wll-native-worker:" + match.group(1) in first["argv"])
        result = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
                                "--property=Description", "--property=MainPID", "--property=ControlGroup"],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False)
        _require(result.returncode == 0 and len(result.stdout) <= 8192)
        state = dict(line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line)
        # Worker --collect may already have removed a confirmed stopped unit.
        _require(state.get("Description") in {"wll-native-worker:" + match.group(1), unit})
        if int(state.get("MainPID", "-1")) > 0:
            _require(state["Description"] == "wll-native-worker:" + match.group(1))
            stopped = subprocess.run(["/usr/bin/systemctl", "--user", "stop", unit],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=10, check=False)
            _require(stopped.returncode == 0)
        from scripts.native_appworld_scorer import _main_pid

        _require(_main_pid(unit) == 0)
        group = state.get("ControlGroup", "")
        _require(not group or (group.startswith("/") and ".." not in Path(group).parts))
        directory = Path("/sys/fs/cgroup") / group.lstrip("/") if group else None
        _require(directory is None or not directory.exists() or not (directory / "cgroup.procs").read_text().strip())
        return {"verified": True, "unit": unit, "identity_source": "first_private_transport_launch_intent"}

    def never_started(self, job, state):
        _require(state.get("admission_cancelled") is True and state.get("job") is None
                 and state.get("model_pids") == [])
        art = queue.REMOTE / "artifacts"
        _require(not queue.entry_path(art, job).exists() and not queue.ready_path(art, job).exists())
        return True

    def close(self):
        if self.journal is not None:
            self.journal.close()
        if self.fd is not None:
            os.close(self.fd)


def _unit(runtime, kind, unit, arguments, *, guard):
    spec = runtime.launch(kind, unit, arguments, log_name=unit + ".log")
    started = runtime.now()
    limit = 120 if kind == "scorer" else 900
    try:
        while True:
            guard()
            state = runtime.unit_status(spec)
            _require(state.get("verified") is True and type(state.get("running")) is bool)
            if not state["running"]:
                _require(type(state.get("exit_code")) is int and state["exit_code"] == 0
                         and state.get("success") is True and state.get("main_pid") == 0)
                return spec, runtime.now() - started
            _require(runtime.now() - started < limit)
            runtime.wait(PULSE_SECONDS)
    except BaseException:
        runtime.stop_unit(spec)
        raise


def _costs(receipt, journal):
    outcome = receipt.get("actor_outcome") or {}
    observed = outcome.get("cost") or {}
    result = {}
    for key in _COST_KEYS:
        value = observed.get(key)
        _require(value is None or (type(value) is int and value >= 0))
        result[key] = value
    returned = sum(row.get("event") == "generation_result" for row in journal)
    intents = sum(row.get("event") == "generation_intent" for row in journal)
    result["generation_results"] = returned
    result["generation_intents"] = intents
    result["dispatch_intents"] = sum(row.get("event") == "dispatch_intent" for row in journal)
    result["dispatch_results"] = sum(row.get("event") == "dispatch_result" for row in journal)
    result["unreturned_generations"] = (None if result["generations"] is None else
                                       max(0, result["generations"] - returned))
    result["unreturned_generation_intents_upper_bound"] = max(0, intents - returned)
    for key in ("prompt_tokens", "completion_tokens"):
        result["journal_returned_" + key + "_lower_bound"] = sum(
            row["response"][key] for row in journal if row.get("event") == "generation_result"
            and type(row.get("response")) is dict and type(row["response"].get(key)) is int
            and row["response"][key] >= 0)
    result["counts_complete"] = all(result[key] is not None for key in _COST_KEYS) and (
        outcome.get("outcome_unknown") is False and outcome.get("unknown_cost") is False)
    result["unknown_cost"] = not result["counts_complete"]
    result["replay"] = copy.deepcopy(receipt.get("replay_outcome"))
    result["controller_elapsed_seconds"] = receipt.get("elapsed_seconds")
    result["CPU_counting_setup_save_cleanup_included_in_parent_wall"] = True
    return result


def _run_episode(runtime, plan, sha, job, service, index, phase, initial, prefix, guard, report):
    owner = job["owner"]
    run_dir = Path(plan["private_run_root"]) / f"unit-{index:02d}-{phase}-{owner}"
    experiment = f"native-source-001-{index:02d}-{phase}-{owner}"
    config = {"plan_sha256": sha, "source_index": index, "phase": phase, "run_dir": str(run_dir),
              "job": job, "expected_backend": service["backend"], "prefix_binding": prefix,
              "initial_context": initial}
    binding = runtime.write(f"config-{index:02d}-{phase}.json", config)
    unit = f"wll-native-source-{index:02d}-{phase}-{owner}.service"
    parent = None
    cost_recorded = False
    started = runtime.now()
    try:
        parent, wall = _unit(runtime, "episode", unit,
                             ["--config", binding["path"], "--config-sha256", binding["sha256"]], guard=guard)
        evidence = runtime.artifacts(run_dir)
        receipt, journal = evidence["receipt"], evidence["journal"]
        report["costs"]["episodes"].append({"source_index": index, "phase": phase,
                                            "parent_wall_seconds": wall, **_costs(receipt, journal)})
        cost_recorded = True
        _require(receipt.get("task_id") == registration.TASK_IDS[index]
                 and receipt.get("status") == "saved_and_closed" and receipt.get("is_policy_episode") is True
                 and type(receipt.get("worker_exit_code")) is int and receipt["worker_exit_code"] == 0
                 and type(evidence.get("receipt_binding")) is dict
                 and evidence["receipt_binding"]["path"] == str(run_dir / "execution-receipt.json")
                 and _sha(evidence["receipt_binding"]["sha256"]))
        prefix_value = None if prefix is None else _decode(runtime.read(prefix["path"]))
        hint = plan["design"]["review"]["hint"] if phase == "review" else None
        metadata, records, generations = actor_evidence(
            receipt, journal, expected_context=initial,
            prefix_records=None if prefix_value is None else prefix_value["prefix_records"], review_hint=hint)
        if phase != "first_attempt":
            _require(type(receipt.get("replay_outcome")) is dict
                     and receipt["replay_outcome"].get("status") == "matched")
        # This selection occurs before the scorer is launched and has no score input.
        boundary = select_prefix(metadata, records) if phase == "first_attempt" else None
        cleanup = runtime.cleanup_worker(run_dir, registration.TASK_IDS[index], experiment)
        _require(cleanup.get("verified") is True)
        runtime.record({"event": "worker_release", "source_index": index, "phase": phase, "release": cleanup})
        score_dir = Path(plan["private_run_root"]) / f"score-{index:02d}-{phase}-{owner}"
        runtime.create_score_directory(score_dir)
        score_unit = f"wll-native-source-{index:02d}-score-{phase}-{owner}.service"
        scorer = None
        score_started, score_validated = runtime.now(), False
        try:
            scorer, score_wall = _unit(runtime, "scorer", score_unit,
                ["--receipt", evidence["receipt_binding"]["path"],
                 "--receipt-sha256", evidence["receipt_binding"]["sha256"],
                 "--out-dir", str(score_dir), "--parent-unit-name", parent["unit"]], guard=guard)
            score = runtime.score_result(score_dir)
            _require(score.get("status") == "scored" and type(score.get("success")) is bool
                     and score.get("receipt_sha256") == evidence["receipt_binding"]["sha256"]
                     and score.get("is_policy_episode") is True)
            score_validated = True
        finally:
            score_cost = {"source_index": index, "phase": phase,
                          "wall_seconds": runtime.now() - score_started,
                          "attempted": True, "result_validated": score_validated,
                          "unknown_cost": not score_validated, "cleanup_verified": False}
            report["costs"]["scorers"].append(score_cost)
            if scorer is not None:
                score_cost["cleanup_verified"] = runtime.stop_unit(scorer).get("verified") is True
                _require(score_cost["cleanup_verified"])
        _require(runtime.unit_status(parent).get("main_pid") == 0)
        return {"status": "known", "model_reported_complete": receipt["actor_outcome"]["status"] == "model_reported_complete",
                "official_success": score["success"], "receipt_sha256": evidence["receipt_binding"]["sha256"]}, boundary, generations
    finally:
        if not cost_recorded:
            failed_cost = {"source_index": index, "phase": phase, "unknown_cost": True,
                           "parent_wall_seconds": runtime.now() - started,
                           "counts_complete": False}
            try:
                partial = runtime.artifacts(run_dir)
                failed_cost.update(_costs(partial["receipt"], partial["journal"]))
            except Exception:
                receipt, journal, complete, prefix_bytes = {}, [], False, 0
                try:
                    raw = runtime.read(run_dir / "controller.jsonl")
                    prefix_raw = raw[:raw.rfind(b"\n") + 1]
                    if prefix_raw:
                        journal = _journal(prefix_raw)
                        prefix_bytes = len(prefix_raw)
                    complete = bool(raw) and raw.endswith(b"\n")
                except Exception:
                    pass
                try:
                    parsed = _decode(runtime.read(run_dir / "execution-receipt.json"))
                    if type(parsed) is dict:
                        receipt = parsed
                except Exception:
                    pass
                failed_cost.update(_costs(receipt, journal), journal_complete=complete,
                                   durable_journal_prefix_bytes=prefix_bytes)
            failed_cost.update(unknown_cost=True, counts_complete=False)
            report["costs"]["episodes"].append(failed_cost)
        # Cleanup reads only the first durable worker launch identity, even when
        # an episode failed before producing a receipt or its Python parent died.
        try:
            cleanup = runtime.cleanup_worker(run_dir, registration.TASK_IDS[index], experiment)
            _require(cleanup.get("verified") is True)
        finally:
            if parent is not None:
                _require(runtime.stop_unit(parent).get("verified") is True)
            runtime.record({"event": "episode_lifecycle_end", "source_index": index, "phase": phase,
                            "wall_seconds_including_failure_and_cleanup": runtime.now() - started})


def _release(runtime, job, known_pids):
    started = runtime.now()
    state = runtime.queue("stop", job, known_pids)
    while True:
        _queue_state(state, runtime, stopping=True)
        if state.get("released") is True:
            info = state.get("job")
            _require(type(info) is dict and info.get("status") == "finished"
                     and _finite(info.get("started_epoch")) and _finite(info.get("finished_epoch"))
                     and 0 <= info["finished_epoch"] - info["started_epoch"] <= 7200)
            return {"verified": True, "GPU_seconds": info["finished_epoch"] - info["started_epoch"],
                    "state": state, "release_wall_seconds": runtime.now() - started}
        if state.get("job") is None and runtime.never_started(job, state):
            return {"verified": True, "GPU_seconds": 0, "state": state,
                    "release_wall_seconds": runtime.now() - started}
        _require(runtime.now() - started < RELEASE_SECONDS)
        runtime.wait(PULSE_SECONDS)
        state = runtime.queue("status", job, known_pids)


def run_pilot(plan_sha, root=WORKSPACE, _runtime=None):
    """Run the frozen twelve-source schedule once; never resume or fill slots."""
    _require(_sha(plan_sha))
    runtime = _runtime or _Runtime(root)
    slots = initial_slots()
    report = {"id": registration.ID, "plan_sha256": plan_sha, "status": "unknown", "slots": slots,
              "costs": {"episodes": [], "scorers": [], "paired_CPU_counts": [], "GPU_jobs": []},
              "data_feasibility_gate": {"status": "not_evaluable"}, "candidate_student_target_sha256": [],
              "new_qualified_experiences": 0, "writer_calls": 0, "training_jobs": 0,
              "utility_or_dev_or_test_calls": 0, "automatic_retry": False}
    targets, current, phase, begun = {}, None, None, False
    try:
        plan = runtime.load_plan(plan_sha)
        _require(plan.get("id") == registration.ID and plan.get("status") == "registered"
                 and registration._same(plan.get("task_ids"), list(registration.TASK_IDS))
                 and plan.get("model_arm") == "B3" and plan.get("model_id") == registration.MODEL_ID
                 and plan.get("automatic_retry") is False
                 and all(plan.get(key) is False for key in ("training_authorized", "writer_authorized",
                                                            "utility_authorized", "dev_or_test_authorized")))
        runtime.begin(plan, plan_sha, slots)
        begun = True
        rows = runtime.context_rows(plan)
        _require(type(rows) is list and [row.get("task_id") for row in rows] == list(registration.TASK_IDS))
        for batch_index in range(3):
            if batch_index:
                _require(registration._same(runtime.load_plan(plan_sha), plan))
            runtime.monitor(empty_port=True)
            job = queue.make_job(batch_index, runtime.owner(), plan_sha)
            heartbeat, known_pids, enqueued = None, [], False
            admission_start = runtime.now()
            admission_epoch = runtime.epoch()
            runtime.record({"event": "GPU_enqueue_intent", "job": job})
            try:
                enqueued = True
                state = runtime.queue("enqueue", job)
                _queue_state(state, runtime)
                heartbeat = runtime.heartbeat(job)
                service = None
                while service is None:
                    heartbeat.check()
                    runtime.monitor()
                    _queue_state(state, runtime)
                    _require(state.get("released") is False)
                    info = state.get("job")
                    if type(info) is dict:
                        _require(info.get("status") in {"running", "pending", "waiting", "dispatching"})
                        if info.get("status") == "running":
                            _require(_finite(info.get("started_epoch")))
                            if runtime.epoch() - info["started_epoch"] >= 7200 - SHUTDOWN_RESERVE_SECONDS:
                                raise _BudgetStop("GPU_cap")
                    service = runtime.ready(job, state)
                    if service is None:
                        if runtime.now() - admission_start >= 86400:
                            raise _BudgetStop("admission_timeout")
                        runtime.wait(PULSE_SECONDS)
                        state = runtime.queue("pulse", job)
                known_pids = list(state.get("model_pids", []))
                started_epoch = state["job"]["started_epoch"]
                runtime.record({"event": "GPU_ready", "job": job, "service": service})

                def guard(heartbeat=heartbeat, started_epoch=started_epoch, job=job):
                    heartbeat.check()
                    if runtime.epoch() - started_epoch >= 7200 - SHUTDOWN_RESERVE_SECONDS:
                        raise _BudgetStop("GPU_cap")
                    latest = runtime.queue("status", job)
                    _queue_state(latest, runtime)
                    _require(latest.get("released") is False and type(latest.get("job")) is dict
                             and latest["job"].get("status") == "running")

                for index in range(batch_index * 4, batch_index * 4 + 4):
                    current, phase = slots[index], "first_attempt"
                    guard()
                    initial = {key: rows[index][key] for key in ("metadata_sha256", "messages_sha256", "input_tokens")}
                    current[phase] = {"status": "running"}
                    result, prefix, _generations = _run_episode(runtime, plan, plan_sha, job, service,
                        index, phase, initial, None, guard, report)
                    current[phase] = result
                    if prefix is None:
                        current["pair_class"] = "ineligible_no_boundary"
                        for name in ("control", "review"):
                            current[name] = {"status": "known_ineligible", "reason": "no_eligible_first_4xx_prefix"}
                        runtime.record({"event": "source_known_ineligible", "slot": current})
                        continue
                    prefix_binding = runtime.write(f"prefix-{index:02d}.json", prefix)
                    contexts = {}
                    count_started = runtime.now()
                    count_attempts = 0
                    try:
                        for name in ("control", "review"):
                            guard()
                            messages = actor.build_actor_messages(prefix["metadata"], prefix_records=prefix["prefix_records"],
                                review_hint=plan["design"]["review"]["hint"] if name == "review" else None)
                            count_attempts += 1
                            count = runtime.count(messages)
                            _require(type(count) is int and count > 0)
                            contexts[name] = {"metadata_sha256": _hash(_encode(prefix["metadata"])),
                                              "messages_sha256": _message_payload(messages)[1], "input_tokens": count}
                    finally:
                        report["costs"]["paired_CPU_counts"].append({"source_index": index, "counts": contexts,
                            "wall_seconds": runtime.now() - count_started, "CPU_calls": count_attempts,
                            "counts_complete": len(contexts) == 2, "unknown_cost": len(contexts) != 2})
                    if any(context["input_tokens"] > 15360 for context in contexts.values()):
                        current["pair_class"] = "ineligible_initial_context"
                        for name in ("control", "review"):
                            current[name] = {"status": "known_ineligible", "reason": "paired_initial_context_does_not_fit"}
                        continue
                    review_generations = []
                    for name in (("control", "review") if index % 2 == 0 else ("review", "control")):
                        phase = name
                        guard()
                        current[name] = {"status": "running"}
                        result, _boundary, generations = _run_episode(runtime, plan, plan_sha, job, service,
                            index, name, contexts[name], prefix_binding, guard, report)
                        current[name] = result
                        if name == "review":
                            review_generations = generations
                    control, review = current["control"], current["review"]
                    if not (control["model_reported_complete"] and review["model_reported_complete"]):
                        current["pair_class"] = "ineligible_incomplete_pair"
                    else:
                        values = (control["official_success"], review["official_success"])
                        current["pair_class"] = {(False, True): "help", (True, False): "harm",
                            (True, True): "tie_success", (False, False): "tie_failure"}[values]
                    if current["pair_class"] == "help":
                        for target in student_targets(review_generations, prefix_length=len(prefix["prefix_records"]),
                                                      hint=plan["design"]["review"]["hint"]):
                            targets.setdefault(target["target_sha256"], {"task_id": current["task_id"], **target})
                    runtime.record({"event": "source_finished", "slot": current})
                current, phase = None, None
            finally:
                heartbeat_failure = None
                if heartbeat is not None:
                    try:
                        heartbeat.stop()
                    except BaseException as error:
                        heartbeat_failure = error
                if enqueued:
                    release = {"verified": False, "GPU_seconds": None, "state": {}, "unknown_cost": True}
                    try:
                        release = _release(runtime, job, known_pids)
                    finally:
                        report["costs"]["GPU_jobs"].append({"job": job, "release": release,
                            "admission_to_release_seconds": runtime.now() - admission_start,
                            "admission_wait_seconds": max(0, (release["state"].get("job") or {}).get(
                                "started_epoch", runtime.epoch()) - admission_epoch) if release["verified"] else None,
                            "unknown_cost": not release["verified"]})
                    runtime.record({"event": "GPU_released", "release": release})
                if heartbeat_failure is not None:
                    raise heartbeat_failure
        _require(sum(item["release"]["GPU_seconds"] for item in report["costs"]["GPU_jobs"]) <= 21600)
        report["status"] = "complete"
    except _BudgetStop as error:
        report["status"] = "stopped_" + str(error)
        if current is not None and phase is not None and current[phase]["status"] == "running":
            current[phase] = {"status": "unknown", "reason": "budget_stop_during_episode"}
            current["pair_class"] = "unknown"
    except BaseException:
        if not begun:
            raise NativeSourcePilotError(_ERROR) from None
        report["status"] = "unknown"
        if current is not None:
            current["pair_class"] = "unknown"
            if phase is not None and current[phase]["status"] == "running":
                current[phase] = {"status": "unknown"}
    finally:
        if begun:
            report["candidate_student_target_sha256"] = sorted(targets)
            help_scenarios = {row["task_id"].rsplit("_", 1)[0] for row in slots if row["pair_class"] == "help"}
            complete = report["status"] == "complete" and all(row["first_attempt"]["status"] == "known"
                and all(row[name]["status"] in {"known", "known_ineligible"} for name in ("control", "review"))
                for row in slots)
            report["data_feasibility_gate"] = {"status": (
                "passed" if len(help_scenarios) >= 4 and len(targets) >= 20 else "insufficient")
                if complete else "not_evaluable", "complete_help_scenarios": len(help_scenarios),
                "distinct_student_targets": len(targets), "source_denominator": 12}
            try:
                runtime.write("candidate-student-targets.json", list(targets.values()))
                runtime.record({"event": "study_stop", "report": report})
                runtime.write("study-result.json", report)
            except BaseException:
                raise NativeSourcePilotError(_ERROR) from None
            finally:
                runtime.close()
    return report


def launch_pilot(plan_sha, *, root=WORKSPACE):
    """Launch the coordinator in a bounded user unit that survives SSH exit."""
    runtime = _Runtime(root)
    plan = runtime.load_plan(plan_sha)
    private = Path(plan["private_run_root"])
    fd = episode._private_directory(private)
    try:
        owner = uuid.uuid4().hex
        unit = "wll-native-source-pilot-" + owner + ".service"
        log = private / "pilot-parent.log"
        with _exclusive(fd, log.name):
            pass
        command = unit_command("pilot", unit, ["--run", "--plan-sha256", plan_sha], root=root, log_path=log)
        with _exclusive(fd, "parent-launch-intent.json") as file:
            runtime._write(file, {"unit": unit, "argv": command, "plan_sha256": plan_sha})
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10, check=False)
        _require(result.returncode == 0)
        return {"unit": unit, "plan_sha256": plan_sha, "status": "launch_returned_not_completion"}
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-sha256", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.launch:
            launch_pilot(args.plan_sha256)
            return 0

        def interrupted(_signum, _frame):
            raise NativeSourcePilotError(_ERROR)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        return 0 if run_pilot(args.plan_sha256)["status"] == "complete" else 1
    except BaseException:
        sys.stderr.write(_ERROR + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
