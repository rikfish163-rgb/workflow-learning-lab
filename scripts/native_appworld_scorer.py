"""One-shot private scorer for a closed native AppWorld execution.

Run this module in its own trusted CPU process after the actor parent has
terminated. The source-bound study launcher, not the policy, must supply the
exact receipt digest and the actual parent unit name and enforce this process's
CPU, memory, wall-time and network limits. Neither scores nor this API belong
in a policy callback, experience writer, training set, or retrieval index.
"""

import argparse
import contextlib
import hashlib
import io
import os
import re
import socket
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scripts import native_appworld_worker_client as client


class NativeScorerError(RuntimeError):
    """Content-free scorer or private persistence failure."""


_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,190}\.service", re.ASCII)
_WORKER = re.compile(r"wll-native-worker-([0-9a-f]{16})-([0-9a-f]{32})\.service", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MAX_DOCUMENT = 64 * 1024 * 1024
_MAX_LINE = 2 * client.MAX_RPC_BYTES + 1024
_MAX_FILES = 4096
_MAX_DB_BYTES = 8 * 1024 * 1024 * 1024
_HASH_SECONDS = 180
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_IDENTITY = ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size",
             "st_mtime_ns", "st_ctime_ns")


def _fail():
    raise NativeScorerError("Native scoring validation or persistence failed.") from None


def _same(left, right):
    return client._same(left, right)


def _main_pid(unit):
    """A fresh systemd query; unknown, malformed and failed queries reject."""
    result = subprocess.run(
        [client.SYSTEMCTL, "--user", "show", unit, "--property=MainPID"],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False,
    )
    if result.returncode != 0 or len(result.stdout) > 256:
        _fail()
    match = re.fullmatch(rb"MainPID=([0-9]+)\n", result.stdout)
    if match is None:
        _fail()
    return int(match.group(1))


def _require_stopped(parent, worker):
    if (type(parent) is not str or _UNIT.fullmatch(parent) is None
            or type(worker) is not str or _WORKER.fullmatch(worker) is None or parent == worker):
        _fail()
    for unit in (parent, worker):
        pid = _main_pid(unit)
        if type(pid) is not int or pid != 0:
            _fail()


def _stable_read(directory, name, limit):
    fd = os.open(name, _READ_FLAGS, dir_fd=directory)
    try:
        before = client._private(fd, directory=False)
        if before.st_size > limit:
            _fail()
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (len(data) > limit or len(data) != before.st_size
                or any(getattr(before, key) != getattr(after, key)
                       or getattr(after, key) != getattr(named, key) for key in _IDENTITY)):
            _fail()
        return data
    finally:
        os.close(fd)


def _document(data):
    if not data.endswith(b"\n"):
        _fail()
    return client._decode(data)


def _journal(data):
    if not data or not data.endswith(b"\n"):
        _fail()
    lines = data.splitlines(keepends=True)
    if len(lines) > 100000:
        _fail()
    records = []
    for line in lines:
        if len(line) >= _MAX_LINE or not line.endswith(b"\n"):
            _fail()
        value = client._decode(line)
        if type(value) is not dict or type(value.get("event")) is not str:
            _fail()
        records.append(value)
    return records


def _nonnegative_cost(value):
    if (type(value) is not dict or set(value) != client._COST_KEYS
            or any(type(item) is not int or item < 0 for item in value.values())
            or value["api_invocations"] > value["action_attempts"]):
        _fail()


def _read_evidence(run_fd, worker_fd):
    return {
        "receipt": _stable_read(run_fd, "execution-receipt.json", client.MAX_RPC_BYTES),
        "controller": _stable_read(run_fd, "controller.jsonl", _MAX_DOCUMENT),
        "status": _stable_read(worker_fd, "status.json", client.MAX_RPC_BYTES),
        "actions": _stable_read(worker_fd, "actions.jsonl", _MAX_DOCUMENT),
        "transport": _stable_read(worker_fd, "transport.jsonl", _MAX_DOCUMENT),
    }


def _same_private_directory(fd, path):
    current = client._private_tree(path)
    try:
        left, right = os.fstat(fd), os.fstat(current)
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            _fail()
    finally:
        os.close(current)


def _one(records, event):
    matches = [(index, record) for index, record in enumerate(records) if record["event"] == event]
    if len(matches) != 1:
        _fail()
    return matches[0]


def _validate_receipt_and_logs(evidence, run_path, expected_sha):
    if hashlib.sha256(evidence["receipt"]).hexdigest() != expected_sha:
        _fail()
    receipt = _document(evidence["receipt"])
    if type(receipt) is not dict:
        _fail()
    task, experiment = receipt.get("task_id"), receipt.get("experiment_name")
    outcome, manifest, profile = (receipt.get("actor_outcome"), receipt.get("snapshot"),
                                   receipt.get("worker_profile"))
    if (type(task) is not str or not client._TASK_ID.fullmatch(task)
            or type(experiment) is not str or not client._EXPERIMENT.fullmatch(experiment)
            or receipt.get("status") != "saved_and_closed"
            or type(receipt.get("is_policy_episode")) is not bool
            or type(receipt.get("worker_exit_code")) is not int or receipt["worker_exit_code"] != 0
            or type(outcome) is not dict or outcome.get("status") not in ("incomplete", "model_reported_complete")
            or outcome.get("outcome_unknown") is not False or outcome.get("unknown_cost") is not False
            or type(outcome.get("stop_reason")) is not str
            or type(manifest) is not dict or type(profile) is not dict):
        _fail()
    cost = manifest.get("cost")
    _nonnegative_cost(cost)
    actor_cost = outcome.get("cost")
    required_actor_cost = client._COST_KEYS | {"generations", "prompt_tokens", "completion_tokens"}
    if (type(actor_cost) is not dict or not required_actor_cost <= actor_cost.keys()
            or any(type(value) is not int or value < 0 for value in actor_cost.values())
            or not _same({key: actor_cost[key] for key in client._COST_KEYS}, cost)):
        _fail()
    unit = manifest.get("unit_name")
    match = _WORKER.fullmatch(unit) if type(unit) is str else None
    expected_unit_digest = hashlib.sha256(
        (task + experiment + str(run_path / "worker")).encode("utf-8")).hexdigest()[:16]
    db_path = client.ASSET_ROOT / "experiments" / "outputs" / experiment / "tasks" / task / "dbs"
    if (match is None or match.group(1) != expected_unit_digest
            or profile.get("unit_name") != unit or manifest.get("db_path") != str(db_path)
            or set(manifest) != {"unit_name", "db_path", "files", "cost", "manifest_sha256"}
            or type(manifest.get("manifest_sha256")) is not str
            or not _HASH.fullmatch(manifest["manifest_sha256"])):
        _fail()
    bare = {key: manifest[key] for key in ("unit_name", "db_path", "files", "cost")}
    if hashlib.sha256(client._encode(bare)).hexdigest() != manifest["manifest_sha256"]:
        _fail()
    files = manifest["files"]
    if type(files) is not list or not 0 < len(files) <= _MAX_FILES:
        _fail()
    for item in files:
        if (type(item) is not dict or set(item) != {"path", "size", "sha256"}
                or type(item["path"]) is not str or not item["path"]
                or item["path"].startswith("/") or "\\" in item["path"]
                or any(part in ("", ".", "..") for part in item["path"].split("/"))
                or type(item["size"]) is not int or item["size"] < 0
                or type(item["sha256"]) is not str or not _HASH.fullmatch(item["sha256"])):
            _fail()
    if [item["path"] for item in files] != sorted(set(item["path"] for item in files)):
        _fail()
    if sum(item["size"] for item in files) > _MAX_DB_BYTES:
        _fail()

    status = _document(evidence["status"])
    if type(status) is not dict:
        _fail()
    max_actions, max_api = status.get("max_actions"), status.get("max_api_requests")
    if any(type(value) is not int or value < 1 for value in (max_actions, max_api)):
        _fail()
    expected_status = {
        "state": "closed", "task_id": task, "experiment_name": experiment,
        "max_actions": max_actions, "max_api_requests": max_api, "cost": cost,
        "last_successful_save_through_attempt": cost["action_attempts"],
        "current_state_saved": True, "score_eligible": True,
        "outcome_unknown": False, "unknown_cost": False,
    }
    if (not _same(status, expected_status) or cost["action_attempts"] > max_actions
            or cost["api_invocations"] > max_api or cost["upstream_requests"] > max_api):
        _fail()

    transport = _journal(evidence["transport"])
    if len(transport) < 8 or transport[0].get("event") != "launch_intent" or transport[0].get("unit") != unit:
        _fail()
    if (transport[1].get("event") != "launch_result" or type(transport[1].get("pid")) is not int
            or transport[1]["pid"] < 1
            or not _same(transport[-2], {"event": "exit_result", "returncode": 0})
            or not _same(transport[-1], {"event": "snapshot_result", "manifest": manifest})):
        _fail()
    # Reapply the client's exact reply/freshness/cost contract to the durable transcript.
    verifier = object.__new__(client.NativeWorkerClient)
    verifier._task_id, verifier._experiment_name = task, experiment
    verifier._max_actions, verifier._max_api_requests = max_actions, max_api
    verifier._cost = dict.fromkeys(client._COST_KEYS, 0)
    verifier._freshness = dict.fromkeys(client._FRESH_KEYS, False)
    verifier._freshness["last_successful_save_through_attempt"] = None
    expected_actions, operations, replies = [], [], {}
    middle = transport[2:-2]
    if len(middle) % 2:
        _fail()
    for index in range(0, len(middle), 2):
        intent, result = middle[index:index + 2]
        if set(intent) != {"event", "request"} or intent["event"] != "request_intent":
            _fail()
        if set(result) != {"event", "reply"} or result["event"] != "request_result":
            _fail()
        request, reply = intent["request"], result["reply"]
        if type(request) is not dict or type(request.get("op")) is not str:
            _fail()
        op = request["op"]
        if (op not in ("metadata", "dispatch", "save", "close")
                or set(request) != ({"op", "action"} if op == "dispatch" else {"op"})
                or (op == "dispatch" and type(request["action"]) is not str)):
            _fail()
        if (op == "metadata" and
                ((operations and operations[-1] != "metadata")
                 or ("metadata" in replies and not _same(replies["metadata"], reply)))):
            _fail()
        verifier._validate_reply(reply, op)
        verifier._cost, verifier._freshness = reply["cost"].copy(), reply["freshness"].copy()
        operations.append(op)
        if op == "dispatch":
            attempt = verifier._cost["action_attempts"]
            expected_actions.extend([{"event": "attempt", "attempt": attempt, "action": request["action"]},
                                     {"event": "result", "attempt": attempt, "reply": reply}])
        elif op == "save":
            expected_actions.extend([{"event": "save_attempt"},
                                     {"event": "save_result", "success": True,
                                      "saved_through_attempt": verifier._cost["action_attempts"]}])
        if op != "dispatch":
            replies.setdefault(op, reply)
    if (len(operations) < 3 or operations[0] != "metadata" or operations[-2:] != ["save", "close"]
            or any(op not in ("metadata", "dispatch") for op in operations[1:-2])
            or not _same(verifier._cost, cost) or not _same(_journal(evidence["actions"]), expected_actions)
            or not expected_actions or expected_actions[-1] != {
                "event": "save_result", "success": True,
                "saved_through_attempt": cost["action_attempts"]}):
        _fail()

    controller = _journal(evidence["controller"])
    events = ("controller_intent", "metadata_intent", "metadata_result", "worker_effective_profile",
              "actor_result", "save_intent", "save_result", "close_intent", "close_result",
              "execution_snapshot", "controller_stop")
    selected = [_one(controller, event) for event in events]
    indexes, rows = zip(*selected, strict=True)
    if list(indexes) != sorted(indexes) or indexes[0] != 0 or indexes[-1] != len(controller) - 1:
        _fail()
    intent = rows[0]
    if (intent.get("task_id") != task or intent.get("experiment_name") != experiment
            or intent.get("is_policy_episode") is not receipt["is_policy_episode"]
            or type(intent.get("pid")) is not int or intent["pid"] < 1
            or type(intent.get("boot_id")) is not str or not intent["boot_id"]
            or rows[1].get("unit_name") != unit or not _same(rows[2].get("reply"), replies["metadata"])
            or not _same(rows[3].get("profile"), profile) or not _same(rows[4].get("outcome"), outcome)
            or not _same(rows[6].get("reply"), replies["save"])
            or not _same(rows[8].get("reply"), replies["close"])
            or not _same(rows[9].get("snapshot"), manifest)
            or not _same(rows[10].get("result"), receipt)
            or any(row["event"] == "controller_failure" for row in controller)):
        _fail()
    return receipt, manifest, db_path


def _owned_db_directory(path):
    root = client.PRIVATE_STATE_ROOT
    if root not in path.parents or str(path) != os.fspath(path) or ".." in path.parts:
        _fail()
    fd = client._open_directory(root)
    try:
        client._private(fd, directory=True)
        for part in path.relative_to(root).parts:
            child = os.open(part, client._DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or not stat.S_ISDIR(info.st_mode):
                _fail()
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_db(path, expected):
    deadline = time.monotonic() + _HASH_SECONDS
    found, total, directories = [], 0, 0

    def names(fd):
        result = []
        with os.scandir(fd) as entries:
            for entry in entries:
                result.append(entry.name)
                if len(result) > _MAX_FILES:
                    _fail()
        return sorted(result)

    def scan(fd, prefix, depth):
        nonlocal total, directories
        directories += 1
        if directories > _MAX_FILES:
            _fail()
        if depth > 16 or time.monotonic() > deadline:
            _fail()
        original = names(fd)
        for name in original:
            if time.monotonic() > deadline:
                _fail()
            name.encode("utf-8", "strict")
            relative = prefix + name
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(before.st_mode):
                child = os.open(name, client._DIR_FLAGS, dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    if opened.st_uid != os.geteuid() or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                        _fail()
                    scan(child, relative + "/", depth + 1)
                    named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
                        _fail()
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                child = os.open(name, _READ_FLAGS, dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid()
                            or opened.st_nlink != 1 or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)):
                        _fail()
                    digest, size = hashlib.sha256(), 0
                    while True:
                        if time.monotonic() > deadline:
                            _fail()
                        chunk = os.read(child, 1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        total += len(chunk)
                        if total > _MAX_DB_BYTES:
                            _fail()
                        digest.update(chunk)
                    after = os.fstat(child)
                    named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if (size != opened.st_size or any(
                            getattr(opened, key) != getattr(after, key)
                            or getattr(after, key) != getattr(named, key) for key in _IDENTITY)):
                        _fail()
                    found.append({"path": relative, "size": size, "sha256": digest.hexdigest()})
                    if len(found) > _MAX_FILES:
                        _fail()
                finally:
                    os.close(child)
            else:
                _fail()
        if original != names(fd):
            _fail()

    fd = _owned_db_directory(path)
    try:
        scan(fd, "", 0)
        current = _owned_db_directory(path)
        try:
            a, b = os.fstat(fd), os.fstat(current)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                _fail()
        finally:
            os.close(current)
    finally:
        os.close(fd)
    found.sort(key=lambda item: item["path"])
    if not _same(found, expected):
        _fail()


def _official_evaluate(**kwargs):
    # Imports and ground-truth loading happen only after all lifecycle checks.
    if any(name == "appworld" or name.startswith("appworld.") for name in sys.modules):
        _fail()
    deps = client.ASSET_ROOT / "deps"
    fd = _owned_db_directory(deps)
    os.close(fd)
    sys.path.insert(0, str(deps))
    os.environ["APPWORLD_ROOT"] = str(client.ASSET_ROOT)
    from appworld.apps.lib.models.db import CachedDBHandler, ModelHashHandler
    from appworld.common.path_store import path_store
    from appworld.evaluator import evaluate_task

    path_store.update_root(str(client.ASSET_ROOT))
    try:
        return evaluate_task(**kwargs)
    finally:
        cleanup_error = None
        for cleanup in (CachedDBHandler.reset, ModelHashHandler.reset):
            try:
                cleanup()
            except BaseException as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


def _write(file, value):
    data = client._encode(value) + b"\n"
    view = memoryview(data)
    while view:
        written = file.write(view)
        if not written:
            _fail()
        view = view[written:]
    file.flush()
    os.fsync(file.fileno())


def score_saved_task(receipt_path, expected_receipt_sha256, out_dir, *, parent_unit_name, _evaluate=None):
    """Return a private scored/unknown record; never return evaluator details.

    The caller must keep the return value and all files private. A fresh output
    directory is required and no failed evaluation is automatically retried.
    Input/lifecycle/evaluator failures become unknown; private persistence
    failures raise the content-free NativeScorerError.
    """
    files, run_fd, worker_fd, out_fd = [], None, None, None
    try:
        if socket.gethostname() != "amax" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            _fail()
        raw = os.fspath(receipt_path)
        receipt_path = Path(raw)
        if (raw != str(receipt_path) or receipt_path.name != "execution-receipt.json"
                or type(expected_receipt_sha256) is not str or not _HASH.fullmatch(expected_receipt_sha256)):
            _fail()
        out_path = Path(out_dir)
        if os.fspath(out_dir) != str(out_path):
            _fail()
        out_fd = client._private_tree(out_path)
        journal = client._exclusive(out_fd, "scoring-journal.jsonl")
        files.append(journal)
        runtime = io.TextIOWrapper(client._exclusive(out_fd, "scoring-runtime.log"),
                                   encoding="utf-8", write_through=True)
        files.append(runtime)
        result_file = client._exclusive(out_fd, "scoring-result.json")
        files.append(result_file)
        os.fsync(out_fd)
        _write(journal, {"event": "scoring_intent", "receipt_sha256": expected_receipt_sha256,
                         "parent_unit_name": parent_unit_name})
        result = {"status": "unknown", "success": None, "receipt_sha256": expected_receipt_sha256}
        try:
            run_path = receipt_path.parent
            run_fd = client._private_tree(run_path)
            worker_fd = client._private_tree(run_path / "worker")
            evidence = _read_evidence(run_fd, worker_fd)
            receipt, manifest, db_path = _validate_receipt_and_logs(evidence, run_path, expected_receipt_sha256)
            unit = manifest["unit_name"]
            _require_stopped(parent_unit_name, unit)
            _verify_db(db_path, manifest["files"])
            _write(journal, {"event": "validated_before_evaluation", "manifest_sha256": manifest["manifest_sha256"]})
            _same_private_directory(run_fd, run_path)
            _same_private_directory(worker_fd, run_path / "worker")
            _same_private_directory(out_fd, out_path)
            if _read_evidence(run_fd, worker_fd) != evidence:
                _fail()
            _require_stopped(parent_unit_name, unit)
            tracker, failed = None, False
            try:
                with contextlib.redirect_stdout(runtime), contextlib.redirect_stderr(runtime):
                    tracker = (_evaluate or _official_evaluate)(
                        task_id=receipt["task_id"], experiment_name=receipt["experiment_name"],
                        suppress_errors=True, save_report=False)
            except BaseException:
                failed = True
                traceback.print_exc(file=runtime)
            finally:
                runtime.flush()
                os.fsync(runtime.fileno())
                _verify_db(db_path, manifest["files"])
                _same_private_directory(run_fd, run_path)
                _same_private_directory(worker_fd, run_path / "worker")
                _same_private_directory(out_fd, out_path)
                if _read_evidence(run_fd, worker_fd) != evidence:
                    _fail()
                _require_stopped(parent_unit_name, unit)
            if failed:
                _fail()
            total, expected = tracker.total_count, tracker.num_tests
            success = tracker.success
            if (type(total) is not int or type(expected) is not int or not 0 <= total <= expected
                    or type(success) is not bool or (success and total != expected)):
                _fail()
            result = {"status": "scored", "success": success, "receipt_sha256": expected_receipt_sha256,
                      "manifest_sha256": manifest["manifest_sha256"],
                      "is_policy_episode": receipt["is_policy_episode"]}
        except BaseException:
            traceback.print_exc(file=runtime)
            runtime.flush()
            os.fsync(runtime.fileno())
        _write(journal, {"event": "scoring_stop", "status": result["status"]})
        _write(result_file, result)
        os.fsync(out_fd)
        return result
    except BaseException:
        raise NativeScorerError("Native scorer failed; inspect private state.") from None
    finally:
        close_failed = False
        for file in files:
            try:
                file.close()
            except BaseException:
                close_failed = True
        for fd in (worker_fd, run_fd, out_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException:
                    close_failed = True
        if close_failed:
            raise NativeScorerError("Native scorer failed; inspect private state.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parent-unit-name", required=True)
    args = parser.parse_args(argv)
    try:
        result = score_saved_task(args.receipt, args.receipt_sha256, args.out_dir,
                                  parent_unit_name=args.parent_unit_name)
        return 0 if result["status"] == "scored" else 1
    except NativeScorerError:
        sys.stderr.write("Native scorer failed; inspect private state.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
