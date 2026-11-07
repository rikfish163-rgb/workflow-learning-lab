"""Bind the native receipt and lifecycle checks to a private authored asset root."""

import argparse
import hashlib
import os
import re
import socket
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import native_appworld_worker as worker
from scripts import native_appworld_worker_client as client
from scripts.native_source_registration import public_read
from scripts.native_state_recovery_authored_panel_001 import PANEL
from scripts.probe_native_state_recovery_authored_panel_001 import _input, _input_hash, evaluate
from scripts.probe_native_state_recovery_authored_world_001 import ASSETS, SCRATCH

SOURCE = {
    "native_appworld_worker_client": "16eac615ae5d9f2e143bb86b43ade4435ad0114e3bb626c3a1c90533e7d978dc",
    "native_appworld_worker": "8fe81c0e4cd8b95f0486cd2a711d3b0d6ae2004581d14468daef93079ebfc421",
    "native_appworld_scorer": "8fa2b73909168d8a8edc94bd8df9915578a9c602b8e3e85973e2f202ffe5c4c0",
}
API_SCHEMA_SHA256 = "ed5f53e26143142cc0dac5b1bf77801b499698e2141d7ed169e4a1ad7433b193"


def _require(condition):
    if not condition:
        raise ValueError("native_authored_runtime_binding_invalid")


def _source(name):
    relative = f"scripts/{name}.py"
    raw = public_read(client.WORKSPACE, relative)
    _require(hashlib.sha256(raw).hexdigest() == SOURCE[name])
    return raw.decode()


def _module(name, text):
    module = ModuleType(f"scripts._authored_{name}")
    module.__file__ = str(client.WORKSPACE / f"scripts/{name}.py")
    # Only pinned repository source and trusted bootstrap arguments are compiled.
    exec(compile(text, module.__file__, "exec", dont_inherit=True), vars(module))
    return module


def _verify_api_schema(root):
    directory = root / "data/api_docs/standard"
    fd = client._open_directory(directory)
    try:
        entries = []
        for name in sorted(os.listdir(fd)):
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=fd)
            with os.fdopen(child, "rb") as file:
                info = os.fstat(file.fileno())
                _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                         and info.st_nlink == 1 and 0 < info.st_size <= 2_000_000)
                data = file.read(info.st_size + 1)
                _require(len(data) == info.st_size)
                entries.append((name, hashlib.sha256(data).hexdigest()))
        _require(hashlib.sha256(client._encode(entries)).hexdigest() == API_SCHEMA_SHA256)
    finally:
        os.close(fd)


def verify_assets(root, expected_sha):
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and type(expected_sha) is str and re.fullmatch(r"[a-f0-9]{64}", expected_sha) is not None)
    root = Path(root)
    _require(root.parent == SCRATCH and root.name.startswith("authored-panel-"))
    fd = client._private_tree(root)
    os.close(fd)
    for path in (root / "deps", root / "data", root / "data/tasks", root / "data/base_dbs"):
        fd = client._private_tree(path)
        os.close(fd)
    task_ids = {path.name for path in (root / "data/tasks").iterdir()}
    _require(task_ids and task_ids <= PANEL.keys() and not any((root / "deps").iterdir()))
    for parent in (root / "data/tasks", root / "data/base_dbs"):
        _require(all(not item.is_symlink() for item in parent.rglob("*")))
    _require(_input_hash(root) == expected_sha)
    _verify_api_schema(root)
    return root


def _client(root, expected_sha):
    root = verify_assets(root, expected_sha)
    name = "native_appworld_worker_client"
    text = _source(name)
    anchor = '"from scripts.native_appworld_worker import main;raise SystemExit(main())"'
    bootstrap = (
        "from scripts.native_state_recovery_authored_runtime_001 import worker_main;"
        f"raise SystemExit(worker_main({str(root)!r},{expected_sha!r}))"
    )
    _require(text.count(anchor) == 1)
    core = _module(name, text.replace(anchor, repr(bootstrap), 1))
    core.ASSET_ROOT = root
    return core


def worker_factory(root, expected_sha):
    return _client(root, expected_sha).NativeWorkerClient


def worker_main(root, expected_sha):
    root = verify_assets(root, expected_sha)
    _source("native_appworld_worker")
    _require(Path(worker.__file__) == client.WORKSPACE / "scripts/native_appworld_worker.py")
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / "cache"))
    sys.path.insert(0, str(ASSETS / "deps"))
    worker.ASSET_ROOT = root  # the isolated worker process has one immutable asset binding
    return worker.main()


def parent_unit(root, experiment):
    digest = hashlib.sha256(client._encode([str(root), experiment])).hexdigest()[:32]
    return f"wll-authored-parent-{digest}.service"


def _unchanged_other_apps(root, task, snapshot):
    base_apps = {path.stem for path in (root / "data/base_dbs").glob("*.db")}
    initial = _input(root, task) / "dbs"
    # Native AppWorld restores SQL deltas from the same empty base. Exact
    # equality also rejects an out-of-scope write that is later reverted.
    return all((initial / f"{app}.jsonl").read_bytes() == (snapshot / f"{app}.jsonl").read_bytes()
               for app in base_apps - {task.app, "supervisor"})


def score(root, expected_sha, receipt, receipt_sha, output):
    core_client = _client(root, expected_sha)
    root = core_client.ASSET_ROOT
    receipt = Path(receipt)
    experiment = receipt.parent.name
    _require(receipt.parent.parent == root / "runs"
             and client._EXPERIMENT.fullmatch(experiment) is not None
             and Path(output).parent == root / "scores")
    scorer = _module("native_appworld_scorer", _source("native_appworld_scorer"))
    scorer.client = core_client

    def authored_evaluate(*, task_id, experiment_name, suppress_errors, save_report):
        _require(experiment_name == experiment and task_id in PANEL
                 and suppress_errors is True and save_report is False)
        verify_assets(root, expected_sha)
        task = PANEL[task_id]
        snapshot = root / "experiments/outputs" / experiment / "tasks" / task_id / "dbs"
        unchanged = _unchanged_other_apps(root, task, snapshot)
        passed, _completion = evaluate(root, task, snapshot)
        verify_assets(root, expected_sha)
        success = unchanged and passed
        return SimpleNamespace(total_count=int(success), num_tests=1, success=success)

    return scorer.score_saved_task(
        str(receipt), receipt_sha, str(output), parent_unit_name=parent_unit(root, experiment),
        _evaluate=authored_evaluate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = score(args.asset_root, args.initial_sha256, args.receipt, args.receipt_sha256, args.out_dir)
    return 0 if result["status"] == "scored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
