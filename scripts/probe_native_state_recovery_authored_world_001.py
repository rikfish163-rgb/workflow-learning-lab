"""Exercise an authored AppWorld task and an independent scorer on amax CPU."""

import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import native_appworld_worker as native

ASSETS = native.ASSET_ROOT
SCRATCH = native.PRIVATE_STATE_ROOT / "runtime-tmp"
TASK_ID = hashlib.sha256(b"wll-independent-note-cpu-probe-001").hexdigest()[:7] + "_1"
SUPERVISOR = {"first_name": "Demo", "last_name": "Author", "email": "fixture@example.org",
              "phone_number": "+12025550142"}
PASSWORD = "synthetic-appworld-probe-only"
INSTRUCTION = "Pin the note titled Authored target. Leave the other note unchanged."
TARGET = ("Authored target", "Target content")
GUARD = ("Authored guard", "Guard content")
ATTEMPTS = {"correct": 6, "no_change": 5, "collateral_change": 7}


def _require(condition):
    if not condition:
        raise ValueError("authored_world_probe_invariant")


def _configure_appworld(root):
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / "cache"))
    sys.path.insert(0, str(ASSETS / "deps"))


def _prepare(root):
    _require(not (ASSETS / "data/tasks" / TASK_ID).exists())
    _configure_appworld(root)
    from appworld.apps import get_all_apps
    from appworld.collections.models import ModelCollection
    from appworld.common.constants import DB_VERSION

    task = root / "data/tasks" / TASK_ID
    dbs = task / "dbs"
    dbs.mkdir(parents=True)
    base = root / "data/base_dbs"
    base.mkdir()
    (root / "deps").mkdir()
    shutil.copytree(ASSETS / "data/api_docs/standard", root / "data/api_docs/standard")
    (task / "specs.json").write_text(json.dumps({
        "supervisor": SUPERVISOR, "instruction": INSTRUCTION,
        "datetime": "2025-01-01T12:00:00", "db_version": DB_VERSION,
    }))
    apps = set(get_all_apps(skip_admin=True)) | {"admin", "supervisor"}
    models = ModelCollection.load(from_db_home_path=str(base), load_apps=sorted(apps), create_db=True)
    try:
        supervisor = models.supervisor.Supervisor.create_save(**SUPERVISOR, birthday="1990-01-01")
        models.supervisor.AccountPassword.create_save(
            supervisor_id=supervisor.id, account_name="simple_note", password=PASSWORD,
        )
        models.supervisor.Task.create_save(supervisor_id=supervisor.id, instruction=INSTRUCTION)
        identity = {key: SUPERVISOR[key] for key in ("first_name", "last_name", "email")}
        user = models.simple_note.User.create_save(**identity, password=PASSWORD, verified=True)
        for title, content in (TARGET, GUARD):
            models.simple_note.Note.create_save(
                user_id=user.id, title=title, content=content, pinned=False,
            )
    finally:
        models.close()
    _require({path.stem for path in base.glob("*.db")} == apps)
    for app in sorted(apps):
        (dbs / f"{app}.jsonl").touch()
    return len(apps)


def _snapshot(root, case):
    return root / "experiments/outputs" / case / "tasks" / TASK_ID / "dbs"


def _act(root, case):
    _configure_appworld(root)
    run_dir = root / f"run-{case}"
    run_dir.mkdir(mode=0o700)
    with patch.object(native, "ASSET_ROOT", root):
        worker = native.NativeAppWorldWorker(
            TASK_ID, case, run_dir, max_actions=8, max_api_requests=12,
        )
        try:
            def rpc(op, **kwargs):
                reply = worker.handle_request(json.dumps({"op": op, **kwargs}))
                _require(reply["ok"] is True)
                return reply["result"]

            def call(name, **arguments):
                return rpc("dispatch", action=json.dumps({"name": name, "arguments": arguments}))

            metadata = rpc("metadata")
            _require(metadata["instruction"] == INSTRUCTION)
            credentials = call("supervisor__show_account_passwords")
            password, = (r["password"] for r in credentials if r["account_name"] == "simple_note")
            token = call("simple_note__login", username=metadata["supervisor"]["email"],
                         password=password)["access_token"]
            target = call("simple_note__show_note", note_id=1, access_token=token)
            _require(target["title"] == TARGET[0] and target["pinned"] is False)
            if case != "no_change":
                call("simple_note__update_note", note_id=1, pinned=True, access_token=token)
            if case == "collateral_change":
                call("simple_note__update_note", note_id=2, pinned=True, access_token=token)
            final = call("simple_note__show_note", note_id=1, access_token=token)
            _require(final["pinned"] is (case != "no_change"))
            call("supervisor__complete_task")
            _require(rpc("save")["db_path"] == str(_snapshot(root, case)))
        finally:
            worker.close()


def _score(root, case):
    receipt = json.loads((root / f"run-{case}" / "status.json").read_text())
    _require(receipt["state"] == "closed" and receipt["score_eligible"] is True
             and receipt["last_successful_save_through_attempt"] == ATTEMPTS[case]
             and receipt["cost"] == dict.fromkeys(
                 ("action_attempts", "api_invocations", "upstream_requests"), ATTEMPTS[case],
             ))
    _configure_appworld(root)
    from appworld.collections.models import ModelCollection

    models = ModelCollection.load(
        from_db_home_path=str(_snapshot(root, case)), to_db_home_path=":memory:authored_verify",
        load_apps=["simple_note", "supervisor"], skip_mandatory_apps=True,
    )
    try:
        task, = models.supervisor.Task.all()
        _require(task.instruction == INSTRUCTION and task.status == "success")
        notes = models.simple_note.Note.all()
        return sorted((n.id, n.user_id, n.title, n.content, n.pinned, n.tags) for n in notes) == [
            (1, 1, *TARGET, True, []), (2, 1, *GUARD, False, []),
        ]
    finally:
        models.close()


def _child(role, root, case):
    completed = subprocess.run(
        [sys.executable, "-B", "-m", __spec__.name, role, str(root), case],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=90,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"authored_world_probe_{role}_{case}_failed")
    return completed.stdout


def report():
    with tempfile.TemporaryDirectory(prefix="authored-probe-", dir=SCRATCH) as directory:
        root = Path(directory)
        schema_count = _prepare(root)
        outcomes = {}
        for case in ATTEMPTS:
            _require(_child("actor", root, case) == "")
            outcomes[case] = json.loads(_child("score", root, case))
        _require(outcomes == {"correct": True, "no_change": False, "collateral_change": False})
    return {
        "scope": "authored_cpu_pipeline_only", "synthetic_task_count": 1,
        "schema_count": schema_count, "fresh_actor_processes": len(outcomes),
        "post_exit_scorer_processes": len(outcomes), "scripted_api_calls": sum(ATTEMPTS.values()),
        "scripted_self_reported_completions": len(outcomes),
        "terminal_predicate": outcomes, "official_task_bodies_opened": 0,
        "official_scores": 0, "model_calls": 0, "gpu_jobs": 0,
        "study_registered": False,
    }


def main():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and SCRATCH.resolve(strict=True) == SCRATCH)
    os.umask(0o077)
    if len(sys.argv) == 1:
        print(json.dumps(report(), sort_keys=True))
        return
    role, directory, case = sys.argv[1:]
    root = Path(directory)
    info = root.stat(follow_symlinks=False)
    _require(root.parent == SCRATCH and root.name.startswith("authored-probe-")
             and stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
             and info.st_uid == os.getuid() and case in ATTEMPTS and role in ("actor", "score"))
    if role == "actor":
        _act(root, case)
    else:
        print(json.dumps(_score(root, case)))


if __name__ == "__main__":
    main()
