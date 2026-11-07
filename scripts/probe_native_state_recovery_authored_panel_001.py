"""Run authored positive and negative cases through AppWorld on remote CPU."""

import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from scripts import native_appworld_worker as native
from scripts.native_state_recovery_authored_panel_001 import PANEL
from scripts.probe_native_state_recovery_authored_world_001 import (
    ASSETS,
    SCRATCH,
    SUPERVISOR,
    _configure_appworld,
)

PASSWORD = "synthetic-authored-panel-only"
BRANCHES = {"correct": True, "wrong": False, "collateral": False}
APPS = {
    "simple_note": ("Note", "search_notes", "show_note", "note_id"),
    "todoist": ("Task", "show_tasks", "show_task", "task_id"),
    "phone": ("Alarm", "show_alarms", "show_alarm", "alarm_id"),
    "gmail": ("UserEmailThread", "show_inbox_threads", "show_thread", "email_thread_id"),
}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _input(root, task):
    return root / "data/tasks" / task.task_id


def _output(root, task, branch):
    return root / "experiments/outputs" / branch / "tasks" / task.task_id / "dbs"


def _seed(models, task):
    supervisor = models.supervisor.Supervisor.create_save(**SUPERVISOR, birthday="1990-01-01")
    models.supervisor.AccountPassword.create_save(
        supervisor_id=supervisor.id, account_name=task.app, password=PASSWORD,
    )
    models.supervisor.Task.create_save(supervisor_id=supervisor.id, instruction=task.instruction)
    app = models[task.app]
    identity = {key: SUPERVISOR[key] for key in ("first_name", "last_name")}
    if task.app == "phone":
        user = app.User.create_save(**identity, phone_number=SUPERVISOR["phone_number"], password=PASSWORD)
    elif task.app == "gmail":
        user = app.User.create_save(**identity, email=SUPERVISOR["email"], password=PASSWORD)
    else:
        user = app.User.create_save(**identity, email=SUPERVISOR["email"], password=PASSWORD, verified=True)
    if task.app == "todoist":
        project = app.Project.create_save(user_id=user.id, name="Inbox", is_inbox=True)
        app.ProjectCollaboratorLink.create_save(project_id=project.id, user_id=user.id)
    elif task.app == "gmail":
        sender = app.User.create_save(first_name="Panel", last_name="Sender",
                                      email="sender@example.org", password=PASSWORD)
    for index, (label, initial) in enumerate(zip(task.labels, task.initial, strict=True)):
        if task.app == "simple_note":
            fields = {"title": label, "content": f"Original text {index}", "pinned": False}
            app.Note.create_save(user_id=user.id, **(fields | {task.field: initial}))
        elif task.app == "todoist":
            fields = {"title": label, "description": f"Original text {index}", "is_completed": False}
            app.Task.create_save(user_id=user.id, project_id=project.id, priority="medium",
                                 order_index=index, **(fields | {task.field: initial}))
        elif task.app == "phone":
            app.Alarm.create_save(user_id=user.id, time=f"0{index + 6}:30", label=label, enabled=initial)
        else:
            email = app.Email.create_save(sender_id=sender.id, recipient_ids=[user.id],
                                          subject=label, body=f"Original text {index}")
            thread = app.GlobalEmailThread.create_save(email_ids=[email.id])
            app.UserEmailThread.create_save(user_id=user.id, global_email_thread_id=thread.id,
                                            email_ids=[email.id], read=initial,
                                            incoming=True, outgoing=False)


def _prepare(root, tasks):
    _configure_appworld(root)
    from appworld.apps import get_all_apps
    from appworld.collections.models import ModelCollection
    from appworld.common.constants import DB_VERSION
    from freezegun import freeze_time

    base = root / "data/base_dbs"
    base.mkdir(parents=True)
    (root / "deps").mkdir()
    (root / "runs").mkdir()
    shutil.copytree(ASSETS / "data/api_docs/standard", root / "data/api_docs/standard")
    apps = sorted(set(get_all_apps(skip_admin=True)) | {"admin", "supervisor"})
    ModelCollection.load(from_db_home_path=str(base), load_apps=apps, create_db=True).close()
    with freeze_time("2024-12-31T12:00:00"):
        for task in tasks:
            _require(not (ASSETS / "data/tasks" / task.task_id).exists(), "official_id_collision")
            directory = _input(root, task)
            directory.mkdir(parents=True)
            (directory / "specs.json").write_text(json.dumps({
                "supervisor": SUPERVISOR, "instruction": task.instruction,
                "datetime": "2025-01-01T12:00:00", "db_version": DB_VERSION,
            }, sort_keys=True))
            models = ModelCollection.load(
                from_db_home_path=str(base), to_db_home_path=f":memory:seed_{task.task_id}",
                load_apps=[task.app, "supervisor"], skip_mandatory_apps=True,
            )
            try:
                _seed(models, task)
                models.save(str(directory / "dbs"), format="changes", skip_mandatory_apps=True)
            finally:
                models.close()
            for app in apps:
                (directory / "dbs" / f"{app}.jsonl").touch(exist_ok=True)


def _input_hash(root):
    files = sorted((root / "data/base_dbs").glob("*.db"))
    files.extend(sorted(path for path in (root / "data/tasks").rglob("*") if path.is_file()))
    manifest = [(str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def _act(root, task, branch):
    _configure_appworld(root)
    run_dir = root / "runs" / f"{task.task_id}-{branch}"
    run_dir.mkdir(mode=0o700)
    with patch.object(native, "ASSET_ROOT", root):
        worker = native.NativeAppWorldWorker(task.task_id, branch, run_dir, max_actions=16, max_api_requests=24)
        try:
            def rpc(op, **kwargs):
                reply = worker.handle_request(json.dumps({"op": op, **kwargs}))
                _require(reply["ok"] is True, f"actor_rpc_{op}")
                return reply["result"]

            def call(name, **args):
                action = json.dumps({"name": name, "arguments": args})
                reply = worker.handle_request(json.dumps({"op": "dispatch", "action": action}))
                _require(reply["ok"] is True, f"actor_api_{name}")
                return reply["result"]

            meta = rpc("metadata")
            _require(meta["instruction"] == task.instruction, "instruction")
            password, = (r["password"] for r in call("supervisor__show_account_passwords")
                         if r["account_name"] == task.app)
            username_key = "phone_number" if task.app == "phone" else "email"
            token = call(f"{task.app}__login", username=meta["supervisor"][username_key],
                         password=password)["access_token"]

            def api(name, **args):
                return call(f"{task.app}__{name}", access_token=token, **args)

            _, list_api, read_api, id_key = APPS[task.app]
            rows = api(list_api, **({"project_id": 0} if task.app == "todoist" else {}))
            label_key = {"phone": "label", "gmail": "subject"}.get(task.app, "title")
            if task.app == "todoist":
                rows = rows["no_section_tasks"]
            target, = (row[id_key] for row in rows if row[label_key] == task.labels[task.target_id - 1])
            _require(target == task.target_id, "target_id")
            observed = api(read_api, **{id_key: target})
            _require(observed[task.field] == task.initial[target - 1], "initial_value")

            def write(record_id, value):
                keys = {id_key: record_id}
                if task.app == "gmail":
                    api("mark_thread_read" if value else "mark_thread_unread", **keys)
                elif task.mode == "append":
                    api("add_content_to_note", append_or_prepend="append", added_content=value, **keys)
                else:
                    update = {"simple_note": "update_note", "todoist": "update_task", "phone": "update_alarm"}
                    api(update[task.app], **keys, **{task.field: value})

            answer = {}
            if task.mode in ("set", "append") and branch != "wrong":
                write(target, task.goal)
            if branch == "wrong" and task.mode == "append":
                write(target, task.goal)
                write(target, task.goal)
            if branch == "wrong" and task.mode == "keep":
                write(target, not task.goal)
            if branch == "collateral":
                old = task.initial[task.guard_id - 1]
                wrong = not old if type(old) is bool else "Changed by mistake"
                write(task.guard_id, wrong)
            if task.mode == "read":
                answer = {"answer": "Wrong answer" if branch == "wrong" else observed[task.field]}
            call("supervisor__complete_task", **answer)
            _require(rpc("save")["db_path"] == str(_output(root, task, branch)), "save_path")
        finally:
            worker.close()


def _state(path, app, slot):
    from appworld.collections.models import ModelCollection

    models = ModelCollection.load(
        from_db_home_path=str(path), to_db_home_path=f":memory:panel_score_{slot}",
        load_apps=[app, "supervisor"], skip_mandatory_apps=True,
    )
    try:
        completion, = models.supervisor.Task.all()
        tables = {}
        for name, table in models[app].SQLModel.name_to_model().items():
            rows = {row.id: row.to_dict(keep_computed=False) for row in table.all()}
            if name == "User":
                for row in rows.values():
                    row.pop("last_logged_in", None)
            tables[name] = rows
        return tables, completion.status, completion.answer
    finally:
        models.close()


def evaluate(root, task, snapshot):
    """Evaluate saved business state independently of the execution receipt."""
    _configure_appworld(root)
    expected, _, _ = _state(_input(root, task) / "dbs", task.app, "start")
    actual, completion, answer = _state(snapshot, task.app, "end")
    if task.mode not in ("keep", "read"):
        table = APPS[task.app][0]
        expected_row = expected[table][task.target_id]
        expected_row[task.field] = task.final
        expected_row.pop("updated_at", None)
        if task.target_id in actual[table]:
            actual[table][task.target_id].pop("updated_at", None)
    correct_answer = task.mode != "read" or (completion == "success" and json.loads(answer) == task.goal)
    return expected == actual and correct_answer, completion


def _score(root, task, branch):
    run_dir = root / "runs" / f"{task.task_id}-{branch}"
    receipt = json.loads((run_dir / "status.json").read_text())
    counts = receipt["cost"]
    _require(receipt["state"] == "closed" and receipt["score_eligible"] is True
             and receipt["task_id"] == task.task_id and receipt["experiment_name"] == branch
             and 1 <= counts["upstream_requests"] == counts["action_attempts"] == counts["api_invocations"] <= 16
             and receipt["last_successful_save_through_attempt"] == counts["action_attempts"],
             "score_eligibility")
    passed, completion = evaluate(root, task, _output(root, task, branch))
    _require(completion == "success", "missing_self_report")
    return {"pass": passed, "api_calls": counts["upstream_requests"]}


def _child(role, root, task, branch):
    command = [sys.executable, "-B", "-m", __spec__.name, role, str(root), task.task_id, branch]
    result = subprocess.run(command, capture_output=True, text=True, timeout=150, check=False)
    if result.returncode:
        failure = result.stderr.splitlines()
        kind = failure[-1].split(":")[0] if failure else ""
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", kind, re.ASCII) is None:
            kind = "unavailable"
        detail = failure[-1].removeprefix("ValueError: ") if failure else ""
        if re.fullmatch(r"actor_api_[a-z0-9_]+", detail, re.ASCII):
            kind = detail
        raise ValueError(f"{role} failed for {task.scenario}/{task.variant}/{branch} ({kind})")
    return result.stdout


def _verify_case(root, task, branch):
    _child("actor", root, task, branch)
    result = json.loads(_child("score", root, task, branch))
    _require(result["pass"] is BRANCHES[branch], f"predicate_{task.scenario}_{task.variant}_{branch}")
    return result["api_calls"]


def report(sample):
    tasks = [task for task in PANEL.values() if not sample or task.variant == 1]
    _require(len(PANEL) == 27 and len({task.scenario for task in PANEL.values()}) == 9, "roster")
    with tempfile.TemporaryDirectory(prefix="authored-panel-", dir=SCRATCH) as directory:
        root = Path(directory)
        _prepare(root, tasks)
        before = _input_hash(root)
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = [(task, branch, executor.submit(_verify_case, root, task, branch))
                       for task in tasks for branch in BRANCHES]
            total = 0
            for task, branch, future in pending:
                total += future.result()
                if branch == "collateral":
                    print(f"verified authored cases: {task.scenario}/{task.variant}", file=sys.stderr, flush=True)
        _require(_input_hash(root) == before, "initial_state_changed")
    return {
        "scope": "authored_panel_scripted_cpu", "sample": sample, "scenarios": 9,
        "variants_validated": len(tasks), "independent_actors": len(tasks) * len(BRANCHES),
        "independent_scorers": len(tasks) * len(BRANCHES), "scripted_api_calls": total,
        "expected_positive_passes": len(tasks), "expected_negative_rejections": 2 * len(tasks),
        "initial_state_manifest_sha256": before, "official_task_bodies_opened": 0,
        "official_scores": 0, "model_calls": 0, "gpu_jobs": 0, "study_registered": False,
    }


def main():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and SCRATCH.resolve(strict=True) == SCRATCH, "cpu_host")
    os.umask(0o077)
    args = sys.argv[1:]
    if args in ([], ["--sample"]):
        print(json.dumps(report(bool(args)), sort_keys=True))
        return
    role, directory, task_id, branch = args
    root = Path(directory)
    info = root.stat(follow_symlinks=False)
    _require(root.parent == SCRATCH and root.name.startswith("authored-panel-")
             and stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
             and info.st_uid == os.getuid() and branch in BRANCHES and role in ("actor", "score"), "child_root")
    task = PANEL[task_id]
    if role == "actor":
        _act(root, task, branch)
    else:
        print(json.dumps(_score(root, task, branch)))


if __name__ == "__main__":
    main()
