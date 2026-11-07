"""Verify the authored scorer against real native CPU units and private receipts."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import native_appworld_controller as controller
from scripts import native_appworld_scorer as original_scorer
from scripts import native_appworld_worker_client as client
from scripts import native_state_recovery_authored_runtime_001 as runtime
from scripts.native_appworld_actor import ActorLimits
from scripts.native_state_recovery_authored_panel_001 import PANEL
from scripts.probe_native_state_recovery_authored_panel_001 import _input, _input_hash, _prepare
from scripts.probe_native_state_recovery_authored_world_001 import SCRATCH, SUPERVISOR

TASK = next(task for task in PANEL.values() if task.scenario == "pin_note" and task.variant == 1)
ACTOR_SECONDS = 45
WORKER_SECONDS = 135


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _add_other_app(root):
    from appworld.collections.models import ModelCollection
    from freezegun import freeze_time

    source = _input(root, TASK) / "dbs"
    models = ModelCollection.load(
        from_db_home_path=str(source), to_db_home_path=":memory:authored_auxiliary",
        load_apps=["phone", "supervisor"], skip_mandatory_apps=True,
    )
    with freeze_time("2024-12-31T12:00:00"):
        try:
            phone = models.phone.User.create_save(
                first_name=SUPERVISOR["first_name"], last_name=SUPERVISOR["last_name"],
                phone_number=SUPERVISOR["phone_number"], password="synthetic-auxiliary-only",
            )
            models.phone.Alarm.create_save(user_id=phone.id, time="08:00", enabled=False)
            supervisor, = models.supervisor.Supervisor.all()
            models.supervisor.AccountPassword.create_save(
                supervisor_id=supervisor.id, account_name="phone", password="synthetic-auxiliary-only",
            )
            staging = root / "auxiliary"
            models.save(str(staging), format="changes", skip_mandatory_apps=True)
            for app in ("phone", "supervisor"):
                os.replace(staging / f"{app}.jsonl", source / f"{app}.jsonl")
            staging.rmdir()
        finally:
            models.close()


def _scripted_actions(worker, _model, _token_counter, *, metadata, **_):
    reply = None

    def call(name, **arguments):
        nonlocal reply
        reply = worker.request({"op": "dispatch", "action": json.dumps({
            "name": name, "arguments": arguments,
        })})
        _require(reply["ok"] is True, "scripted_api_failed")
        return reply["result"]

    passwords = {row["account_name"]: row["password"] for row in call("supervisor__show_account_passwords")}
    supervisor = metadata["supervisor"]
    token = call("simple_note__login", username=supervisor["email"], password=passwords["simple_note"])
    keys = {"note_id": TASK.target_id, "access_token": token["access_token"]}
    _require(call("simple_note__show_note", **keys)["pinned"] is False, "scripted_initial_state")
    call("simple_note__update_note", **keys, pinned=True)
    _require(call("simple_note__show_note", **keys)["pinned"] is True, "scripted_updated_state")
    if worker._experiment_name == "collateral":
        phone = call("phone__login", username=supervisor["phone_number"], password=passwords["phone"])
        call("phone__update_alarm", alarm_id=1, enabled=True, access_token=phone["access_token"])
    call("supervisor__complete_task")
    return {
        "status": "model_reported_complete", "stop_reason": "scripted_CPU_only",
        "outcome_unknown": False, "unknown_cost": False,
        "cost": {**reply["cost"], "generations": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }


def _actor(root, initial_sha, experiment):
    root = runtime.verify_assets(root, initial_sha)
    parent = runtime.parent_unit(root, experiment)
    _require(original_scorer._main_pid(parent) == os.getpid(), "actor_parent_identity")
    run_dir = root / "runs" / experiment
    run_dir.mkdir(mode=0o700)
    result = controller.run_task(
        TASK.task_id, experiment, run_dir,
        limits=ActorLimits(16, 16, 24, 2048, 4096, 1024, 256, ACTOR_SECONDS),
        seed=17, model=SimpleNamespace(check_health=lambda: {"scripted_cpu": True}), token_counter=None,
        rpc_timeout=75, worker_wall_seconds=WORKER_SECONDS, setup_seconds=75, cleanup_seconds=15,
        is_policy_episode=False, _worker_factory=runtime.worker_factory(root, initial_sha),
        _actor=_scripted_actions,
    )
    _require(result["status"] == "saved_and_closed" and result["worker_exit_code"] == 0, "actor_receipt")
    if experiment == "correct":
        (root / "receipt-ready").touch()
        deadline = time.monotonic() + 45
        while not (root / "allow-exit").is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        _require((root / "allow-exit").is_file(), "actor_gate_deadline")


def _unit(unit, *argv):
    env = {key: os.environ[key] for key in (
        "PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
    ) if key in os.environ}
    env.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(SCRATCH), PYTHONDONTWRITEBYTECODE="1",
               LANG="C.UTF-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    bootstrap = f"import runpy,sys;sys.path.insert(0,{str(client.WORKSPACE)!r});runpy.run_module({argv[0]!r},run_name='__main__')"
    return subprocess.Popen(
        [client.SYSTEMD_RUN, "--user", "--pipe", "--wait", "--collect", "--quiet",
         f"--unit={unit}", f"--working-directory={client.WORKSPACE}",
         "--property=MemoryMax=3G", "--property=CPUQuota=200%", "--property=TasksMax=128",
         "--property=NoNewPrivileges=yes", "--property=RestrictAddressFamilies=AF_UNIX",
         "--property=KillMode=control-group", "--property=RuntimeMaxSec=180",
         "--", "/usr/bin/env", "-i", *(f"{key}={value}" for key, value in sorted(env.items())),
         client.PYTHON, "-s", "-P", "-B", "-c", bootstrap, *argv[1:]],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _receipt(root, experiment):
    path = root / "runs" / experiment / "execution-receipt.json"
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _score(root, initial_sha, experiment, name, expected, *, wrong_digest=False):
    receipt, digest = _receipt(root, experiment)
    if wrong_digest:
        digest = "0" * 64
    score_dir = root / "scores" / name
    score_dir.mkdir(mode=0o700)
    unit = runtime.parent_unit(root, f"scorer-{name}").replace("-parent-", "-score-")
    process = _unit(unit, runtime.__name__,
        "--asset-root", str(root), "--initial-sha256", initial_sha,
        "--receipt", str(receipt), "--receipt-sha256", digest, "--out-dir", str(score_dir))
    exit_code = process.wait(timeout=185)
    result = json.loads((score_dir / "scoring-result.json").read_bytes())
    journal = [json.loads(line)["event"] for line in (score_dir / "scoring-journal.jsonl").read_bytes().splitlines()]
    scored = expected is not None
    _require(exit_code == (0 if scored else 1)
             and result["status"] == ("scored" if scored else "unknown")
             and result["success"] is expected
             and journal == (["scoring_intent", "validated_before_evaluation", "scoring_stop"]
                             if scored else ["scoring_intent", "scoring_stop"]),
             f"unexpected_score_{name}")
    _require(original_scorer._main_pid(unit) == 0, "score_unit_still_running")


def report():
    with tempfile.TemporaryDirectory(prefix="authored-panel-", dir=SCRATCH) as directory:
        root = Path(directory)
        _prepare(root, [TASK])
        _add_other_app(root)
        (root / "scores").mkdir()
        initial_sha = _input_hash(root)
        schema = root / "data/api_docs/standard/simple_note.json"
        original = schema.read_bytes()
        schema.write_bytes(original + b"\n")
        try:
            runtime.verify_assets(root, initial_sha)
        except ValueError as error:
            _require(str(error) == "native_authored_runtime_binding_invalid", "unexpected_asset_rejection")
        else:
            raise ValueError("modified_api_schema_accepted")
        finally:
            schema.write_bytes(original)
        for experiment in ("correct", "collateral"):
            unit = runtime.parent_unit(root, experiment)
            process = _unit(unit, __spec__.name, "actor", str(root), initial_sha, experiment)
            if experiment == "correct":
                try:
                    deadline = time.monotonic() + WORKER_SECONDS
                    while not (root / "receipt-ready").exists() and process.poll() is None:
                        _require(time.monotonic() < deadline, "actor_receipt_deadline")
                        time.sleep(0.2)
                    _require((root / "receipt-ready").is_file() and original_scorer._main_pid(unit) > 0,
                             "actor_early_exit")
                    _score(root, initial_sha, experiment, "still-running", None)
                finally:
                    (root / "allow-exit").touch()
                    code = process.wait(timeout=185)
            else:
                code = process.wait(timeout=185)
            _require(code == 0 and original_scorer._main_pid(unit) == 0, "actor_unit_exit")
            _score(root, initial_sha, experiment, experiment, experiment == "correct")
        _score(root, initial_sha, "correct", "wrong-digest", None, wrong_digest=True)
        db_path = root / "experiments/outputs/correct/tasks" / TASK.task_id / "dbs/simple_note.jsonl"
        with db_path.open("ab") as file:
            file.write(b"\n")
        _score(root, initial_sha, "correct", "modified-snapshot", None)
        _require(_input_hash(root) == initial_sha, "initial_state_modified")
    return {
        "scope": "authored_isolated_CPU_receipt_integration", "scripted_episodes": 2,
        "correct_state_passed": True, "cross_app_write_rejected": True,
        "live_parent_rejected_before_evaluation": True, "wrong_receipt_digest_rejected": True,
        "modified_api_schema_rejected": True,
        "modified_snapshot_rejected_before_evaluation": True, "model_calls": 0, "gpu_jobs": 0,
        "official_scores": 0, "study_registered": False,
    }


if __name__ == "__main__":
    os.umask(0o077)
    if sys.argv[1:] and sys.argv[1] == "actor":
        _, root, initial_sha, experiment = sys.argv[1:]
        _require(experiment in ("correct", "collateral"), "experiment")
        _actor(root, initial_sha, experiment)
    else:
        _require(len(sys.argv) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "probe_config")
        print(json.dumps(report(), sort_keys=True))
