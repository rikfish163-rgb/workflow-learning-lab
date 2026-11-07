"""Check the write refusal and readback against fresh AppWorld fixture records."""

import hashlib
import os
import socket
import sys
from pathlib import Path

from scripts import native_source_registration as registration
from scripts import native_state_recovery_evidence_001 as witness
from scripts import run_native_state_recovery_source_003 as source
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode
from scripts.native_state_refusal_client_001 import NativeWorkerClient

ID = "native-state-recovery-fixture-CPU-001"
PRIVATE = Path(registration.PRIVATE_STATE_ROOT) / ID
PROOF = WORKSPACE / "artifacts" / f"{ID}.json"
CASES = (
    ("simple_note", "create_note", "show_note", "update_note", "note_id", "pinned",
     {"title": "WLL fixture note", "content": "Synthetic fixture record", "pinned": False}),
    ("todoist", "create_task", "show_task", "update_task", "task_id", "is_completed",
     {"project_id": 0, "title": "WLL fixture task"}),
)


class FixtureError(ValueError):
    """Fixed diagnostic code for a failed fixture invariant."""


def _require(condition, code):
    if not condition:
        raise FixtureError(code)


def _case(task_id, spec):
    app, create_api, read_api, write_api, resource_key, field, create_args = spec
    experiment = ID + "-" + app
    private = PRIVATE / app
    records, replies = [], []
    worker = None
    try:
        worker = NativeWorkerClient(task_id, experiment, private,
            max_actions=12, max_api_requests=12, rpc_timeout=60, wall_seconds=240)
        metadata = worker.request({"op": "metadata"}, timeout=120)
        _require(metadata["ok"] is True, "metadata_failed")

        def dispatch(name, arguments):
            action = _encode({"name": name, "arguments": arguments}).decode("ascii")
            reply = worker.request({"op": "dispatch", "action": action}, timeout=60)
            observation = {key: value for key, value in reply.items() if key not in {"cost", "freshness"}}
            records.append({"action": action, "observation": observation})
            replies.append(reply)
            return reply

        def success(name, arguments, error):
            reply = dispatch(name, arguments)
            _require(reply["ok"] is True, error)
            return reply["result"]

        credentials = success("supervisor__show_account_passwords", {}, "fixture_account_lookup_failed")
        accounts = [item["password"] for item in credentials if item["account_name"] == app]
        _require(len(accounts) == 1, "fixture_account_unavailable")
        login = success(f"{app}__login", {"username": metadata["result"]["supervisor"]["email"],
                                         "password": accounts[0]}, "fixture_login_failed")
        auth = {"access_token": login["access_token"]}
        resource = success(f"{app}__{create_api}", {**auth, **create_args}, "fixture_creation_failed")[resource_key]
        args = {**auth, resource_key: resource}
        before = success(f"{app}__{read_api}", args, "fixture_initial_read_failed")
        _require(before[resource_key] == resource and before[field] is False, "fixture_initial_state_invalid")
        write = {**args, field: True}
        refused = dispatch(f"{app}__{write_api}", write)
        _require(refused["ok"] is False and refused["error_type"] == "ToolDispatchError"
                 and refused["message"] == witness.FAULT_MESSAGE
                 and all(refused["cost"][key] == replies[-2]["cost"][key]
                         for key in ("api_invocations", "upstream_requests")), "fixture_refusal_invalid")
        unchanged = success(f"{app}__{read_api}", args, "fixture_refusal_read_failed")
        _require(unchanged[resource_key] == resource and unchanged[field] is False, "fixture_refusal_mutated_state")
        success(f"{app}__{write_api}", write, "fixture_correction_failed")
        after = success(f"{app}__{read_api}", args, "fixture_readback_failed")
        _require(after[resource_key] == resource and after[field] is True, "fixture_readback_invalid")
        _require(worker.request({"op": "save"}, timeout=60)["ok"] is True, "fixture_save_failed")
        _require(worker.request({"op": "close"}, timeout=60)["ok"] is True and worker.wait(timeout=60) == 0,
                 "fixture_close_failed")
        snapshot = worker.snapshot(timeout=60)
        fault = _decode((private / "fault-provenance.json").read_bytes())
        classified = witness.public_projection(witness.classify(records, fault))
        _require(classified["status"] == "qualified" and fault["fault_index"] == 4, "fixture_witness_failed")
        return {"app": app, "witness": classified, "state_unchanged_after_refusal": True,
                "fixture_write_verified_after_readback": True, "snapshot_manifest_sha256": snapshot["manifest_sha256"],
                "cost": snapshot["cost"]}
    finally:
        if worker is not None:
            worker.release()


def main():
    _require(socket.gethostname() == "amax" and Path.cwd() == WORKSPACE and sys.executable == registration.PYTHON
             and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("TMPDIR") == registration.TMPDIR,
             "fixture_host_invalid")
    os.umask(0o077)
    roles, _ = registration._cohort(WORKSPACE)
    active_tasks = set(source.episode_binding().TASK_IDS)
    chosen = sorted(task for task in roles["origin"] if task.endswith("_3") and task not in active_tasks)[:len(CASES)]
    _require(len(chosen) == len(CASES) and not PROOF.exists(), "fixture_namespace_invalid")
    PRIVATE.mkdir(mode=0o700)
    rows = []
    for task, spec in zip(chosen, CASES, strict=True):
        try:
            rows.append(_case(task, spec))
        except Exception as error:
            safe = str(error) if type(error) is FixtureError else "worker_error"
            rows.append({"app": spec[0], "failure": safe})
    passed = all(row.get("fixture_write_verified_after_readback") is True for row in rows)
    proof = {"id": ID, "scope": "human_authored_fixture_actions_no_model", "status": "passed" if passed else "failed",
             "model_calls": 0, "GPU_jobs": 0, "official_scorer_calls": 0, "official_tasks_attempted": 0,
             "probe_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": rows}
    raw = _encode(proof) + b"\n"
    with PROOF.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    print(raw.decode("ascii"), end="")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
