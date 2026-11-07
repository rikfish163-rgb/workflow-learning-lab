"""Exercise the real fixture actor, controller and worker with a scripted policy."""

import hashlib
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from scripts import native_state_recovery_fixture_episode_001 as fixture
from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts import prepare_native_source_contexts as preflight
from scripts import run_native_state_recovery_fixture_001 as run
from scripts.native_appworld_worker_client import WORKSPACE, _encode

ID = "native-state-recovery-fixture-actor-CPU-001"


class ScriptedPolicy:
    max_tokens = 1024

    def __init__(self, case, input_tokens):
        self.case, self.input_tokens = case, input_tokens
        self.deadline = time.monotonic() + 780
        self.state = {}

    def check_health(self):
        return {"ready": True, "scripted_cpu_probe": True}

    def count(self, _messages):
        return self.input_tokens

    def generate(self, messages, seed):
        from scripts.native_appworld_worker_client import _decode

        step = seed - 17
        if step == 0:
            meta = _decode(messages[1]["content"].encode())
            assert meta["instruction"] == protocol.instructions(self.case)
            self.state["email"] = meta["supervisor"]["email"]
            name, args = "supervisor__show_account_passwords", {}
        else:
            observation = _decode(messages[-1]["content"].encode())
            if step == 1:
                password, = [item["password"] for item in observation["result"] if item["account_name"] == "simple_note"]
                name, args = "simple_note__login", {"username": self.state["email"], "password": password}
            elif step == 2:
                self.state["access_token"] = observation["result"]["access_token"]
                name, args = "simple_note__create_note", {"access_token": self.state["access_token"],
                    "title": self.case["title"], "content": "WLL synthetic fixture", "pinned": False}
            else:
                if step == 3:
                    self.state["note_id"] = observation["result"]["note_id"]
                handle = {key: self.state[key] for key in ("access_token", "note_id")}
                if step in (3, 5, 7):
                    name, args = "simple_note__show_note", handle
                elif step in (4, 6):
                    assert observation["result"]["pinned"] is False
                    name, args = "simple_note__update_note", {**handle, "pinned": True}
                else:
                    assert step == 8 and observation["result"]["pinned"] is True
                    name, args = "supervisor__complete_task", {}
        return {"text": _encode({"name": name, "arguments": args}).decode(),
                "prompt_tokens": self.input_tokens, "completion_tokens": 1}


def main():
    preflight._host_preflight()
    os.umask(0o077)
    design = run._design(WORKSPACE)
    rows = design["rows"]
    template_ids = tuple(row["template_task_id"] for row in rows)
    base, queue, _, _ = run.bindings()
    root = Path(run.PRIVATE_STATE_ROOT) / ID
    root.mkdir(mode=0o700)
    reg = SimpleNamespace(**{**vars(base), "PRIVATE_RUN_ROOT": str(root)})
    episode = run.natural.prior._load("native_state_recovery_episode_001", (
        ("native-state-recovery-001-", run.EXPERIMENT, 2),
        ('receipt.get("is_policy_episode") is True', 'receipt.get("is_policy_episode") is False', 1),
        ('intent.get("is_policy_episode") is True', 'intent.get("is_policy_episode") is False', 1),
    ), ID=run.ID, PLAN=run.PLAN, TASK_IDS=template_ids)
    fixture.install(episode, rows)
    bound = episode.bind(reg, queue)
    owner = uuid.uuid4().hex
    run_dir = root / f"unit-00-first_attempt-{owner}"
    run_dir.mkdir(mode=0o700)
    context = fixture._case_context(rows[0], protocol.CASES[0])
    model = ScriptedPolicy(protocol.CASES[0], context["input_tokens"])
    receipt = bound.core.run_task(template_ids[0], f"{run.EXPERIMENT}00-first_attempt-{owner}", run_dir,
        limits=episode.limits(), seed=17, model=model, token_counter=model.count, rpc_timeout=60,
        worker_wall_seconds=780, setup_seconds=120, cleanup_seconds=60, is_policy_episode=False,
        source_metadata=None, prefix_records=None, review_hint=None, _worker_factory=bound.core.NativeWorkerClient)
    assert receipt["status"] == "saved_and_closed" and receipt["is_policy_episode"] is False
    reconciled = fixture.collect(episode, bound.core, run_dir, receipt, context)
    assert reconciled["fixture"]["fixture_target_verified"] is True and run._main_pid(receipt["snapshot"]["unit_name"]) == 0
    proof = {"id": ID, "status": "passed", "scope": "scripted_cpu_policy_real_AppWorld_tools",
             "model_calls": 0, "GPU_jobs": 0, "official_scorer_calls": 0,
             "scripted_actions": receipt["actor_outcome"]["cost"]["action_attempts"],
             "fixture_target_verified": True, "worker_release_verified": True,
             "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (WORKSPACE / "artifacts" / f"{ID}.json").open("xb") as file:
        file.write(_encode(proof) + b"\n")
        file.flush()
        os.fsync(file.fileno())
    print("fixture actor, live worker, exact readback, and release verified without a model call")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("native_state_recovery_fixture_actor_probe_failed") from None
