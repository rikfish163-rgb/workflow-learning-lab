"""CPU inputs for a prospective, target-grounded AppWorld mechanism study."""

import hashlib
import os
from pathlib import Path

from scripts import native_appworld_actor_004 as actor
from scripts import native_source_registration as registration
from scripts import native_state_recovery_evidence_001 as witness
from scripts import native_verification_token_counter as counter
from scripts import prepare_native_source_contexts as base
from scripts import run_native_state_recovery_source_003 as source
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.verification_token_counter import _message_payload

ID = "native-state-recovery-fixture-contexts-CPU-001"
PROOF = WORKSPACE / "artifacts" / f"{ID}.json"
PROBE = "artifacts/native-state-recovery-fixture-CPU-002.json"
PROBE_SHA256 = "78688cc10c16b2072397e177c1b25b608d624d7cd618ea24bdce85c4896f4cdf"
CASES = (
    {"case": "note-pin", "app": "simple_note", "create": "create_note", "resource_key": "note_id", "field": "pinned",
     "title": "WLL fixture blue notebook", "initial": False, "desired": True},
    {"case": "todo-complete-a", "app": "todoist", "create": "create_task", "resource_key": "task_id", "field": "is_completed",
     "title": "WLL fixture check the projector", "initial": False, "desired": True},
    {"case": "note-unpin", "app": "simple_note", "create": "create_note", "resource_key": "note_id", "field": "pinned",
     "title": "WLL fixture green notebook", "initial": True, "desired": False},
    {"case": "todo-complete-b", "app": "todoist", "create": "create_task", "resource_key": "task_id", "field": "is_completed",
     "title": "WLL fixture check the meeting room", "initial": False, "desired": True},
)


def _require(value):
    if not value:
        raise ValueError("native_state_recovery_fixture_protocol_invalid")


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def instructions(case):
    _require(any(_same(case, registered) for registered in CASES))
    if case["app"] == "simple_note":
        initial = "pinned" if case["initial"] else "unpinned"
        target = "unpinned" if case["initial"] else "pinned"
        action = (f'In Simple Note, create a new {initial} note titled "{case["title"]}" with the text '
                  f'"WLL synthetic fixture". Then make that same new note {target}.')
    else:
        action = f'In Todoist, create a new Inbox task titled "{case["title"]}". Then mark that same new task complete.'
    return (action + " Read the state of the new record before changing it, and read it again after the change. "
            "Only change the new record you create for this workflow.")


def actor_metadata(raw, case, raw_sha256):
    """Replace the trusted fixture instruction alone and preserve the API setting."""
    _require(type(raw) is dict and set(raw) == {"instruction", "supervisor", "app_descriptions"}
             and type(raw["instruction"]) is str and _hash(raw) == raw_sha256)
    return {**raw, "instruction": instructions(case)}


def qualify_witness(records, fault, case):
    """After transport authentication, require the transition on the new record."""
    _require(any(_same(case, registered) for registered in CASES))
    checked = witness.classify(records, fault)
    public = witness.public_projection(checked)
    result = {"witness": public, "fixture_target_verified": False}
    if checked["status"] != "qualified":
        return result
    indices = checked["private_evidence"]["indices"]
    decoded = witness._decode_records(records)
    blocked = decoded[indices["fault"]][0]
    app, field = case["app"], case["field"]
    target = witness._PAIRS.get(blocked["name"])
    if (not blocked["name"].startswith(app + "__") or target is None
            or target[1:] != (case["resource_key"], field) or blocked["arguments"].get(field) is not case["desired"]):
        return result
    key = case["resource_key"]
    resource = blocked["arguments"][key]
    before, after = (decoded[indices[name]][1]["result"] for name in ("before", "after"))
    if not (before.get("title") == after.get("title") == case["title"] and before.get(field) is case["initial"]):
        return result
    creates = [(action, observation) for action, observation in decoded[:indices["before"]]
               if action is not None and action["name"] == app + "__" + case["create"]
               and action["arguments"].get("title") == case["title"] and observation["ok"]]
    result["fixture_target_verified"] = (len(creates) == 1 and type(creates[0][1]["result"]) is dict
        and _same(creates[0][1]["result"].get(key), resource)
        and _same(creates[0][0]["arguments"].get("access_token"), blocked["arguments"]["access_token"]))
    return result


def prepare():
    """Count only custom initial messages; no model, world, or evaluator starts."""
    base._host_preflight()
    raw = registration.public_read(WORKSPACE, PROBE)
    _require(hashlib.sha256(raw).hexdigest() == PROBE_SHA256)
    engineering = _decode(raw)
    _require(engineering["status"] == "passed" and engineering["model_calls"] == engineering["GPU_jobs"] == 0
             and all(row["fixture_write_verified_after_readback"] is True for row in engineering["cases"]))
    roles, _ = registration._cohort(WORKSPACE)
    natural = set(source.episode_binding().TASK_IDS)
    templates = sorted(task for task in roles["origin"] if task.endswith("_3") and task not in natural)[2:2 + len(CASES)]
    _require(len({task[:-2] for task in templates}) == len(CASES)
             and not {task[:-2] for task in templates} & {task[:-2] for task in roles["utility"]})
    runtime = {"config.json": registration.MODEL_CONFIG_SHA256, **registration.TOKENIZER_SHA256}
    _require(base._runtime_hashes() == runtime)
    rows = []
    progress = {"ground_truth_load_attempts": 0}
    with base._native_task_classes() as (task_type, ground_truth):
        with base._forbid_ground_truth(ground_truth, progress):
            for task, case in zip(templates, CASES, strict=True):
                original = base._task_metadata(task_type.load(task_id=task, load_ground_truth=False), task)
                original_sha = _hash(original)
                adapted = actor_metadata(original, case, original_sha)
                messages = actor.build_actor_messages(adapted, prefix_records=None, review_hint=None)
                tokens = counter.count_input_tokens(messages)
                _require(type(tokens) is int and 1 <= tokens <= 15360)
                rows.append({"case": case["case"], "template_task_id": task, "worker_metadata_sha256": original_sha,
                    "actor_metadata_sha256": _hash(adapted), "instruction_sha256": _hash(adapted["instruction"]),
                    "messages_sha256": _message_payload(messages)[1], "input_tokens": tokens})
    _require(not progress["ground_truth_load_attempts"] and base._cuda_uninitialized() and base._runtime_hashes() == runtime)
    return {"id": ID, "status": "prospective_not_registered", "scope": "authored_workflows_on_real_AppWorld_tools",
            "activation": "only_after_source_003_terminal_with_insufficient_witnesses", "minimum_qualified_scenarios": 2,
            "source_denominator": 4, "original_task_success_scored": False, "template_role": "origin",
            "read_before_and_after_in_instruction": True, "correction_strategy_in_instruction": False,
            "controlled_refusal_in_instruction": False, "engineering_probe_sha256": PROBE_SHA256,
            "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "ground_truth_load_attempts": 0, "model_calls": 0, "GPU_jobs": 0, "rows": rows}


def main():
    os.umask(0o077)
    proof = prepare()
    with PROOF.open("xb") as file:
        file.write(_encode(proof) + b"\n")
        file.flush()
        os.fsync(file.fileno())
    print("fixture contexts measured without model calls", len(proof["rows"]))


if __name__ == "__main__":
    main()
