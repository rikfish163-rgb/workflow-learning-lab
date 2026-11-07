"""Exercise both draft prompt conditions with a scripted backend and real CPU units."""

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from scripts import native_appworld_actor_004 as actor
from scripts import native_appworld_controller_006 as controller
from scripts import native_appworld_scorer as scorer
from scripts import native_state_recovery_authored_runtime_001 as runtime
from scripts import native_state_recovery_transition_review_002 as review
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import run_native_source_pilot as audit
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode
from scripts.native_generation_accounting_004 import generation_accounting
from scripts.native_state_recovery_authored_panel_001 import PANEL
from scripts.native_verification_token_counter_003 import count_input_tokens
from scripts.probe_native_state_recovery_authored_panel_001 import _input_hash, _prepare
from scripts.probe_native_state_recovery_authored_runtime_001 import _score, _unit
from scripts.probe_native_state_recovery_authored_world_001 import SCRATCH
from scripts.run_native_experience_utility_001 import _limits
from scripts.verification_token_counter import _message_payload

TASK = next(task for task in PANEL.values() if task.scenario == "pin_note" and task.variant == 1)
SCRIPT_STEPS = 7


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _hint(prior):
    source = prior["source"]
    bundle = review.load_reviewed_bundle(source["writer_plan_sha256"], source["writer_result_sha256"],
                                         source["review_manifest_sha256"])
    return bundle["insertion_text"]


class ScriptedBackend:
    """Produce JSON actions using only the supplied task metadata and tool observations."""

    def __init__(self, *, max_tokens, seed, hint, label, worker_seconds):
        self.max_tokens = max_tokens
        self.seed = seed
        self.hint = hint
        self.label = label
        self.deadline = time.monotonic() + worker_seconds
        self.step = 0
        self.last_count = None

    def check_health(self):
        return {"scripted_cpu_backend": True, "network_requests": 0}

    def count(self, messages):
        tokens = count_input_tokens(messages, deadline=self.deadline)
        self.last_count = (_message_payload(messages)[1], tokens)
        return tokens

    def generate(self, messages, *, seed):
        offset = 2 + (self.hint is not None)
        _require(self.last_count is not None and _message_payload(messages)[1] == self.last_count[0]
                 and seed == self.seed + self.step and len(messages) == offset + 2 * self.step,
                 "scripted_generation_binding")
        if self.hint is not None:
            _require(messages[2] == {"role": "user", "content": self.hint}, "scripted_hint_binding")

        def observation(index):
            reply = json.loads(messages[offset + 2 * index + 1]["content"])
            _require(reply["ok"] is True, "scripted_observation")
            return reply["result"]

        if self.step == 0:
            name, arguments = "supervisor__show_account_passwords", {}
        elif self.step == 1:
            password, = (row["password"] for row in observation(0) if row["account_name"] == "simple_note")
            metadata = json.loads(messages[1]["content"])
            name = "simple_note__login"
            arguments = {"username": metadata["supervisor"]["email"], "password": password}
        else:
            token = observation(1)["access_token"]
            if self.step == 2:
                name, arguments = "simple_note__search_notes", {"access_token": token}
            else:
                note, = (row["note_id"] for row in observation(2) if row["title"] == self.label)
                keys = {"note_id": note, "access_token": token}
                if self.step == 3:
                    name, arguments = "simple_note__show_note", keys
                elif self.step == 4:
                    _require(observation(3)["pinned"] is False, "scripted_before_state")
                    name, arguments = "simple_note__update_note", {**keys, "pinned": True}
                elif self.step == 5:
                    name, arguments = "simple_note__show_note", keys
                else:
                    _require(self.step == 6 and observation(5)["pinned"] is True, "scripted_after_state")
                    name, arguments = "supervisor__complete_task", {}
        self.step += 1
        return {"text": _encode({"name": name, "arguments": arguments}).decode(),
                "prompt_tokens": self.last_count[1], "completion_tokens": 1}


def _act(root, initial_sha, arm):
    root = runtime.verify_assets(root, initial_sha)
    unit = runtime.parent_unit(root, arm)
    _require(scorer._main_pid(unit) == os.getpid(), "probe_actor_unit")
    prior = comparison._read_inputs()[comparison.PRIOR_PLAN]
    policy = prior["utility"]
    hint = _hint(prior) if arm == "experience" else None
    seed = policy["generation_seed"]
    backend = ScriptedBackend(max_tokens=policy["max_new_tokens"], seed=seed, hint=hint,
                              label=TASK.labels[TASK.target_id - 1], worker_seconds=policy["worker_wall_seconds"])
    run_dir = root / "runs" / arm
    run_dir.mkdir(mode=0o700)
    result = controller.run_task(
        TASK.task_id, arm, run_dir, limits=_limits(policy), seed=seed, model=backend,
        token_counter=backend.count, rpc_timeout=policy["rpc_timeout_default_seconds"],
        worker_wall_seconds=policy["worker_wall_seconds"], setup_seconds=policy["setup_seconds"],
        cleanup_seconds=policy["cleanup_seconds"], review_hint=hint, is_policy_episode=False,
        _worker_factory=runtime.worker_factory(root, initial_sha),
    )
    _require(result["status"] == "saved_and_closed" and backend.step == SCRIPT_STEPS
             and result["actor_outcome"]["status"] == "model_reported_complete", "probe_actor_receipt")


def _normalize(records):
    values = []
    for record in records:
        action = json.loads(record["action"])
        if "access_token" in action["arguments"]:
            action["arguments"]["access_token"] = "<session>"
        observation = record["observation"].copy()
        if action["name"] == "simple_note__login":
            observation["result"] = {**observation["result"], "access_token": "<session>"}
        values.append((action, observation))
    return values


def _audit(root, arm, hint, seed):
    run_dir = root / "runs" / arm
    receipt = _decode((run_dir / "execution-receipt.json").read_bytes())
    journal = scorer._journal((run_dir / "controller.jsonl").read_bytes())
    _, metadata_row = scorer._one(journal, "metadata_result")
    _, actor_start = scorer._one(journal, "actor_start")
    _, controller_intent = scorer._one(journal, "controller_intent")
    metadata = metadata_row["reply"]["result"]
    expected = actor.build_actor_messages(metadata, review_hint=hint)
    context = {"metadata_sha256": _hash(_encode(metadata)), "messages_sha256": _message_payload(expected)[1]}
    _, records, generations = audit.actor_evidence(receipt, journal, expected_context=context, review_hint=hint)
    accounting = generation_accounting(journal)
    costs = receipt["actor_outcome"]["cost"]
    _require(receipt["is_policy_episode"] is False and receipt["official_scores_computed"] == 0
             and receipt["replay_outcome"] is None and receipt["worker_exit_code"] == 0
             and actor_start["seed"] == seed and controller_intent["review_hint_present"] is (hint is not None)
             and costs["generations"] == costs["action_attempts"] == costs["api_invocations"]
             == costs["upstream_requests"] == accounting["generation_results"] == len(records) == SCRIPT_STEPS
             and costs["prompt_tokens"] == accounting["journal_returned_prompt_tokens_lower_bound"]
             and accounting["unreturned_generation_intents_upper_bound"] == 0
             and all(entry["intent"]["seed"] == seed + index for index, entry in enumerate(generations)),
             "probe_trace_binding")
    return {
        "context": context, "limits": controller_intent["limits"], "messages": actor_start["messages"],
        "records": _normalize(records), "prompt_tokens": costs["prompt_tokens"],
    }


def report():
    draft = comparison.drafts()["authored"]
    prior = comparison._read_inputs()[comparison.PRIOR_PLAN]
    hint = _hint(prior)
    pair_index, = (index for index, (task, _, _) in enumerate(draft["expected_pairs"]) if task == TASK.task_id)
    order = draft["arm_orders"][pair_index]
    seed = draft["expected_pairs"][pair_index][2]
    with tempfile.TemporaryDirectory(prefix="authored-panel-", dir=SCRATCH) as directory:
        root = Path(directory)
        _prepare(root, PANEL.values())
        (root / "scores").mkdir()
        initial_sha = _input_hash(root)
        _require(initial_sha == draft["initial_state_manifest_sha256"], "probe_initial_manifest")
        audited = {}
        for arm in order:
            unit = runtime.parent_unit(root, arm)
            process = _unit(unit, __spec__.name, "actor", str(root), initial_sha, arm)
            code = process.wait(timeout=185)
            _require(code == 0 and scorer._main_pid(unit) == 0, "probe_actor_exit")
            audited[arm] = _audit(root, arm, hint if arm == "experience" else None, seed)
            _score(root, initial_sha, arm, arm, True)
        control, experience = (audited[arm] for arm in comparison.ARMS)
        _require(control["context"]["metadata_sha256"] == experience["context"]["metadata_sha256"]
                 and control["messages"] + [{"role": "user", "content": hint}] == experience["messages"]
                 and control["records"] == experience["records"] and control["limits"] == experience["limits"]
                 and experience["prompt_tokens"] > control["prompt_tokens"]
                 and _input_hash(root) == initial_sha, "probe_matched_pair")
    result = comparison.summarize(draft, [])
    _require(result["complete_pairs"] == 0 and result["statistics"] is None, "probe_counts_as_real")
    return {
        "scope": "authored_CPU_pair_with_scripted_generation", "scripted_pairs": 1, "scripted_arms": 2,
        "scripted_generations": 2 * SCRIPT_STEPS, "real_tool_calls": 2 * SCRIPT_STEPS,
        "separate_validated_authored_scores": 2, "same_metadata_and_limits": True,
        "only_reviewed_hint_changed": True, "same_tool_trace_excluding_session_tokens": True,
        "prompt_tokens_by_arm": {arm: audited[arm]["prompt_tokens"] for arm in comparison.ARMS},
        "initial_state_manifest_sha256": initial_sha, "official_task_bodies_opened": 0,
        "real_model_comparisons": 0, "model_calls": 0, "gpu_jobs": 0, "study_registered": False,
    }


if __name__ == "__main__":
    os.umask(0o077)
    _require(Path.cwd() == WORKSPACE and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "probe_config")
    if sys.argv[1:] and sys.argv[1] == "actor":
        _, root, initial_sha, arm = sys.argv[1:]
        _require(arm in comparison.ARMS, "probe_arm")
        _act(root, initial_sha, arm)
    else:
        _require(len(sys.argv) == 1, "probe_arguments")
        print(json.dumps(report(), sort_keys=True))
