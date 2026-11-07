"""CPU fixtures for the fixed source coordinator; no jobs or model execution."""

import copy
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_source_registration as registration
from scripts import run_native_source_pilot as pilot
from scripts.native_appworld_actor import build_actor_messages
from scripts.native_appworld_worker_client import _decode, _encode
from scripts.verification_token_counter import _message_payload

SHA = "a" * 64
HINT = "Fixed synthetic generic review hint."


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _action(name, **arguments):
    return _encode({"name": name, "arguments": arguments}).decode("ascii")


def _fixture_episode(index, phase, context, prefix, *, complete=True, no_boundary=False,
                     generation_count=6, returned_unused=False, target_repetition=False):
    task = registration.TASK_IDS[index]
    metadata = {"instruction": "Synthetic " + task, "supervisor": {},
                "app_descriptions": {"synthetic": "CPU fixture public API"}}
    prefix_records = None if prefix is None else prefix["prefix_records"]
    messages = build_actor_messages(metadata, prefix_records=prefix_records,
                                    review_hint=HINT if phase == "review" else None)
    cost = {"generations": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "action_attempts": len(prefix_records or []), "api_invocations": len(prefix_records or []),
            "upstream_requests": len(prefix_records or [])}
    rows = [{"event": "metadata_result", "reply": {"ok": True, "result": metadata}},
            {"event": "actor_start", "messages": copy.deepcopy(messages),
             "review_hint_index": 2 + 2 * len(prefix_records or []) if phase == "review" else None}]
    actions = ([_action("synthetic__fail", task=task), _action("supervisor__complete_task")]
               if phase == "first_attempt" and not no_boundary else
               [_action("supervisor__complete_task")] if phase == "first_attempt" else
               [_action("synthetic__step", task="same" if target_repetition else task,
                        step=step) for step in range(generation_count - 1)] +
               [_action("supervisor__complete_task")])
    for action_index, action in enumerate(actions):
        response = {"text": action, "prompt_tokens": 100, "completion_tokens": 1,
                    "format_error": None, "raw_reply": {"model": registration.MODEL_ID,
                        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": action}}]}}
        rows.extend([{"event": "generation_intent", "messages": copy.deepcopy(messages),
                      "seed": 17 + len(prefix_records or []) + action_index, "prompt_tokens": 100},
                     {"event": "generation_result", "response": response}])
        cost["generations"] += 1
        cost["prompt_tokens"] += 100
        cost["completion_tokens"] += 1
        if returned_unused and action_index == len(actions) - 1:
            break
        observation = ({"ok": False, "error_type": "ObservedAPIError", "status_code": 422,
                        "message": "Synthetic confirmed HTTP4xx"}
                       if phase == "first_attempt" and not no_boundary and action_index == 0 else
                       {"ok": True, "result": {"fixture": True}})
        for name in ("action_attempts", "api_invocations", "upstream_requests"):
            cost[name] += 1
        rows.extend([{"event": "dispatch_intent", "action": action},
                     {"event": "dispatch_result", "reply": {**observation,
                        "cost": {key: cost[key] for key in ("action_attempts", "api_invocations", "upstream_requests")},
                        "freshness": {"outcome_unknown": False, "unknown_cost": False}}}])
        messages.extend([{"role": "assistant", "content": action},
                         {"role": "user", "content": _encode(observation).decode("ascii")}])
    outcome = {"status": "model_reported_complete" if complete and not returned_unused else "incomplete",
               "stop_reason": "supervisor_complete" if complete and not returned_unused else "wall_seconds",
               "cost": cost, "outcome_unknown": False, "unknown_cost": False}
    rows.append({"event": "actor_stop", "outcome": copy.deepcopy(outcome)})
    receipt = {"task_id": task, "status": "saved_and_closed", "is_policy_episode": True,
               "worker_exit_code": 0, "actor_outcome": outcome, "elapsed_seconds": 2,
               "replay_outcome": None if phase == "first_attempt" else {
                   "status": "matched", "replay_actions_attempted": len(prefix_records),
                   "replay_generations": 0, "replay_prompt_tokens": 0, "replay_completion_tokens": 0}}
    assert _digest(metadata) == context["metadata_sha256"]
    return receipt, rows


class FakeHeartbeat:
    def __init__(self, runtime):
        self.runtime = runtime

    def check(self):
        if self.runtime.fail == "heartbeat":
            raise RuntimeError("Synthetic heartbeat failure")

    def stop(self):
        self.runtime.calls.append(("heartbeat_stop",))
        self.check()


class FakeRuntime:
    def __init__(self, *, fail=None, no_boundary=False, context_overflow=False,
                 complete=True, target_repetition=False, generation_count=6,
                 returned_unused=False, help_indices=None):
        self.fail, self.no_boundary, self.context_overflow = fail, no_boundary, context_overflow
        self.complete, self.target_repetition, self.generation_count = complete, target_repetition, generation_count
        self.returned_unused = returned_unused
        self.help_indices = set(range(12)) if help_indices is None else set(help_indices)
        self.clock, self.calls, self.files, self.units, self.evidence = 0, [], {}, {}, {}
        self.cap_triggered = False
        self.job, self.stopped, self.attempted, self.begun = None, False, False, False
        self.plan = {"id": registration.ID, "status": "registered", "task_ids": list(registration.TASK_IDS),
                     "model_arm": "B3", "model_id": registration.MODEL_ID, "automatic_retry": False,
                     "private_run_root": "/synthetic/private", "design": {"review": {"hint": HINT}},
                     **{key: False for key in ("training_authorized", "writer_authorized",
                                              "utility_authorized", "dev_or_test_authorized")}}

    def now(self):
        return self.clock

    def epoch(self):
        return 100000 + self.clock

    def wait(self, seconds):
        self.calls.append(("wait", seconds))
        self.clock += seconds

    def owner(self):
        return "b" * 32

    def load_plan(self, sha):
        self.calls.append(("private_registration_gate", sha))
        if self.fail == "registration":
            raise ValueError("Synthetic gate failure")
        return copy.deepcopy(self.plan)

    def begin(self, plan, sha, slots):
        self.calls.append(("begin", copy.deepcopy(slots)))
        if self.begun:
            raise FileExistsError("Synthetic exclusive study intent exists")
        self.begun = True

    def record(self, value):
        self.calls.append(("record", copy.deepcopy(value)))
        if self.fail == "journal":
            raise OSError("Synthetic journal failure")

    def write(self, name, value):
        path = "/synthetic/private/" + name
        if path in self.files:
            raise FileExistsError("Synthetic exclusive output exists")
        self.files[path] = _encode(value) + b"\n"
        return {"path": path, "sha256": hashlib.sha256(self.files[path]).hexdigest()}

    def read(self, path):
        return self.files[str(path)]

    def context_rows(self, plan):
        rows = []
        for task in registration.TASK_IDS:
            metadata = {"instruction": "Synthetic " + task, "supervisor": {},
                        "app_descriptions": {"synthetic": "CPU fixture public API"}}
            rows.append({"task_id": task, "metadata_sha256": _digest(metadata),
                         "messages_sha256": _message_payload(build_actor_messages(metadata))[1],
                         "input_tokens": 100, "initial_fits": True})
        return rows

    def monitor(self, *, empty_port=False):
        self.calls.append(("monitor", empty_port))
        if self.fail == "monitor":
            raise RuntimeError("Synthetic monitor failure")

    def queue(self, operation, job, known_pids=()):
        self.calls.append(("queue", operation, job["batch_index"], tuple(known_pids)))
        if operation == "enqueue":
            self.job, self.stopped, self.started_epoch = copy.deepcopy(job), False, self.epoch()
        if operation == "stop":
            self.stopped = True
        waiting = self.fail == "admission_timeout"
        info = None if waiting else {"status": "finished" if self.stopped else "running",
            "started_epoch": self.started_epoch, "finished_epoch": self.epoch() if self.stopped else None}
        return {"owned": self.fail != "ownership", "monitor_live": True, "epoch": self.epoch(),
                "admission_cancelled": self.stopped, "released": self.stopped and not waiting and self.fail != "release",
                "job": info, "model_pids": [] if self.stopped or waiting else [123]}

    def heartbeat(self, job):
        self.calls.append(("heartbeat_start", job["batch_index"]))
        return FakeHeartbeat(self)

    def ready(self, job, state):
        self.calls.append(("ready", job["batch_index"]))
        if self.fail == "admission_timeout":
            self.clock += 86400
            return None
        return {"backend": {"arm": "B3"}, "identity": [123, "start"]}

    def launch(self, kind, unit, arguments, *, log_name):
        self.calls.append(("launch", kind, unit, list(arguments)))
        spec = {"kind": kind, "unit": unit, "arguments": list(arguments), "polls": 0}
        self.units[unit] = spec
        if kind == "episode":
            config = _decode(self.read(arguments[arguments.index("--config") + 1]))
            prefix = None if config["prefix_binding"] is None else _decode(self.read(config["prefix_binding"]["path"]))
            index, phase = config["source_index"], config["phase"]
            self.calls.append(("episode", index, phase, copy.deepcopy(config)))
            receipt, journal = _fixture_episode(index, phase, config["initial_context"], prefix,
                complete=self.complete, no_boundary=self.no_boundary, generation_count=self.generation_count,
                returned_unused=self.returned_unused, target_repetition=self.target_repetition)
            if self.fail == "replay" and phase != "first_attempt":
                receipt["replay_outcome"]["status"] = "observation_mismatch"
            if self.fail == "episode_unknown":
                receipt["status"] = "unknown"
            self.evidence[config["run_dir"]] = {"receipt": receipt, "journal": journal,
                "receipt_binding": {"path": config["run_dir"] + "/execution-receipt.json", "sha256": _digest(receipt)}}
            spec["index"], spec["phase"] = index, phase
        return spec

    def unit_status(self, spec):
        spec["polls"] += 1
        if self.fail == "GPU_cap" and spec["kind"] == "episode" and not self.cap_triggered:
            self.clock += 7080
            self.cap_triggered = True
        failed = self.fail == "unit_exit" or (self.fail == "scorer" and spec["kind"] == "scorer")
        return {"verified": True, "running": False, "main_pid": 0,
                "success": not failed, "exit_code": 1 if failed else 0}

    def stop_unit(self, spec):
        self.calls.append(("stop_unit", spec["unit"]))
        return {"verified": self.fail != "unit_cleanup"}

    def artifacts(self, path):
        return copy.deepcopy(self.evidence[str(path)])

    def cleanup_worker(self, run_dir, task_id, experiment):
        self.calls.append(("cleanup_worker", str(run_dir), task_id, experiment))
        return {"verified": self.fail != "worker_cleanup"}

    def create_score_directory(self, path):
        self.calls.append(("mkdir_score_0700", str(path)))

    def score_result(self, path):
        name = Path(path).name
        index = int(name.split("-")[1])
        phase = name.split("-")[2]
        run = "/synthetic/private/" + name.replace("score-", "unit-", 1)
        success = phase == "review" and index in self.help_indices
        return {"status": "scored", "success": success, "is_policy_episode": True,
                "receipt_sha256": self.evidence[run]["receipt_binding"]["sha256"]}

    def count(self, messages):
        self.calls.append(("CPU_count", copy.deepcopy(messages)))
        self.clock += 1
        if self.fail == "count":
            raise RuntimeError("Synthetic CPU count failure")
        return 15361 if self.context_overflow and any(message["content"] == HINT for message in messages) else 100

    def never_started(self, job, state):
        return state["job"] is None and state["admission_cancelled"] and not state["model_pids"]

    def close(self):
        self.calls.append(("close",))


def test_unit_commands_bound_parent_and_separate_scorer():
    root = Path("/synthetic/workspace")
    unit = "wll-native-source-00-first_attempt-" + "a" * 32 + ".service"
    command = pilot.unit_command("episode", unit, ["--config", "/synthetic/config.json",
                                                   "--config-sha256", "b" * 64],
                                 root=root, log_path="/synthetic/private/episode.log")
    assert "--property=RuntimeMaxSec=900" in command
    assert "--property=MemoryMax=6G" in command
    assert "--property=CPUQuota=400%" in command
    assert "--property=TasksMax=256" in command
    assert "--property=NoNewPrivileges=yes" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=RemainAfterExit=yes" in command
    assert "--property=RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6" in command
    assert "scripts.native_source_episode" in " ".join(command)
    scorer = pilot.unit_command("scorer", unit.replace("first_attempt", "score-first_attempt"),
                                ["--receipt", "/synthetic/execution-receipt.json",
                                 "--receipt-sha256", "c" * 64,
                                 "--parent-unit-name", unit, "--out-dir", "/synthetic/score"],
                                root=root, log_path="/synthetic/private/scorer.log")
    assert "--property=RuntimeMaxSec=120" in scorer
    assert "--property=MemoryMax=3G" in scorer
    assert "--property=CPUQuota=200%" in scorer
    assert "--property=TasksMax=128" in scorer
    assert "--property=RestrictAddressFamilies=AF_UNIX" in scorer


def test_default_task_schedule_has_twelve_fixed_sources():
    slots = pilot.initial_slots()
    assert [row["task_id"] for row in slots] == list(registration.TASK_IDS)
    assert all(row["first_attempt"]["status"] == "unstarted" for row in slots)
    assert all(row["control"]["status"] == row["review"]["status"] == "unstarted" for row in slots)


def test_full_twelve_source_interleaving_balanced_order_and_three_serial_leases():
    runtime = FakeRuntime()
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert result["data_feasibility_gate"] == {"status": "passed", "complete_help_scenarios": 12,
                                               "distinct_student_targets": 61, "source_denominator": 12}
    scheduled = [(row[1], row[2]) for row in runtime.calls if row[0] == "episode"]
    assert scheduled == [(index, phase) for index in range(12) for phase in (
        ("first_attempt", "control", "review") if index % 2 == 0 else ("first_attempt", "review", "control"))]
    assert [row[2] for row in runtime.calls if row[:2] == ("queue", "enqueue")] == [0, 1, 2]
    assert [row[2] for row in runtime.calls if row[:2] == ("queue", "stop")] == [0, 1, 2]
    queue_events = [row for row in runtime.calls if row[0] == "queue" and row[1] in {"enqueue", "stop"}]
    assert [row[1] for row in queue_events] == ["enqueue", "stop"] * 3
    assert runtime.calls[0] == ("private_registration_gate", SHA)
    assert all(slot["first_attempt"]["model_reported_complete"] for slot in result["slots"])
    assert len(result["costs"]["episodes"]) == len(result["costs"]["scorers"]) == 36
    assert len(result["costs"]["paired_CPU_counts"]) == 12
    assert all(cost["CPU_calls"] == 2 for cost in result["costs"]["paired_CPU_counts"])
    assert sum(cost["generations"] for cost in result["costs"]["episodes"]) == 168
    assert sum(cost["unreturned_generations"] for cost in result["costs"]["episodes"]) == 0
    assert all(result[key] == 0 for key in ("new_qualified_experiences", "writer_calls", "training_jobs",
                                          "utility_or_dev_or_test_calls"))
    for row in runtime.calls:
        if row[0] == "episode":
            config = row[3]
            assert set(config) == {"plan_sha256", "source_index", "phase", "run_dir", "job",
                                   "expected_backend", "prefix_binding", "initial_context"}
            assert config["expected_backend"]["arm"] == "B3"
            assert config["job"]["max_runtime_seconds"] == 7200
            assert config["source_index"] // 4 == config["job"]["batch_index"]
    # Both counts precede every source's first suffix launch, including odd review-first items.
    for index in range(12):
        suffix = next(n for n, row in enumerate(runtime.calls) if row[:2] == ("episode", index)
                      and row[2] != "first_attempt")
        previous_source = max(n for n, row in enumerate(runtime.calls[:suffix]) if row[:3] == ("episode", index, "first_attempt"))
        assert sum(row[0] == "CPU_count" for row in runtime.calls[previous_source:suffix]) == 2


def test_score_directories_and_parent_stop_precede_scorer_and_scores_are_separate():
    runtime = FakeRuntime(no_boundary=True)
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert all(row["pair_class"] == "ineligible_no_boundary" for row in result["slots"])
    assert sum(row[0] == "episode" for row in runtime.calls) == 12
    assert not any(row[0] == "CPU_count" for row in runtime.calls)
    assert result["data_feasibility_gate"]["status"] == "insufficient"
    for position, row in enumerate(runtime.calls):
        if row[:2] == ("launch", "scorer"):
            assert any(item[0] == "mkdir_score_0700" for item in runtime.calls[:position])
            arguments = row[3]
            assert "--parent-unit-name" in arguments and "--receipt-sha256" in arguments
            assert arguments[arguments.index("--receipt-sha256") + 1] != ""


def test_either_initial_context_overflow_abstains_both_arms_without_sampling():
    runtime = FakeRuntime(context_overflow=True)
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert sum(row[0] == "episode" for row in runtime.calls) == 12
    assert sum(row[0] == "CPU_count" for row in runtime.calls) == 24
    assert all(row["pair_class"] == "ineligible_initial_context" for row in result["slots"])
    assert all(row["control"]["status"] == row["review"]["status"] == "known_ineligible" for row in result["slots"])


@pytest.mark.parametrize("failure", ["episode_unknown", "replay", "count", "scorer", "unit_exit",
                                      "worker_cleanup", "unit_cleanup", "heartbeat", "release"])
def test_infrastructure_failure_stops_remaining_sources_and_never_retries(failure):
    runtime = FakeRuntime(fail=failure)
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "unknown"
    assert result["data_feasibility_gate"]["status"] == "not_evaluable"
    remaining_start = 4 if failure == "release" else 1
    assert all(slot["first_attempt"]["status"] == "unstarted" for slot in result["slots"][remaining_start:])
    assert len([row for row in runtime.calls if row[:2] == ("queue", "enqueue")]) == 1
    assert any(row[:2] == ("queue", "stop") for row in runtime.calls)
    assert not any(row[0] == "episode" and row[1] >= remaining_start for row in runtime.calls)
    assert runtime.calls[-1] == ("close",)


def test_known_incomplete_budget_failure_continues_full_fixed_schedule():
    runtime = FakeRuntime(complete=False)
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert len([row for row in runtime.calls if row[0] == "episode"]) == 36
    assert all(row["pair_class"] == "ineligible_incomplete_pair" for row in result["slots"])
    assert result["candidate_student_target_sha256"] == []


def test_returned_generation_without_dispatch_is_known_and_charged():
    runtime = FakeRuntime(returned_unused=True)
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert all(row["first_attempt"]["status"] == "known" for row in result["slots"])
    assert all(cost["generation_results"] == cost["generations"] for cost in result["costs"]["episodes"])
    assert all(cost["prompt_tokens"] == cost["generations"] * 100 for cost in result["costs"]["episodes"])
    assert all(row["pair_class"] == "ineligible_incomplete_pair" for row in result["slots"])


def test_unstarted_generation_intent_at_known_wall_stop_does_not_become_unknown():
    runtime = FakeRuntime(no_boundary=True)
    row = runtime.context_rows(runtime.plan)[0]
    receipt, events = _fixture_episode(0, "first_attempt", row, None, no_boundary=True, returned_unused=True)
    # No dispatch and no actual generate: the deadline expired after durable intent.
    events = [event for event in events if event["event"] != "generation_result"]
    receipt["actor_outcome"]["cost"].update(generations=0, prompt_tokens=0, completion_tokens=0)
    events[-1]["outcome"] = copy.deepcopy(receipt["actor_outcome"])
    _metadata, records, targets = pilot.actor_evidence(receipt, events, expected_context=row)
    assert records == targets == []


@pytest.mark.parametrize("runtime,expected", [
    (FakeRuntime(help_indices=range(3)), "insufficient"),
    (FakeRuntime(help_indices=range(4)), "passed"),
    (FakeRuntime(help_indices=range(4), generation_count=5), "insufficient"),
    (FakeRuntime(help_indices=range(4), target_repetition=True), "insufficient"),
])
def test_four_scenarios_and_twenty_distinct_student_targets_both_required(runtime, expected):
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "complete"
    assert result["data_feasibility_gate"]["status"] == expected
    assert result["data_feasibility_gate"]["source_denominator"] == 12


def test_student_projection_preserves_full_history_and_deduplicates_canonical_actions():
    runtime = FakeRuntime()
    row = runtime.context_rows(runtime.plan)[0]
    source, source_rows = _fixture_episode(0, "first_attempt", row, None)
    metadata, records, _generations = pilot.actor_evidence(source, source_rows, expected_context=row)
    prefix = pilot.select_prefix(metadata, records)
    messages = build_actor_messages(metadata, prefix_records=prefix["prefix_records"], review_hint=HINT)
    context = {"metadata_sha256": row["metadata_sha256"], "messages_sha256": _message_payload(messages)[1], "input_tokens": 100}
    receipt, events = _fixture_episode(0, "review", context, prefix)
    _metadata, _records, generations = pilot.actor_evidence(receipt, events, expected_context=context,
        prefix_records=prefix["prefix_records"], review_hint=HINT)
    targets = pilot.student_targets(generations, prefix_length=1, hint=HINT)
    assert len(targets) == 6
    for target, generation in zip(targets, generations, strict=True):
        expected = copy.deepcopy(generation["intent"]["messages"])
        del expected[4]
        assert target["student_messages"] == expected
    duplicate = copy.deepcopy(generations[0])
    parsed = json.loads(duplicate["action"])
    alternate = json.dumps(parsed, indent=2)
    duplicate["action"] = duplicate["response"]["text"] = alternate
    duplicate["response"]["raw_reply"]["choices"][0]["message"]["content"] = alternate
    assert len(pilot.student_targets([generations[0], duplicate], prefix_length=1, hint=HINT)) == 1
    for mutate in (lambda g: g["reply"].update(ok=False),
                   lambda g: g["response"].update(format_error="synthetic"),
                   lambda g: g["response"]["raw_reply"]["choices"][0].update(finish_reason="length"),
                   lambda g: g["response"].update(text="different")):
        invalid = copy.deepcopy(generations[0])
        mutate(invalid)
        assert pilot.student_targets([invalid], prefix_length=1, hint=HINT) == []


def test_wait_budget_is_separate_and_unstarted_slots_are_preserved():
    runtime = FakeRuntime(fail="admission_timeout")
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "stopped_admission_timeout"
    assert all(row["first_attempt"]["status"] == "unstarted" for row in result["slots"])
    assert result["costs"]["GPU_jobs"][0]["release"]["GPU_seconds"] == 0
    assert result["costs"]["GPU_jobs"][0]["admission_wait_seconds"] >= 86400


def test_GPU_cap_stops_active_episode_and_retains_later_sources():
    runtime = FakeRuntime(fail="GPU_cap")
    result = pilot.run_pilot(SHA, _runtime=runtime)
    assert result["status"] == "stopped_GPU_cap"
    assert all(row["first_attempt"]["status"] == "unstarted" for row in result["slots"][1:])
    assert any(row[0] == "cleanup_worker" for row in runtime.calls)


def test_exclusive_intent_prevents_resume_and_registration_failure_prevents_enqueue():
    runtime = FakeRuntime(no_boundary=True)
    assert pilot.run_pilot(SHA, _runtime=runtime)["status"] == "complete"
    with pytest.raises(pilot.NativeSourcePilotError):
        pilot.run_pilot(SHA, _runtime=runtime)
    forbidden = FakeRuntime(fail="registration")
    with pytest.raises(pilot.NativeSourcePilotError):
        pilot.run_pilot(SHA, _runtime=forbidden)
    assert not any(row[0] == "queue" for row in forbidden.calls)


def test_heartbeat_pulses_while_main_thread_is_blocked():
    runtime = FakeRuntime()
    job = {"batch_index": 0}
    runtime.queue("enqueue", job)
    pulse = pilot._Heartbeat(runtime, job, interval=0.01).start()
    try:
        threading.Event().wait(0.055)
        pulse.check()
        assert len([row for row in runtime.calls if row[:2] == ("queue", "pulse")]) >= 2
    finally:
        pulse.stop()


def test_missing_receipt_costs_remain_unknown_with_journal_lower_bounds():
    costs = pilot._costs({}, [{"event": "generation_intent"},
                              {"event": "generation_result", "response": {"prompt_tokens": 99, "completion_tokens": 3}},
                              {"event": "generation_intent"}])
    assert all(costs[key] is None for key in pilot._COST_KEYS)
    assert costs["unknown_cost"] is True and costs["counts_complete"] is False
    assert costs["generation_intents"] == 2 and costs["generation_results"] == 1
    assert costs["unreturned_generations"] is None
    assert costs["unreturned_generation_intents_upper_bound"] == 1
    assert costs["journal_returned_prompt_tokens_lower_bound"] == 99
    assert costs["journal_returned_completion_tokens_lower_bound"] == 3


def test_failed_scorer_count_and_release_keep_attempted_cost_entries():
    scored = pilot.run_pilot(SHA, _runtime=FakeRuntime(fail="scorer"))
    assert len(scored["costs"]["scorers"]) == 1
    assert scored["costs"]["scorers"][0]["attempted"] is True
    assert scored["costs"]["scorers"][0]["result_validated"] is False
    counted = pilot.run_pilot(SHA, _runtime=FakeRuntime(fail="count"))
    assert len(counted["costs"]["paired_CPU_counts"]) == 1
    assert counted["costs"]["paired_CPU_counts"][0]["CPU_calls"] == 1
    assert counted["costs"]["paired_CPU_counts"][0]["counts_complete"] is False
    released = pilot.run_pilot(SHA, _runtime=FakeRuntime(fail="release"))
    assert len(released["costs"]["GPU_jobs"]) == 1
    assert released["costs"]["GPU_jobs"][0]["release"]["verified"] is False
    assert released["costs"]["GPU_jobs"][0]["release"]["GPU_seconds"] is None
    assert released["costs"]["GPU_jobs"][0]["unknown_cost"] is True


def test_attempted_unreturned_generation_is_charged_and_remains_unknown():
    receipt = {"actor_outcome": {"status": "unknown", "outcome_unknown": False, "unknown_cost": True,
        "cost": {"generations": 1, "prompt_tokens": 0, "completion_tokens": 0,
                 "action_attempts": 0, "api_invocations": 0, "upstream_requests": 0}}}
    costs = pilot._costs(receipt, [{"event": "generation_intent"}, {"event": "generation_failure"}])
    assert costs["generations"] == costs["unreturned_generations"] == 1
    assert costs["unknown_cost"] is True


def test_dispatching_status_is_a_bounded_normal_readiness_state():
    class DispatchingRuntime(FakeRuntime):
        def queue(self, operation, job, known_pids=()):
            result = super().queue(operation, job, known_pids)
            if operation == "enqueue":
                result["job"]["status"] = "dispatching"
            return result

        def ready(self, job, state):
            return None if state["job"]["status"] == "dispatching" else super().ready(job, state)

    runtime = DispatchingRuntime(no_boundary=True)
    assert pilot.run_pilot(SHA, _runtime=runtime)["status"] == "complete"
    assert any(row[:2] == ("wait", 5) for row in runtime.calls)
    assert len([row for row in runtime.calls if row[0] == "private_registration_gate"]) == 3


def test_post_launch_journal_failure_stops_the_owned_unit(tmp_path, monkeypatch):
    runtime = pilot._Runtime(tmp_path)
    runtime.private = tmp_path
    runtime.fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    stopped = []

    def record(value):
        if value["event"] == "unit_launch_return":
            raise OSError("Synthetic durable post-launch journal failure")

    runtime.record = record
    monkeypatch.setattr(pilot.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    runtime.stop_unit = lambda spec: stopped.append(spec) or {"verified": True}
    unit = "wll-native-source-00-first_attempt-" + "a" * 32 + ".service"
    try:
        with pytest.raises(OSError, match="post-launch journal failure"):
            runtime.launch("episode", unit, ["--help"], log_name="post-launch-failure.log")
        assert len(stopped) == 1
        assert stopped[0] is runtime.units[unit]
        assert stopped[0]["unit"] == unit
    finally:
        runtime.close()


@pytest.mark.parametrize("receipt_available", [False, True])
def test_failed_episode_recovers_complete_journal_prefix_without_receipt(receipt_available):
    class PartialJournalRuntime(FakeRuntime):
        def artifacts(self, path):
            raise FileNotFoundError("Synthetic missing or unreadable receipt")

        def read(self, path):
            path = str(path)
            if path.endswith("/controller.jsonl"):
                return b"\n".join(_encode(row) for row in [
                    {"event": "generation_intent"},
                    {"event": "generation_result", "response": {"prompt_tokens": 77, "completion_tokens": 4}},
                    {"event": "dispatch_intent"},
                ]) + b'\n{"event":"dispatch_result"'
            if path.endswith("/execution-receipt.json"):
                if not receipt_available:
                    raise FileNotFoundError("Synthetic missing receipt")
                return _encode(self.evidence[path.removesuffix("/execution-receipt.json")]["receipt"]) + b"\n"
            return super().read(path)

    result = pilot.run_pilot(SHA, _runtime=PartialJournalRuntime(fail="unit_exit"))
    assert result["status"] == "unknown"
    costs = result["costs"]["episodes"][0]
    assert costs["generation_intents"] == costs["generation_results"] == 1
    assert costs["dispatch_intents"] == 1 and costs["dispatch_results"] == 0
    assert costs["journal_returned_prompt_tokens_lower_bound"] == 77
    assert costs["journal_returned_completion_tokens_lower_bound"] == 4
    assert costs["journal_complete"] is False and costs["durable_journal_prefix_bytes"] > 0
    assert costs["unknown_cost"] is True and costs["counts_complete"] is False
    if not receipt_available:
        assert all(costs[key] is None for key in pilot._COST_KEYS)
