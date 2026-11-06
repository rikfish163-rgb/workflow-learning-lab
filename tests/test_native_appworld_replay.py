import copy
import json

import pytest

from scripts.native_appworld_replay import MAX_ACTION_BYTES, replay_prefix

METADATA = {
    "instruction": "Synthetic replay instruction",
    "supervisor": {"first_name": "Synthetic"},
    "app_descriptions": {"phone": "Synthetic phone app"},
}
ACTION = '{"name":"phone__show_profile","arguments":{}}'
ZERO = {"action_attempts": 0, "api_invocations": 0, "upstream_requests": 0}


def observation(result=None, *, error_type=None, message="Synthetic rejection", status_code=404):
    if error_type is None:
        return {"ok": True, "result": result}
    value = {"ok": False, "error_type": error_type, "message": message}
    if error_type == "ObservedAPIError":
        value["status_code"] = status_code
    return value


def prefix(*observations):
    return [{"action": ACTION, "observation": value} for value in observations]


class Fixture:
    def __init__(self, observations=(), *, metadata=None):
        self.observations = list(copy.deepcopy(observations))
        self.metadata = copy.deepcopy(METADATA if metadata is None else metadata)
        self.events = []
        self.requests = []
        self.cost = ZERO.copy()
        self.time = 10.0
        self.change_reply = None

    def clock(self):
        return self.time

    def record(self, event):
        self.events.append(copy.deepcopy(event))

    def reply(self, value):
        return {
            **copy.deepcopy(value),
            "cost": self.cost.copy(),
            "freshness": {
                "last_successful_save_through_attempt": None,
                "current_state_saved": False,
                "score_eligible": False,
                "outcome_unknown": False,
                "unknown_cost": False,
            },
        }

    def request(self, request, *, timeout):
        self.requests.append((copy.deepcopy(request), timeout))
        self.events.append({"event": "request_effect", "request": copy.deepcopy(request)})
        if request["op"] == "metadata":
            result = self.reply(observation(self.metadata))
        else:
            self.cost["action_attempts"] += 1
            value = self.observations.pop(0)
            if isinstance(value, Exception):
                raise value
            if value.get("error_type") != "ToolDispatchError":
                self.cost["api_invocations"] += 1
                self.cost["upstream_requests"] += 1
            result = self.reply(value)
        if self.change_reply is not None:
            return self.change_reply(request, result)
        return result

    def run(self, records=None, **changes):
        options = {
            "source_metadata": METADATA,
            "prefix_records": prefix(*self.observations) if records is None else records,
            "record": self.record,
            "deadline": 100.0,
            "max_actions": 10,
            "max_api_requests": 10,
            "clock": self.clock,
        }
        options.update(changes)
        return replay_prefix(self, **options)


def test_matches_normal_and_recoverable_errors_with_complete_private_journal():
    values = [
        observation({"nested": [None, True, 1, 1.0, "完整 synthetic value"]}),
        observation(error_type="ToolDispatchError"),
        observation(error_type="ObservedAPIError", message="Synthetic HTTP 404"),
        observation(["last"]),
    ]
    records = prefix(*values)
    fixture = Fixture(values)
    result = fixture.run(records)
    assert result["status"] == "matched"
    assert result["prefix_records"] == records
    assert result["metadata"] == METADATA
    assert result["worker_cost"] == {
        "action_attempts": 4, "api_invocations": 3, "upstream_requests": 3,
    }
    assert result["replay_actions_attempted"] == 4
    assert result["replay_api_invocations"] == 3
    assert result["replay_upstream_requests"] == 3
    assert result["replay_generations"] == result["replay_prompt_tokens"] == 0
    assert result["replay_completion_tokens"] == 0
    assert not result["outcome_unknown"] and not result["unknown_cost"]
    assert [request for request, _ in fixture.requests] == [
        {"op": "metadata"}, *[{"op": "dispatch", "action": ACTION}] * 4,
    ]
    assert all(0 < timeout <= 90 for _, timeout in fixture.requests)
    events = [item["event"] for item in fixture.events]
    assert events.index("replay_metadata_intent") < events.index("request_effect")
    for index, event in enumerate(fixture.events):
        if event["event"] == "request_effect" and event["request"]["op"] == "dispatch":
            assert fixture.events[index - 1]["event"] == "replay_dispatch_intent"
    replies = [event["reply"] for event in fixture.events
               if event["event"] == "replay_dispatch_result"]
    assert len(replies) == 4
    assert all(set(reply) >= {"cost", "freshness"} for reply in replies)
    assert replies[2]["message"] == "Synthetic HTTP 404"
    assert fixture.events[-1]["event"] == "replay_stop"


def test_empty_prefix_still_requires_fresh_equal_metadata():
    fixture = Fixture()
    result = fixture.run([])
    assert result["status"] == "matched"
    assert result["prefix_records"] == []
    assert result["worker_cost"] == ZERO
    assert fixture.requests == [({"op": "metadata"}, 90.0)]


def test_metadata_mismatch_stops_before_dispatch_and_keeps_actual_private_metadata():
    metadata = {**METADATA, "instruction": "Other synthetic instruction"}
    fixture = Fixture([observation(1)], metadata=metadata)
    result = fixture.run()
    assert result["status"] == "observation_mismatch"
    assert result["stop_reason"] == "metadata_mismatch"
    assert result["metadata"] == metadata
    assert result["prefix_records"] == []
    assert result["worker_cost"] == ZERO
    assert len(fixture.requests) == 1
    assert not result["unknown_cost"]


@pytest.mark.parametrize("before,after", [
    (True, 1), (1, True), (1, 1.0), (False, 0), (None, "null"),
    ({"nested": [True]}, {"nested": [1]}), (-0.0, 0.0),
])
def test_canonical_observation_comparison_preserves_json_types(before, after):
    fixture = Fixture([observation(after), observation("must not dispatch")])
    result = fixture.run(prefix(observation(before), observation("must not dispatch")))
    assert result["status"] == "observation_mismatch"
    assert result["stop_reason"] == "observation_mismatch"
    assert result["prefix_records"] == []
    assert result["replay_actions_attempted"] == 1
    assert result["worker_cost"]["api_invocations"] == 1
    assert len(fixture.requests) == 2


def test_object_key_order_is_ignored_without_replacing_or_hashing_values():
    fixture = Fixture([observation({"z": 1, "a": {"second": 2, "first": 1}})])
    source = prefix(observation({"a": {"first": 1, "second": 2}, "z": 1}))
    assert fixture.run(source)["status"] == "matched"


@pytest.mark.parametrize("event", ["replay_start", "replay_metadata_intent", "replay_dispatch_intent"])
def test_failed_durable_intent_blocks_following_effect(event):
    fixture = Fixture([observation(1)])

    def fail(value):
        fixture.record(value)
        if value["event"] == event:
            raise OSError("Synthetic private fsync failure")

    result = fixture.run(record=fail)
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "journal_failure"
    assert result["replay_actions_attempted"] == 0
    assert len(fixture.requests) == (1 if event == "replay_dispatch_intent" else 0)
    assert result["prefix_records"] == []


def test_failed_response_journal_prevents_continuation_and_marks_cost_unknown():
    fixture = Fixture([observation(1), observation(2)])

    def fail(event):
        if event["event"] == "replay_dispatch_result":
            raise OSError("Synthetic private fsync failure")
        fixture.record(event)

    result = fixture.run(record=fail)
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "journal_failure"
    assert result["replay_actions_attempted"] == 1
    assert result["outcome_unknown"] and result["unknown_cost"]
    assert len(fixture.requests) == 2


def test_transport_failure_counts_attempt_without_retry_or_assumed_api_cost():
    fixture = Fixture([RuntimeError("Synthetic private transport failure")])
    result = fixture.run(prefix(observation(1)))
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "worker_failure"
    assert result["replay_actions_attempted"] == 1
    assert result["worker_cost"] == ZERO
    assert result["replay_api_invocations"] == result["replay_upstream_requests"] == 0
    assert result["outcome_unknown"] and result["unknown_cost"]
    assert result["prefix_records"] == []
    assert len(fixture.requests) == 2
    assert any(event["event"] == "replay_dispatch_failure" for event in fixture.events)


@pytest.mark.parametrize("change", [
    lambda reply: reply.update(ok=1),
    lambda reply: reply.update(extra="private"),
    lambda reply: reply.pop("freshness"),
    lambda reply: reply["cost"].update(action_attempts=True),
    lambda reply: reply["cost"].update(action_attempts=0),
    lambda reply: reply["cost"].update(api_invocations=0),
    lambda reply: reply["cost"].update(upstream_requests=0),
    lambda reply: reply["cost"].update(upstream_requests=-1),
    lambda reply: reply["cost"].update(upstream_requests=11),
    lambda reply: reply["freshness"].update(current_state_saved=True),
    lambda reply: reply["freshness"].update(last_successful_save_through_attempt=0),
    lambda reply: reply["freshness"].update(outcome_unknown=True),
    lambda reply: reply["freshness"].update(unknown_cost=True),
    lambda reply: reply.update(result={"invalid": float("nan")}),
    lambda reply: reply.update(result={"invalid": "\ud800"}),
    lambda reply: reply.update(result={1: "non-string key"}),
    lambda reply: reply.update(result=("tuple",)),
])
def test_malformed_worker_reply_is_unknown_and_never_continues(change):
    fixture = Fixture([observation(1), observation(2)])

    def change_dispatch(request, reply):
        if request["op"] == "dispatch":
            change(reply)
        return reply

    fixture.change_reply = change_dispatch
    result = fixture.run()
    assert result["status"] == "unknown"
    assert result["prefix_records"] == []
    assert result["outcome_unknown"] and result["unknown_cost"]
    assert result["replay_actions_attempted"] == 1
    assert len(fixture.requests) == 2


@pytest.mark.parametrize("change", [
    lambda reply: reply["cost"].update(action_attempts=1),
    lambda reply: reply["cost"].update(api_invocations=1),
    lambda reply: reply["cost"].update(upstream_requests=1),
    lambda reply: reply["freshness"].update(current_state_saved=True),
    lambda reply: reply["freshness"].update(last_successful_save_through_attempt=0),
    lambda reply: reply.update(result={**METADATA, "supervisor": {"password": "synthetic"}}),
])
def test_metadata_requires_zero_cost_and_unsaved_fresh_state(change):
    fixture = Fixture([observation(1)])

    def change_metadata(request, reply):
        if request["op"] == "metadata":
            change(reply)
        return reply

    fixture.change_reply = change_metadata
    result = fixture.run()
    assert result["status"] == "unknown"
    assert result["prefix_records"] == []
    assert len(fixture.requests) == 1


@pytest.mark.parametrize("error_type", ["WorkerStopped", "InvalidRPC", "ActionLimitError", "UnexpectedError"])
def test_unknown_or_control_error_is_never_an_actor_observation(error_type):
    fixture = Fixture([observation(error_type=error_type)])
    result = fixture.run(prefix(observation(1)))
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "worker_failure"
    assert result["outcome_unknown"] and result["unknown_cost"]
    assert result["prefix_records"] == []
    assert len(fixture.requests) == 2


def test_all_upstream_requests_and_error_costs_are_retained_and_never_reset():
    values = [observation(1), observation(error_type="ObservedAPIError"),
              observation(error_type="ToolDispatchError"), observation(2)]
    fixture = Fixture(values)

    def extra_upstream(request, reply):
        if request["op"] == "dispatch" and reply.get("error_type") != "ToolDispatchError":
            fixture.cost["upstream_requests"] += 1
            reply["cost"] = fixture.cost.copy()
        return reply

    fixture.change_reply = extra_upstream
    result = fixture.run(max_api_requests=6)
    assert result["status"] == "matched"
    assert result["worker_cost"] == {
        "action_attempts": 4, "api_invocations": 3, "upstream_requests": 6,
    }
    assert result["replay_upstream_requests"] == 6
    assert fixture.cost == result["worker_cost"]


def test_api_budget_stops_before_next_dispatch_including_boundary_rejections():
    fixture = Fixture([observation(1), observation(error_type="ToolDispatchError")])
    result = fixture.run(max_api_requests=1)
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "max_api_requests"
    assert result["replay_actions_attempted"] == 1
    assert result["worker_cost"]["upstream_requests"] == 1
    assert not result["unknown_cost"]
    assert len(fixture.requests) == 2


def test_decreasing_upstream_counter_on_later_action_is_unknown():
    fixture = Fixture([observation(1), observation(2), observation(3)])

    def decrease(request, reply):
        if request["op"] == "dispatch" and reply["cost"]["action_attempts"] == 2:
            reply["cost"]["upstream_requests"] = 0
        return reply

    fixture.change_reply = decrease
    result = fixture.run()
    assert result["status"] == "unknown"
    assert result["worker_cost"] == {
        "action_attempts": 1, "api_invocations": 1, "upstream_requests": 1,
    }
    assert result["replay_actions_attempted"] == 2
    assert len(fixture.requests) == 3


@pytest.mark.parametrize("action", [
    "not JSON or executable Python", '{"name":"a__b","name":"a__c","arguments":{}}',
    '{"name":"a__b","arguments":{"value":NaN}}',
])
def test_boundary_rejected_raw_action_is_replayed_unchanged_as_data(action):
    value = observation(error_type="ToolDispatchError")
    fixture = Fixture([value])
    records = [{"action": action, "observation": value}]
    assert fixture.run(records)["status"] == "matched"
    assert fixture.requests[1][0] == {"op": "dispatch", "action": action}


@pytest.mark.parametrize("action", ["x" * (MAX_ACTION_BYTES + 1), "é" * (MAX_ACTION_BYTES // 2 + 1), "\ud800"])
def test_invalid_utf8_or_overbyte_source_action_is_rejected_before_worker(action):
    fixture = Fixture()
    with pytest.raises(ValueError):
        fixture.run([{"action": action, "observation": observation(error_type="ToolDispatchError")}])
    assert fixture.requests == []


@pytest.mark.parametrize("records", [
    (), [{"action": ACTION}], [{"action": ACTION, "observation": {"ok": 1, "result": 1}}],
    [{"action": ACTION, "observation": {"ok": True, "result": float("inf")}}],
    [{"action": ACTION, "observation": observation(error_type="WorkerStopped")}],
    [{"action": ACTION, "observation": observation(error_type="ObservedAPIError", status_code=True)}],
    [{"action": '{"arguments":{},"arguments":{}}', "observation": observation(1)}],
])
def test_invalid_source_prefix_never_reaches_worker(records):
    fixture = Fixture()
    with pytest.raises(ValueError):
        fixture.run(records)
    assert fixture.requests == []


@pytest.mark.parametrize("changes", [
    {"max_actions": True}, {"max_actions": 0}, {"max_api_requests": 0},
    {"max_api_requests": 1.5}, {"deadline": float("nan")}, {"deadline": True},
    {"source_metadata": {**METADATA, "supervisor": {"private": "field"}}},
    {"max_actions": 1, "prefix_records": prefix(observation(1), observation(2))},
    {"max_api_requests": 1, "prefix_records": prefix(observation(1), observation(2))},
])
def test_invalid_trusted_inputs_fail_before_any_request(changes):
    fixture = Fixture()
    with pytest.raises(ValueError):
        fixture.run([], **changes)
    assert fixture.requests == []


def test_deadline_is_rechecked_after_durable_intent_before_dispatch():
    fixture = Fixture([observation(1)])

    def slow_record(event):
        fixture.record(event)
        if event["event"] == "replay_dispatch_intent":
            fixture.time = 100.0

    result = fixture.run(record=slow_record)
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "wall_seconds"
    assert result["replay_actions_attempted"] == 0
    assert len(fixture.requests) == 1


def test_late_reply_keeps_confirmed_cost_but_cannot_authorize_suffix():
    fixture = Fixture([observation(1), observation(2)])

    def late(request, reply):
        if request["op"] == "dispatch":
            fixture.time = 100.0
        return reply

    fixture.change_reply = late
    result = fixture.run()
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "wall_seconds"
    assert result["worker_cost"] == {
        "action_attempts": 1, "api_invocations": 1, "upstream_requests": 1,
    }
    assert result["prefix_records"] == []
    assert not result["unknown_cost"]
    assert len(fixture.requests) == 2


@pytest.mark.parametrize("bad_clock", [lambda: float("nan"), lambda: True, iter([10.0, 9.0]).__next__])
def test_invalid_or_backwards_clock_stops_without_dispatch(bad_clock):
    fixture = Fixture([observation(1)])
    result = fixture.run(clock=bad_clock)
    assert result["status"] == "unknown"
    assert result["stop_reason"] == "clock_failure"
    assert result["prefix_records"] == []
    assert result["replay_actions_attempted"] == 0


def test_large_observation_is_never_truncated_in_matching_or_private_records():
    text = "Synthetic full private observation " * 3000
    value = observation({"text": text, "tail": "end sentinel"})
    fixture = Fixture([value])
    result = fixture.run()
    assert result["status"] == "matched"
    assert result["prefix_records"][0]["observation"] == value
    actual = next(event["reply"] for event in fixture.events
                  if event["event"] == "replay_dispatch_result")
    assert actual["result"]["text"] == text
    assert json.loads(json.dumps(actual))["result"]["tail"] == "end sentinel"


def test_record_callback_cannot_mutate_source_or_replay_return_value():
    value = observation({"list": ["original"]})
    records = prefix(value)
    fixture = Fixture([value])

    def mutating_record(event):
        fixture.record(event)
        event.clear()

    result = fixture.run(records, record=mutating_record)
    assert result["status"] == "matched"
    assert result["prefix_records"] == records
    assert records == prefix(value)
