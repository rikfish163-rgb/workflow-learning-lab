"""Synthetic transcripts and fake model only; no task or model execution."""

import copy
import json

import pytest

from scripts.native_experience_writer_001 import (
    build_writer_messages,
    parse_candidate,
    write_once,
)

HINT = "Fixed synthetic review hint"
CANDIDATE = {"status": "candidate", "failure_mode": "Unconfirmed update",
             "applies_when": "A visible error leaves an intended update unconfirmed.",
             "avoid_when": "The earlier write succeeded or its outcome is unknown.",
             "rule": "Read the current state before choosing a minimal correction."}


def history():
    start = [{"role": "system", "content": "Synthetic actor system"},
             {"role": "user", "content": HINT},
             {"role": "assistant", "content": '{"name":"synthetic__read","arguments":{}}'},
             {"role": "user", "content": '{"ok":false,"error_type":"ObservedAPIError",'
              '"status_code":422,"message":"Synthetic visible error"}'},
             {"role": "user", "content": HINT}]
    records = [{"action": '{"name":"supervisor__complete_task","arguments":{}}',
                "observation": {"ok": True, "result": {"acknowledgement": "Synthetic final observation"}}}]
    return start, records


def messages():
    return build_writer_messages(*history(), prefix_length=1, fixed_hint=HINT)


class FakeModel:
    max_tokens = 128

    def __init__(self, *, count=100, text=None, finish_reason="stop", fail=None):
        self.count = count
        self.text = json.dumps(CANDIDATE) if text is None else text
        self.finish_reason, self.fail, self.calls = finish_reason, fail, []

    def check_health(self):
        self.calls.append("health")
        if self.fail == "health":
            raise RuntimeError("Synthetic private health body")
        return {"ready": True}

    def count_input_tokens(self, value):
        self.calls.append(("count", copy.deepcopy(value)))
        if self.fail == "count":
            raise RuntimeError("Synthetic private count body")
        return self.count

    def generate(self, value, seed):
        self.calls.append(("generate", copy.deepcopy(value), seed))
        if self.fail == "generation":
            raise RuntimeError("Synthetic private transport body")
        return {"text": self.text, "prompt_tokens": self.count, "completion_tokens": 80,
                "raw_reply": {"choices": [{"finish_reason": self.finish_reason,
                    "message": {"role": "assistant", "content": self.text}}]}}


def run(model=None, *, fail_event=None):
    model = FakeModel() if model is None else model
    events = []

    def record(event):
        if event["event"] == fail_event:
            raise OSError("Synthetic private persistence body")
        events.append(copy.deepcopy(event))

    return write_once(messages(), model, seed=5017, record=record), events, model


def test_full_history_includes_final_action_observation_and_removes_only_indexed_hint():
    start, records = history()
    records[0]["observation"]["result"]["long"] = "完整" * 5000
    before = copy.deepcopy((start, records))
    result = build_writer_messages(start, records, prefix_length=1, fixed_hint=HINT)
    payload = json.loads(result[1]["content"])
    assert set(payload) == {"history"}
    transcript = payload["history"]
    assert transcript[:4] == start[:4]
    assert transcript[-2] == {"role": "assistant", "content": records[0]["action"]}
    assert json.loads(transcript[-1]["content"]) == records[0]["observation"]
    assert transcript[1]["content"] == HINT
    assert len(transcript) == 6 and (start, records) == before


def test_maximum_prefix_and_twenty_four_suffix_records_are_preserved_without_truncation():
    start, records = history()
    start = start[:2] + start[2:4] * 54 + [start[-1]]
    records = [copy.deepcopy(records[0]) for _ in range(24)]
    for index, row in enumerate(records):
        row["observation"]["result"]["sequence"] = index
    before = copy.deepcopy((start, records))
    result = build_writer_messages(start, records, prefix_length=54, fixed_hint=HINT)
    transcript = json.loads(result[1]["content"])["history"]
    assert len(transcript) == 2 + 2 * 54 + 2 * 24
    assert transcript[:110] == start[:110]
    assert transcript[1]["content"] == HINT
    assert [json.loads(transcript[111 + 2 * index]["content"])["result"]["sequence"]
            for index in range(24)] == list(range(24))
    assert (start, records) == before
    with pytest.raises(ValueError, match="^native_experience_writer_invalid$"):
        build_writer_messages(start, records + [records[-1]], prefix_length=54, fixed_hint=HINT)


@pytest.mark.parametrize("change", ["wrong_hint", "wrong_prefix", "score_envelope"])
def test_transcript_requires_exact_hint_position_and_actor_observation_envelope(change):
    start, records = history()
    prefix_length = 1
    if change == "wrong_hint":
        start[-1]["content"] = "Different hint"
    elif change == "wrong_prefix":
        prefix_length = 2
    else:
        records[0]["observation"]["official_success"] = True
    with pytest.raises(ValueError, match="^native_experience_writer_invalid$"):
        build_writer_messages(start, records, prefix_length=prefix_length, fixed_hint=HINT)


def test_candidate_and_abstention_are_strict_data_without_content_certification():
    assert parse_candidate(json.dumps(CANDIDATE)) == CANDIDATE
    assert parse_candidate('{"status":"abstain","reason":"Insufficient visible evidence"}') == {
        "status": "abstain", "reason": "Insufficient visible evidence"}


@pytest.mark.parametrize("text", [
    '{"status":"abstain","status":"candidate","reason":"duplicate"}',
    json.dumps({**CANDIDATE, "score": True}),
    json.dumps({**CANDIDATE, "rule": True}),
    json.dumps({key: value for key, value in CANDIDATE.items() if key != "avoid_when"}),
    '{"status":"abstain","reason":NaN}',
    '{"status":"abstain","reason":""}',
])
def test_duplicate_keys_extra_fields_wrong_types_and_nonfinite_json_are_rejected(text):
    with pytest.raises(ValueError, match="^native_experience_writer_invalid$"):
        parse_candidate(text)


def test_one_returned_candidate_is_unvalidated_and_all_usage_is_charged():
    result, events, model = run()
    assert result["status"] == "candidate_unvalidated" and result["candidate"] == CANDIDATE
    assert result["content_certified"] is False and result["accepted"] is False
    assert result["training_authorized"] is False
    assert [call if type(call) is str else call[0] for call in model.calls] == ["health", "count", "generate"]
    assert result["cost"] == {"health_checks": 1, "CPU_count_attempts": 1, "CPU_counts_returned": 1,
        "generation_attempts": 1, "generation_results": 1,
        "returned_prompt_tokens_lower_bound": 100, "returned_completion_tokens_lower_bound": 80,
        "unknown_cost": False}
    assert next(e for e in events if e["event"] == "generation_intent")["max_new_tokens"] == 128
    assert events[-1]["event"] == "writer_stop"


def test_full_input_overflow_abstains_before_generation_and_keeps_cpu_count_attempt():
    result, _events, model = run(FakeModel(count=15361))
    assert result["status"] == "abstained" and result["reason"] == "context_overflow"
    assert len(model.calls) == 2
    assert result["cost"]["CPU_count_attempts"] == result["cost"]["CPU_counts_returned"] == 1
    assert result["cost"]["generation_attempts"] == 0 and not result["cost"]["unknown_cost"]


def test_exact_input_limit_is_allowed_without_changing_the_counted_messages():
    result, _events, model = run(FakeModel(count=15360))
    assert result["status"] == "candidate_unvalidated"
    assert model.calls[1][1] == model.calls[2][1] == messages()


@pytest.mark.parametrize(("text", "finish"), [
    (json.dumps(CANDIDATE), "length"), (json.dumps(CANDIDATE), "unknown"),
    ("malformed synthetic writer JSON", "stop"),
    ('{"status":"abstain","reason":"Insufficient evidence"}', "stop"),
])
def test_incomplete_malformed_and_abstained_outputs_keep_known_returned_cost(text, finish):
    result, _events, model = run(FakeModel(text=text, finish_reason=finish))
    assert result["status"] == "abstained" and result["candidate"] is None
    assert result["cost"]["returned_prompt_tokens_lower_bound"] == 100
    assert result["cost"]["returned_completion_tokens_lower_bound"] == 80
    assert result["cost"]["generation_results"] == 1 and not result["cost"]["unknown_cost"]
    assert len(model.calls) == 3


@pytest.mark.parametrize("stage", ["health", "count", "generation"])
def test_unreturned_operation_stays_unknown_without_echoing_error_bodies(stage):
    result, events, model = run(FakeModel(fail=stage))
    assert result["status"] == "unknown" and result["cost"]["unknown_cost"]
    assert result["candidate"] is None
    assert result["cost"]["generation_attempts"] == (stage == "generation")
    assert result["cost"]["generation_results"] == 0
    assert "private" not in json.dumps(result) and "private" not in json.dumps(events)
    assert len(model.calls) == {"health": 1, "count": 2, "generation": 3}[stage]


@pytest.mark.parametrize("event", ["generation_intent", "generation_result", "writer_stop"])
def test_journal_failure_is_unknown_and_cannot_discard_an_observed_response(event):
    result, _events, model = run(fail_event=event)
    assert result["status"] == "unknown" and result["reason"] == "journal_failure"
    assert result["cost"]["unknown_cost"] and result["candidate"] is None
    returned = event != "generation_intent"
    assert result["cost"]["generation_attempts"] == returned
    assert result["cost"]["generation_results"] == returned
    assert result["cost"]["returned_completion_tokens_lower_bound"] == (80 if returned else 0)
    assert len(model.calls) == (3 if returned else 2)


def test_wrong_generation_cap_is_rejected_before_health_or_count():
    model = FakeModel()
    model.max_tokens = 129
    with pytest.raises(ValueError, match="^native_experience_writer_invalid$"):
        write_once(messages(), model, seed=5017, record=lambda event: None)
    assert model.calls == []
