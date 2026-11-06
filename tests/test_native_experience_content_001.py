"""Synthetic local content review; no protected input, models or network."""

import copy
import json

import pytest

from scripts import native_experience_content_001 as content
from scripts.native_experience_writer_001 import SYSTEM as WRITER_SYSTEM

CANDIDATE = {"status": "candidate", "failure_mode": "An unconfirmed intended update",
             "applies_when": "The visible request failed and current state can be read.",
             "avoid_when": "A prior write succeeded or its outcome remains unknown.",
             "rule": "Read current state and choose a minimal correction."}


def writer_messages():
    history = [
        {"role": "system", "content": "Synthetic actor system"},
        {"role": "user", "content": json.dumps({"instruction": "Synthetic task",
            "supervisor": {"first_name": "SyntheticOwner"}, "app_descriptions": {}})},
        {"role": "assistant", "content": '{"name":"synthetic__login","arguments":{"password":"private-credential-xyz"}}'},
        {"role": "user", "content": '{"ok":true,"result":{"access_token":"private-access-xyz","object_id":12345}}'},
        {"role": "assistant", "content": '{"name":"supervisor__complete_task","arguments":{"answer":"Synthetic exact answer"}}'},
        {"role": "user", "content": '{"ok":true,"result":{"acknowledged":true}}'},
    ]
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": json.dumps({"history": history}, ensure_ascii=False)}]


def verdict(kind="plausible_pending_utility"):
    reasons = {"plausible_pending_utility": ["visible_conditional_support"],
               "reject": ["unsupported_claim"], "insufficient_evidence": ["insufficient_evidence"]}
    return {"verdict": kind, "reason_codes": reasons[kind],
            "evidence_indices": [2, 3] if kind == "plausible_pending_utility" else []}


def test_insertion_includes_all_fields_verbatim_with_fallibility_reminder():
    before = copy.deepcopy(CANDIDATE)
    rendered = content.insertion(CANDIDATE)
    assert all(value in rendered for key, value in CANDIDATE.items() if key != "status")
    assert "may be wrong" in rendered and CANDIDATE == before


def test_clear_structural_screen_returns_only_codes_hashes_and_limits():
    result = content.screen(writer_messages(), CANDIDATE)
    assert result["status"] == "passed" and result["reason_codes"] == []
    assert result["semantic_review_required"] is True and result["content_certified"] is False
    assert "private-credential" not in json.dumps(result)
    assert CANDIDATE["rule"] not in json.dumps(result)


@pytest.mark.parametrize(("text", "code"), [
    (r"Use private\u002dcredential-xyz again.", "known_credential"),
    ("Use object 12345.", "source_identifier"),
    ("Use source 22cc237_2.", "source_identifier"),
    ("Synthetic exact answer", "copied_answer"),
    ('{"name":"synthetic__login","arguments":{"password":"private-credential-xyz"}}', "copied_action"),
])
def test_exact_and_escaped_source_material_is_rejected_without_exposing_it(text, code):
    candidate = {**CANDIDATE, "rule": text}
    result = content.screen(writer_messages(), candidate)
    assert result["status"] == "rejected" and code in result["reason_codes"]
    assert text not in json.dumps(result)


def test_invalid_history_is_an_error_not_a_clear_screen():
    messages = writer_messages()
    messages[1]["content"] = '{"history":[],"score":true}'
    with pytest.raises(ValueError, match="^native_experience_content_invalid$"):
        content.screen(messages, CANDIDATE)


def test_nested_escape_inventory_is_checked_without_exposing_source_values():
    messages = writer_messages()
    payload = json.loads(messages[1]["content"])
    payload["history"][2]["content"] = json.dumps({"name": "synthetic__login", "arguments": {
        "password": r"private\u002dcredential-xyz"}})
    messages[1]["content"] = json.dumps(payload)
    result = content.screen(messages, {**CANDIDATE, "rule": "Avoid private-credential-xyz."})
    assert result["status"] == "rejected" and "known_credential" in result["reason_codes"]
    assert "private-credential" not in json.dumps(result)


def test_no_literal_flag_does_not_certify_unsupported_advice():
    result = content.screen(writer_messages(), {**CANDIDATE, "rule": "Repeat uncertain writes safely."})
    assert result["status"] == "passed"
    assert result["semantic_review_required"] is True and result["content_certified"] is False


def test_review_input_preserves_full_history_and_exact_candidate_and_insertion():
    original = writer_messages()
    before = copy.deepcopy(original)
    messages = content.build_review_messages(original, CANDIDATE)
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"history", "candidate", "insertion"}
    assert payload["history"] == json.loads(original[1]["content"])["history"]
    assert payload["candidate"] == CANDIDATE
    assert payload["insertion"] == content.insertion(CANDIDATE)
    assert WRITER_SYSTEM not in messages[0]["content"]
    assert original == before


@pytest.mark.parametrize("change", ["no_evidence", "outside_history", "boolean_index", "free_text", "duplicate_key"])
def test_review_parser_rejects_unsupported_approval_and_unstructured_content(change):
    value = verdict()
    if change == "no_evidence":
        value["evidence_indices"] = []
    elif change == "outside_history":
        value["evidence_indices"] = [6]
    elif change == "boolean_index":
        value["evidence_indices"] = [True]
    elif change == "free_text":
        value["reason_codes"] = ["Synthetic private rationale"]
    text = json.dumps(value)
    if change == "duplicate_key":
        text = '{"verdict":"reject","verdict":"plausible_pending_utility","reason_codes":["visible_conditional_support"],"evidence_indices":[3]}'
    with pytest.raises(ValueError, match="^native_experience_content_invalid$"):
        content.parse_review(text, history_length=6)


class Model:
    max_tokens = 256

    def __init__(self, *, count=100, kind="plausible_pending_utility", fail=None, finish="stop"):
        self.count, self.fail, self.finish, self.calls = count, fail, finish, []
        self.text = json.dumps(verdict(kind))

    def check_health(self):
        self.calls.append("health")
        return {"ready": True}

    def count_input_tokens(self, messages):
        self.calls.append(("count", copy.deepcopy(messages)))
        if self.fail == "count":
            raise RuntimeError("Synthetic private counter body")
        return self.count

    def generate(self, messages, seed):
        self.calls.append(("generate", copy.deepcopy(messages), seed))
        if self.fail == "generation":
            raise RuntimeError("Synthetic private transport body")
        return {"text": self.text, "prompt_tokens": self.count, "completion_tokens": 40,
                "raw_reply": {"choices": [{"finish_reason": self.finish,
                    "message": {"role": "assistant", "content": self.text}}]}}


def run(model=None, fail_event=None):
    model = Model() if model is None else model
    events = []

    def record(event):
        if event["event"] == fail_event:
            raise OSError("Synthetic private persistence body")
        events.append(copy.deepcopy(event))

    messages = content.build_review_messages(writer_messages(), CANDIDATE)
    return content.review_once(messages, model, seed=117, record=record), events, model


@pytest.mark.parametrize(("kind", "status"), [
    ("plausible_pending_utility", "plausible_pending_utility"),
    ("reject", "rejected"), ("insufficient_evidence", "abstained"),
])
def test_one_review_keeps_verdict_unaccepted_and_accounts_every_returned_token(kind, status):
    result, events, model = run(Model(kind=kind))
    assert result["status"] == status and result["verdict"] == verdict(kind)
    assert result["accepted"] is result["content_certified"] is result["training_authorized"] is False
    assert result["cost"]["generation_attempts"] == result["cost"]["generation_results"] == 1
    assert result["cost"]["returned_prompt_tokens_lower_bound"] == 100
    assert result["cost"]["returned_completion_tokens_lower_bound"] == 40
    assert not result["cost"]["unknown_cost"] and len(model.calls) == 3
    assert events[-1]["event"] == "review_stop"


def test_full_review_context_overflow_abstains_without_generation_or_truncation():
    result, _events, model = run(Model(count=15361))
    assert result["status"] == "abstained" and result["reason"] == "context_overflow"
    assert result["cost"]["CPU_counts_returned"] == 1 and result["cost"]["generation_attempts"] == 0
    assert len(model.calls) == 2


@pytest.mark.parametrize("kind", ["length", "malformed"])
def test_incomplete_or_malformed_review_is_abstained_but_fully_charged(kind):
    model = Model(finish="length" if kind == "length" else "stop")
    if kind == "malformed":
        model.text = '{"verdict":"approve","private_rationale":"not allowed"}'
    result, _events, _model = run(model)
    assert result["status"] == "abstained" and result["verdict"] is None
    assert result["cost"]["returned_completion_tokens_lower_bound"] == 40
    assert not result["cost"]["unknown_cost"]


@pytest.mark.parametrize("stage", ["count", "generation"])
def test_unreturned_review_operation_stays_unknown(stage):
    result, events, _model = run(Model(fail=stage))
    assert result["status"] == "unknown" and result["verdict"] is None
    assert result["cost"]["unknown_cost"]
    assert result["cost"]["generation_attempts"] == (stage == "generation")
    assert "private" not in json.dumps(result) and "private transport body" not in json.dumps(events)


def test_failed_result_journal_keeps_observed_review_usage_unknown():
    result, _events, _model = run(fail_event="generation_result")
    assert result["status"] == "unknown" and result["reason"] == "journal_failure"
    assert result["cost"]["unknown_cost"] and result["verdict"] is None
    assert result["cost"]["generation_results"] == 1
    assert result["cost"]["returned_completion_tokens_lower_bound"] == 40
