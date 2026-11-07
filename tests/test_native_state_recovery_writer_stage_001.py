import copy
import json

import pytest

from scripts import native_state_recovery_review_001 as review_audit
from scripts import native_state_recovery_writer_evidence_001 as evidence
from scripts import native_state_recovery_writer_stage_001 as stage
from scripts import run_native_state_recovery_writer_001 as coordinator

CASES = [dict.fromkeys(evidence._FACTS, True) for _ in range(2)]
CANDIDATE = {"status": "candidate", "failure_mode": "Known refusal", "applies_when": "A write is explicitly refused.",
             "avoid_when": "Prior execution is uncertain.", "rule": "Read current state, correct once, then read back."}
VERDICT = {"verdict": "plausible_pending_utility", "reason_codes": ["visible_conditional_support"],
           "evidence_indices": [0, 1]}
BACKEND = {"identity": "synthetic-backend"}
MODEL_ID = "synthetic-model"


class Model:
    def __init__(self, max_tokens, response, calls):
        self.max_tokens, self.response, self.calls = max_tokens, response, calls

    def check_health(self):
        pass

    def count_input_tokens(self, _messages):
        return 100

    def generate(self, messages, *, seed):
        self.calls.append((self.max_tokens, seed, json.loads(messages[1]["content"])))
        if self.response is None:
            raise RuntimeError("synthetic transport failure")
        reply = json.dumps(self.response)
        usage = {"prompt_tokens": 100, "completion_tokens": 32}
        return {"text": reply, **usage, "format_error": None, "backend": BACKEND,
                "raw_reply": {"model": MODEL_ID, "backend": BACKEND, "usage": usage,
                              "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": reply}}]}}


def execute(draft, review=VERDICT, guard=lambda: None):
    calls, events = [], []
    result = stage.run(CASES, model_factory=lambda tokens: Model(tokens, draft if tokens == 128 else review, calls),
                       guard=guard, record=events.append)
    return result, calls, events


def test_exact_insertion_and_private_review_only_pending_utility():
    result, calls, events = execute(CANDIDATE)
    assert result["status"] == "plausible_pending_utility"
    assert result["utility_authorized"] is result["training_authorized"] is False
    assert [(limit, seed) for limit, seed, _ in calls] == [(128, 17), (256, 18)]
    assert calls[0][2] == {"observed_state_transition_cases": CASES}
    assert calls[1][2] == {**calls[0][2], "candidate": CANDIDATE, "insertion": stage.content.insertion(CANDIDATE)}
    binding, = [event for event in events if event["event"] == "review_binding"]
    assert result["insertion_sha256"] == binding["insertion_sha256"]
    assert [event["stage"] for event in events if event["event"] == "generation_result"] == ["writer", "review"]


@pytest.mark.parametrize("draft,status", [({"status": "abstain", "reason": "Insufficient scope"}, "abstained"),
    ({"status": "candidate"}, "abstained"), (dict(CANDIDATE, rule="Call api.foo()\nnow"), "structurally_rejected"),
    (None, "unknown")])
def test_no_review_for_abstention_invalid_output_or_transport_failure(draft, status):
    result, calls, _ = execute(draft)
    assert result["status"] == status and result["review"] is None and len(calls) == 1


def test_review_with_missing_case_abstains_without_retry():
    result, calls, _ = execute(CANDIDATE, dict(VERDICT, evidence_indices=[0]))
    assert result["status"] == "abstained" and len(calls) == 2
    assert result["review"]["reason"] == "invalid_review_schema"


def test_failed_lease_check_prevents_request():
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("synthetic release")

    result, calls, _ = execute(CANDIDATE, guard=guard)
    assert result["status"] == "unknown" and calls == []


@pytest.mark.parametrize("gate,audited", [(None, 1), ({"status": "insufficient"}, 10), ({"status": "passed"}, 9)])
def test_source_qualification_rejects_live_partial_and_insufficient_data(monkeypatch, gate, audited):
    monkeypatch.setattr(coordinator.audit, "report", lambda **_: {"gate": gate, "audited_scored_and_released": audited})
    monkeypatch.setattr(evidence, "prepare", lambda _: pytest.fail("ineligible source supplied model input"))
    with pytest.raises(ValueError):
        coordinator.qualify("1" * 64)


def test_full_source_proof_hash_binds_private_lineage(monkeypatch):
    prepared = {"cases": CASES, "private_provenance": [{"source_index": 0}, {"source_index": 4}]}
    monkeypatch.setattr(coordinator.audit, "report", lambda **_: {
        "gate": {"status": "passed", "witness_scenarios": 2}, "audited_scored_and_released": 10})
    monkeypatch.setattr(evidence, "prepare", lambda _: prepared)
    proof = coordinator.qualify("1" * 64)
    assert coordinator._prepare({"qualification": proof}) == prepared
    assert "private_provenance" not in proof
    prepared["private_provenance"][1]["source_index"] = 5
    with pytest.raises(ValueError):
        coordinator._prepare({"qualification": proof})


def test_recorded_model_replay_authenticates_exact_reviewer_input_and_usage():
    result, _, events = execute(CANDIDATE)
    assert review_audit.replay(CASES, events, backend=BACKEND, model_id=MODEL_ID) == result
    tampered = copy.deepcopy(events)
    for event in tampered:
        if event.get("stage") == "review" and event["event"] == "generation_intent":
            event["messages"][1]["content"] += " "
    with pytest.raises(ValueError):
        review_audit.replay(CASES, tampered, backend=BACKEND, model_id=MODEL_ID)
    with pytest.raises(ValueError):
        review_audit.replay(CASES, events, backend={"identity": "other"}, model_id=MODEL_ID)


@pytest.mark.parametrize(("draft", "verdict", "status"), [
    ({"status": "abstain", "reason": "Example abstention"}, VERDICT, "abstained"),
    (dict(CANDIDATE, rule="Call api.foo()\nnow"), VERDICT, "structurally_rejected"),
    (CANDIDATE, {"verdict": "reject", "reason_codes": ["unsafe_repetition"], "evidence_indices": [0]}, "rejected"),
    (CANDIDATE, dict(VERDICT, evidence_indices=[0]), "abstained"),
])
def test_terminal_replay_retains_authenticated_non_positive_results(draft, verdict, status):
    result, _, events = execute(draft, verdict)
    assert result["status"] == status
    assert review_audit.replay(CASES, events, backend=BACKEND, model_id=MODEL_ID) == result


def test_report_projects_only_outcomes_and_cost(monkeypatch):
    result, _, _ = execute(CANDIDATE)
    plan = {"id": "synthetic-writer", "qualification": {"witness_scenarios": 2}}
    monkeypatch.setattr(review_audit, "_terminal_writer", lambda *_: (plan, result, {}, {"physical_GPU": 8}))
    report = review_audit.report("a" * 64, "b" * 64)
    assert report["utility_authorized_by_writer"] is False
    assert report["status"] == "plausible_pending_utility"
    assert set(report["writer"]) == set(report["review"]) == {"status", "reason", "cost"}
    assert report["review"]["cost"]["generation_results"] == 1
    assert CANDIDATE["rule"] not in json.dumps(report)
