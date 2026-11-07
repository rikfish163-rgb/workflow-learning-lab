import copy
import json

import pytest
from test_native_state_recovery_writer_stage_001 import BACKEND, CANDIDATE, MODEL_ID, VERDICT, Model

from scripts import native_state_recovery_transition_review_001 as audit
from scripts import native_state_recovery_transition_stage_001 as stage
from scripts import native_state_recovery_writer_evidence_001 as frozen


def _case(family, initial):
    return {"operation_family": family, "resource": "target", "events": [
        {"event": "read", "observed_value": initial},
        {"event": "write", "requested_value": not initial, "outcome": "explicitly_not_executed"},
        {"event": "write", "requested_value": not initial, "same_call": True, "outcome": "acknowledged"},
        {"event": "read", "observed_value": not initial},
    ]}


CASES = [_case("A", False), _case("B", False), _case("A", True), _case("B", False)]


def test_ordered_payload_keeps_protocol_and_replays_exact_review():
    calls, events = [], []
    verdict = {**VERDICT, "evidence_indices": list(range(4))}
    result = stage.run(CASES, model_factory=lambda limit: Model(
        limit, CANDIDATE if limit == 128 else verdict, calls,
    ), guard=lambda: None, record=events.append)
    assert result["status"] == "plausible_pending_utility" and result["utility_authorized"] is False
    assert [(limit, seed) for limit, seed, _ in calls] == [(128, 17), (256, 18)]
    assert stage.writer_messages(CASES)[0]["content"] == frozen.WRITER_SYSTEM
    assert calls[0][2] == {"observed_state_transition_cases": CASES}
    assert calls[1][2]["observed_state_transition_cases"] == CASES
    assert audit._auditor().replay(CASES, events, backend=BACKEND, model_id=MODEL_ID) == result
    with pytest.raises(ValueError):
        audit._auditor().replay(CASES[::-1], events, backend=BACKEND, model_id=MODEL_ID)


@pytest.mark.parametrize("change", ["uncertain", "reorder", "extra", "wrong_read", "wrong_type"])
def test_ordered_projection_rejects_unsafe_or_unbound_fields(change):
    cases = copy.deepcopy(CASES)
    events = cases[0]["events"]
    if change == "uncertain":
        events[1]["outcome"] = "timeout"
    elif change == "reorder":
        events[0], events[1] = events[1], events[0]
    elif change == "extra":
        events[0]["identifier"] = "private"
    elif change == "wrong_read":
        events[-1]["observed_value"] = False
    else:
        events[-1]["observed_value"] = 1
    with pytest.raises(ValueError):
        stage.writer_messages(cases)


def test_negative_writer_stops_without_review_and_remains_replayable():
    calls, events = [], []
    response = {"status": "abstain", "reason": "Synthetic negative."}
    result = stage.run(CASES, model_factory=lambda limit: Model(limit, response, calls),
                       guard=lambda: None, record=events.append)
    assert result["status"] == "abstained" and result["review"] is None and len(calls) == 1
    assert "operation_family" in json.dumps(calls[0][2])
    assert audit._auditor().replay(CASES, events, backend=BACKEND, model_id=MODEL_ID) == result
