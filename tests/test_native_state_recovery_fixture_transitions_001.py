import json

import pytest
from test_native_state_recovery_evidence_001 import case

from scripts import native_state_recovery_evidence_001 as witness
from scripts import native_state_recovery_fixture_transitions_001 as transitions
from wll.protocol import fingerprint


@pytest.mark.parametrize(("app", "family", "initial"), [("todoist", "B", False), ("simple_note", "A", True)])
def test_projection_preserves_state_direction_without_source_identifiers(app, family, initial):
    records, fault = case(app)
    field = "pinned" if app == "simple_note" else "is_completed"
    for index in (1, 2):
        action = json.loads(records[index]["action"])
        action["arguments"][field] = not initial
        records[index]["action"] = json.dumps(action)
    fault["action_sha256"] = fingerprint(action)
    records[0]["observation"]["result"][field] = initial
    records[3]["observation"]["result"][field] = not initial
    checked = witness.classify(records, fault)
    view = transitions.project(records, checked, fault)
    assert view == {"operation_family": family, "resource": "target", "events": [
        {"event": "read", "observed_value": initial},
        {"event": "write", "requested_value": not initial, "outcome": "explicitly_not_executed"},
        {"event": "write", "requested_value": not initial, "same_call": True, "outcome": "acknowledged"},
        {"event": "read", "observed_value": not initial},
    ]}
    records[3]["observation"]["result"][field] = initial
    with pytest.raises(ValueError):
        transitions.project(records, checked, fault)
