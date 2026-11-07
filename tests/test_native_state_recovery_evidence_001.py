"""Synthetic-only native observation and fault provenance checks."""

import copy
import json

import pytest

from scripts import native_state_recovery_evidence_001 as checker
from wll.protocol import fingerprint


def action(name, **arguments):
    return json.dumps({"name": name, "arguments": arguments})


def case(app="todoist"):
    key, field = ("task_id", "is_completed") if app == "todoist" else ("note_id", "pinned")
    resource, token = 42, "synthetic-token"
    update = {"name": app + ("__update_task" if app == "todoist" else "__update_note"),
              "arguments": {"access_token": token, key: resource, field: True}}
    show = action(app + ("__show_task" if app == "todoist" else "__show_note"),
                  access_token=token, **{key: resource})
    records = [
        {"action": show, "observation": {"ok": True, "result": {key: resource, field: False}}},
        {"action": json.dumps(update), "observation": {"ok": False, "error_type": "ToolDispatchError",
                            "message": checker.FAULT_MESSAGE}},
        {"action": json.dumps(update, sort_keys=True), "observation": {"ok": True,
                            "result": {"message": "synthetic update"}}},
        {"action": show, "observation": {"ok": True, "result": {key: resource, field: True}}},
    ]
    provenance = {"policy": checker.POLICY, "fault_index": 1, "action_sha256": fingerprint(update),
                  "injected_count": 1, "backend_dispatches_at_refusal": 0}
    return records, provenance


@pytest.mark.parametrize("app", ["todoist", "simple_note"])
def test_exact_model_visible_sequence_qualifies_without_exposing_values(app):
    rows, fault = case(app)
    result = checker.classify(rows, fault)
    assert result["status"] == "qualified" and result["reason"] == "verified_transition"
    assert result["private_evidence"]["indices"] == {"before": 0, "fault": 1, "correction": 2, "after": 3}
    assert set(result["private_evidence"]["record_sha256"]) == {"before", "fault", "correction", "after"}
    assert all(len(sha) == 64 for sha in result["private_evidence"]["record_sha256"].values())
    public = checker.public_projection(result)
    assert set(public) == {"status", "reason", "counts"}
    assert "synthetic-token" not in str(public) and "42" not in str(public)


@pytest.mark.parametrize("change", ["missing", "bad_policy", "bad_index", "bad_hash", "count_bool",
                                    "backend_bool", "backend_one", "extra_field", "message"])
def test_claimed_fault_requires_exact_external_provenance_and_observation(change):
    rows, fault = case()
    if change == "missing":
        fault = None
    elif change == "bad_policy":
        fault["policy"] = "other"
    elif change == "bad_index":
        fault["fault_index"] = True
    elif change == "bad_hash":
        fault["action_sha256"] = "0" * 64
    elif change == "count_bool":
        fault["injected_count"] = True
    elif change == "backend_bool":
        fault["backend_dispatches_at_refusal"] = False
    elif change == "backend_one":
        fault["backend_dispatches_at_refusal"] = 1
    elif change == "extra_field":
        fault["untrusted"] = True
    else:
        rows[1]["observation"]["message"] = "ordinary validation error"
    assert checker.classify(rows, fault)["status"] == "unknown"


@pytest.mark.parametrize("change", ["fault_value_int", "fault_id_bool", "extra_write_argument",
                                    "before_value_int", "after_value_int", "before_id_bool", "after_id_bool"])
def test_exact_json_boolean_and_integer_types_are_required(change):
    rows, fault = case()
    write = json.loads(rows[1]["action"])
    if change == "fault_value_int":
        write["arguments"]["is_completed"] = 1
    elif change == "fault_id_bool":
        write["arguments"]["task_id"] = True
    elif change == "extra_write_argument":
        write["arguments"]["title"] = "unverified"
    elif change == "before_value_int":
        rows[0]["observation"]["result"]["is_completed"] = 0
    elif change == "after_value_int":
        rows[3]["observation"]["result"]["is_completed"] = 1
    elif change == "before_id_bool":
        rows[0]["observation"]["result"]["task_id"] = True
    else:
        rows[3]["observation"]["result"]["task_id"] = True
    if change.startswith("fault_") or change == "extra_write_argument":
        rows[1]["action"] = json.dumps(write)
        fault["action_sha256"] = fingerprint(write)
    assert checker.classify(rows, fault)["status"] != "qualified"


@pytest.mark.parametrize("change", ["target", "token", "extra_argument", "value", "post_target",
                                    "no_before", "no_after", "before_already", "after_wrong",
                                    "correction_failed", "repeat", "intervening_write"])
def test_no_cherry_picking_of_changed_missing_or_repeated_writes(change):
    rows, fault = case()
    if change in {"target", "token", "extra_argument", "value"}:
        write = json.loads(rows[2]["action"])
        key, value = {"target": ("task_id", 43), "token": ("access_token", "different"),
                      "extra_argument": ("title", "changed"), "value": ("is_completed", False)}[change]
        write["arguments"][key] = value
        rows[2]["action"] = json.dumps(write)
    elif change == "post_target":
        rows[3]["observation"]["result"]["task_id"] = 43
    elif change == "no_before":
        rows.pop(0)
        fault["fault_index"] = 0
    elif change == "no_after":
        rows.pop()
    elif change == "before_already":
        rows[0]["observation"]["result"]["is_completed"] = True
    elif change == "after_wrong":
        rows[3]["observation"]["result"]["is_completed"] = False
    elif change == "correction_failed":
        rows[2]["observation"] = {"ok": False, "error_type": "ObservedAPIError", "status_code": 422,
                                   "message": "synthetic failure"}
    elif change == "repeat":
        rows.insert(3, copy.deepcopy(rows[2]))
    else:
        rows.insert(2, {"action": action("todoist__delete_task", task_id=42, access_token="synthetic-token"),
                        "observation": {"ok": True, "result": {"message": "synthetic"}}})
    assert checker.classify(rows, fault)["status"] != "qualified"


@pytest.mark.parametrize("change", ["bad_json", "duplicate_key", "unknown_observation", "extra_record_field"])
def test_malformed_or_unrecognized_evidence_is_unknown(change):
    rows, fault = case()
    if change == "bad_json":
        rows[2]["action"] = "not JSON"
    elif change == "duplicate_key":
        rows[2]["action"] = '{"name":"todoist__update_task","name":"other","arguments":{}}'
    elif change == "unknown_observation":
        rows[2]["observation"] = {"ok": False, "error_type": "Other", "message": "synthetic"}
    else:
        rows[2]["official_success"] = True
    assert checker.classify(rows, fault)["status"] == "unknown"


def test_registered_no_fault_is_retained_as_ineligible_without_fabricated_evidence():
    rows, _fault = case()
    no_fault = {"policy": checker.POLICY, "fault_index": None, "action_sha256": None,
                "injected_count": 0, "backend_dispatches_at_refusal": None}
    result = checker.classify(rows, no_fault)
    assert result["status"] == "ineligible" and result["reason"] == "no_fault"
    assert result["private_evidence"] is None
    assert checker.public_projection(result)["counts"]["faults"] == 0


def test_known_read_only_intervening_query_does_not_become_a_write():
    rows, fault = case()
    rows.insert(2, {"action": action("api_docs__show_app_descriptions"),
                    "observation": {"ok": True, "result": [{"name": "todoist"}]}})
    result = checker.classify(rows, fault)
    assert result["status"] == "qualified"
    assert result["private_evidence"]["indices"] == {"before": 0, "fault": 1, "correction": 3, "after": 4}


@pytest.mark.parametrize("change", ["contradictory_read", "later_target_mutation", "unknown_intervening"])
def test_later_contradiction_or_ambiguous_target_mutation_never_qualifies(change):
    rows, fault = case()
    if change == "contradictory_read":
        later = copy.deepcopy(rows[3])
        later["observation"]["result"]["is_completed"] = False
        rows.append(later)
    elif change == "later_target_mutation":
        rows.append({"action": action("todoist__delete_task", task_id=42, access_token="synthetic-token"),
                     "observation": {"ok": True, "result": {"message": "synthetic"}}})
    else:
        rows.insert(2, {"action": action("mystery__operation", task_id=42),
                        "observation": {"ok": True, "result": {}}})
    result = checker.classify(rows, fault)
    assert result["status"] != "qualified"
    assert checker.public_projection(result)["reason"] == result["reason"]


@pytest.mark.parametrize("change", ["reason", "status", "counts", "extra"])
def test_public_projection_refuses_arbitrary_labels_and_noninteger_counts(change):
    rows, fault = case()
    result = checker.classify(rows, fault)
    if change == "reason":
        result["reason"] = "synthetic-token"
    elif change == "status":
        result["status"] = []
    elif change == "counts":
        result["counts"]["faults"] = True
    else:
        result["secret"] = "synthetic-token"
    with pytest.raises(ValueError):
        checker.public_projection(result)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("invalid", ["not JSON", '{"name":"a","name":"b","arguments":{}}',
                                      '{"name":"bad","arguments":{}}'])
def test_authenticated_predispatch_rejection_is_known_without_inventing_an_action(position, invalid):
    rows, fault = case()
    rows.insert(position, {"action": invalid, "observation": {"ok": False, "error_type": "ToolDispatchError",
                                                              "message": "synthetic rejected action"}})
    if position <= fault["fault_index"]:
        fault["fault_index"] += 1
    result = checker.classify(rows, fault)
    assert result["status"] == "qualified" and result["reason"] == "verified_transition"
    assert checker._decode_records(rows)[position][0] is None
    assert result["counts"]["records"] == 5


def test_authenticated_predispatch_rejection_without_fault_is_known_ineligible():
    rows = [{"action": "not JSON", "observation": {"ok": False, "error_type": "ToolDispatchError",
                                                    "message": "synthetic rejected action"}}]
    no_fault = {"policy": checker.POLICY, "fault_index": None, "action_sha256": None,
                "injected_count": 0, "backend_dispatches_at_refusal": None}
    result = checker.classify(rows, no_fault)
    assert result["status"] == "ineligible" and result["reason"] == "no_fault"
    assert result["counts"]["records"] == 1


def test_raw_invalid_unicode_cannot_be_an_authenticated_actor_dispatch():
    rows, fault = case()
    rows.insert(0, {"action": "\ud800", "observation": {"ok": False, "error_type": "ToolDispatchError",
                                                          "message": "synthetic rejected action"}})
    fault["fault_index"] += 1
    assert checker.classify(rows, fault)["status"] == "unknown"


def test_oversized_predispatch_rejection_is_retained():
    rows, fault = case()
    rows.insert(2, {"action": "x" * (checker.MAX_ACTION_BYTES + 1),
                    "observation": {"ok": False, "error_type": "ToolDispatchError", "message": "Too long."}})
    assert checker.classify(rows, fault)["status"] == "qualified"


@pytest.mark.parametrize("observation", [
    {"ok": True, "result": {}},
    {"ok": False, "error_type": "ObservedAPIError", "status_code": 422, "message": "synthetic"},
    {"ok": False, "error_type": "Other", "message": "synthetic"},
])
def test_malformed_action_with_non_predispatch_or_ambiguous_result_stays_unknown(observation):
    rows, fault = case()
    rows.insert(2, {"action": "not JSON", "observation": observation})
    result = checker.classify(rows, fault)
    assert result["status"] == "unknown" and result["reason"] == "invalid_records"


def test_malformed_action_cannot_become_the_bound_fault():
    rows, fault = case()
    rows[1]["action"] = "not JSON"
    fault["action_sha256"] = fingerprint(None)
    result = checker.classify(rows, fault)
    assert result["status"] == "unknown" and result["reason"] == "fault_action_mismatch"


def test_valid_but_rejected_unknown_tool_has_no_intervening_effect():
    rows, fault = case()
    rows.insert(2, {"action": action("mystery__operation", task_id=42),
                    "observation": {"ok": False, "error_type": "ToolDispatchError",
                                    "message": "Unknown public tool."}})
    assert checker.classify(rows, fault)["status"] == "qualified"


def test_predispatch_rejected_repeat_of_same_write_still_disqualifies():
    rows, fault = case()
    repeated = copy.deepcopy(rows[2])
    repeated["observation"] = {"ok": False, "error_type": "ToolDispatchError", "message": "Rejected."}
    rows.insert(2, repeated)
    result = checker.classify(rows, fault)
    assert result["status"] == "ineligible" and result["reason"] == "repeated_write"
