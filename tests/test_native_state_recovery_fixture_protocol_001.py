from copy import deepcopy

import pytest

from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts.native_appworld_worker_client import _decode, _encode
from wll.protocol import fingerprint


def _fixture(case):
    app = case["app"]
    key, field = case["resource_key"], case["field"]
    read, write = ("show_note", "update_note") if app == "simple_note" else ("show_task", "update_task")
    auth = {"access_token": "synthetic-fixture-handle"}
    args = {**auth, key: 42}
    records = []

    def append(name, arguments, observation):
        records.append({"action": _encode({"name": app + "__" + name, "arguments": arguments}).decode(),
                        "observation": observation})

    append(case["create"], {**auth, "title": case["title"]}, {"ok": True, "result": {key: 42}})
    append(read, args, {"ok": True, "result": {key: 42, "title": case["title"], field: case["initial"]}})
    append(write, {**args, field: case["desired"]},
           {"ok": False, "error_type": "ToolDispatchError", "message": protocol.witness.FAULT_MESSAGE})
    append(write, {**args, field: case["desired"]}, {"ok": True, "result": {"message": "Updated"}})
    append(read, args, {"ok": True, "result": {key: 42, "title": case["title"], field: case["desired"]}})
    fault = {"policy": protocol.witness.POLICY, "injected_count": 1, "backend_dispatches_at_refusal": 0,
             "fault_index": 2, "action_sha256": fingerprint(_decode(records[2]["action"].encode()))}
    return records, fault


@pytest.mark.parametrize("case", protocol.CASES, ids=[case["case"] for case in protocol.CASES])
def test_state_transition_must_use_the_created_fixture_record(case):
    records, fault = _fixture(case)
    assert protocol.qualify_witness(records, fault, case)["fixture_target_verified"] is True
    changed = deepcopy(records)
    changed[0]["observation"]["result"][case["resource_key"]] = 43
    result = protocol.qualify_witness(changed, fault, case)
    assert result["witness"]["status"] == "qualified" and result["fixture_target_verified"] is False
    changed = deepcopy(records)
    changed[-1]["observation"]["result"]["title"] = "Different record title"
    assert protocol.qualify_witness(changed, fault, case)["fixture_target_verified"] is False


def test_fixture_metadata_changes_exactly_the_instruction_and_requires_original_hash():
    raw = {"instruction": "Original placeholder", "supervisor": {"first_name": "Fixture"}, "app_descriptions": {}}
    before = deepcopy(raw)
    adjusted = protocol.actor_metadata(raw, protocol.CASES[0], protocol._hash(raw))
    assert raw == before and adjusted["instruction"] == protocol.instructions(protocol.CASES[0])
    assert {key: adjusted[key] for key in ("supervisor", "app_descriptions")} == {
        key: raw[key] for key in ("supervisor", "app_descriptions")}
    with pytest.raises(ValueError):
        protocol.actor_metadata(raw, protocol.CASES[0], "0" * 64)
