from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import native_state_recovery_fixture_episode_001 as episode
from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts import run_native_state_recovery_fixture_001 as run


def _rows(raw):
    result = []
    for i, case in enumerate(protocol.CASES):
        original_sha = protocol._hash(raw)
        adjusted = protocol.actor_metadata(raw, case, original_sha)
        result.append({"case": case["case"], "template_task_id": f"{i:07x}_3",
                       "worker_metadata_sha256": original_sha, "actor_metadata_sha256": protocol._hash(adjusted),
                       "instruction_sha256": protocol._hash(adjusted["instruction"]),
                       "messages_sha256": "a" * 64, "input_tokens": 100})
    return result


def test_transport_projection_is_exact_and_does_not_edit_worker_record(monkeypatch):
    raw = {"instruction": "Original placeholder", "supervisor": {}, "app_descriptions": {}}
    rows = _rows(raw)
    case, row = protocol.CASES[0], rows[0]
    reply = {"ok": True, "result": raw, "cost": {"action_attempts": 0}, "freshness": {}}
    adapted = episode._metadata_reply(reply, case, row)
    transport = [{}, {}, {"event": "request_intent", "request": {"op": "metadata"}},
                 {"event": "request_result", "reply": reply}, *[{} for _ in range(6)]]
    controller = [{"event": "metadata_result", "reply": adapted}]
    saved = deepcopy(transport)
    core = SimpleNamespace(_one=lambda records, _: records[0], _actor_evidence=lambda *_: (adapted["result"], [], []))
    monkeypatch.setattr(protocol, "qualify_witness", lambda *_: {"fixture_target_verified": False})

    def original(_receipt, _controller, _worker, projected, *_args, **_kwargs):
        assert projected[3]["reply"] == adapted and projected[4:] == transport[4:]
        return {"fault_provenance": None}

    verify = episode.authenticate(core, original, rows)
    receipt, context = {"task_id": row["template_task_id"]}, episode._case_context(row, case)
    args = (receipt, controller, [], transport, {}, None)
    assert verify(*args, initial_context=context, run_dir="unused")["fixture"]["fixture_target_verified"] is False
    assert transport == saved
    controller[0]["reply"]["freshness"] = {"modified": True}
    with pytest.raises(ValueError):
        verify(*args, initial_context=context, run_dir="unused")
    controller[0]["reply"] = episode._metadata_reply(reply, case, row)
    raw["instruction"] = "Changed original worker instruction"
    with pytest.raises(ValueError):
        verify(*args, initial_context=context, run_dir="unused")


@pytest.mark.parametrize("active,count,status", [(True, 10, "insufficient"), (False, 9, "insufficient"),
                                                     (False, 10, "passed"), (False, 10, "not_evaluable")])
def test_fixture_admission_rejects_live_partial_or_other_natural_gate(monkeypatch, active, count, status):
    monkeypatch.setattr(run.natural_report, "report", lambda **_: {"coordinator_running": active,
        "audited_scored_and_released": count, "gate": {"status": status, "witness_scenarios": 0}})
    with pytest.raises(ValueError):
        run.qualify("b" * 64)


def test_fixture_gate_requires_full_denominator_and_two_target_transitions():
    tasks = [f"{i:07x}_3" for i in range(4)]
    report = {"status": "complete", "slots": [{"template_task_id": task, "status": "known",
        "fixture_target_verified": i < 2, "witness": {"status": "qualified"}} for i, task in enumerate(tasks)],
        "costs": {"episodes": [{"source_index": i, "counts_complete": True, "unknown_cost": False} for i in range(4)],
                  "GPU_jobs": [{"job": {"batch_index": i}, "release": {"verified": True, "runtime_overrun": False}}
                               for i in range(4)]}}
    assert run.gate(report, tasks) == {"status": "passed", "witness_scenarios": 2, "source_denominator": 4}
    report["slots"][0]["fixture_target_verified"] = False
    assert run.gate(report, tasks)["status"] == "insufficient"
    report["costs"]["GPU_jobs"][-1]["release"]["verified"] = False
    assert run.gate(report, tasks)["status"] == "not_evaluable"
