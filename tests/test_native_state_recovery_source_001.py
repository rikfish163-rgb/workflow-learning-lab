import copy
from types import SimpleNamespace

import pytest

from scripts import native_state_recovery_gpu_queue_001 as queue
from scripts import native_state_recovery_registration_001 as registration
from scripts import run_native_state_recovery_source_001 as runner


def report():
    return {"status": "complete", "slots": [
        {"task_id": task, "first_attempt": {"status": "known", "official_success": False},
         "witness": {"status": "qualified" if index < 2 else "ineligible"}}
        for index, task in enumerate(registration.TASK_IDS)], "costs": {
            "episodes": [{"source_index": i, "counts_complete": True, "unknown_cost": False} for i in range(10)],
            "scorers": [{"source_index": i, "cleanup_verified": True, "result_validated": True} for i in range(10)],
            "GPU_jobs": [{"job": {"batch_index": i}, "release": {"verified": True, "runtime_overrun": False}}
                         for i in range(10)]}}


def test_gate_keeps_all_slots_and_does_not_select_on_official_score():
    data = report()
    assert runner.gate(data) == {"status": "passed", "source_denominator": 10, "witness_scenarios": 2}
    data["slots"][1]["witness"]["status"] = "ineligible"
    assert runner.gate(data)["status"] == "insufficient"
    data["slots"][9]["witness"]["status"] = "unknown"
    assert runner.gate(data)["status"] == "not_evaluable"


@pytest.mark.parametrize("lane,field", [("episodes", "counts_complete"), ("scorers", "cleanup_verified"),
                                       ("GPU_jobs", "verified")])
def test_gate_requires_all_accounting_and_release_rows(lane, field):
    data = report()
    reduced = copy.deepcopy(data)
    reduced["costs"][lane].pop()
    assert runner.gate(reduced)["status"] == "not_evaluable"
    row = data["costs"][lane][9]
    (row["release"] if lane == "GPU_jobs" else row)[field] = False
    assert runner.gate(data)["status"] == "not_evaluable"


def test_new_queue_uses_original_global_fence_and_only_ten_serial_leases():
    core = queue._core()
    job = queue.make_job(9, "a" * 32, "b" * 64)
    core._validate_job(job)
    assert core._get_core().PORT == registration.PORT == 18999
    assert core._get_core().load_registration is registration.load_registration
    assert core._get_core().REMOTE == queue.REMOTE
    assert queue.fenced_release is core.fenced_release
    assert job["max_runtime_seconds"] == 7200
    assert job["argv"][3] == "scripts.native_state_recovery_gpu_queue_001"
    with pytest.raises(ValueError):
        queue.make_job(10, "a" * 32, "b" * 64)


def test_lease_releases_if_episode_raises(monkeypatch):
    data = report()
    data["costs"]["GPU_jobs"].clear()
    events = []
    job = queue.make_job(0, "a" * 32, "b" * 64)
    state = {"released": False, "job": {"status": "running", "started_epoch": 1.0}, "model_pids": [11]}
    heartbeat = SimpleNamespace(check=lambda: None, stop=lambda: events.append("heartbeat stopped"))
    runtime = SimpleNamespace(now=lambda: 2.0, epoch=lambda: 2.0, monitor=lambda **_: None,
        record=lambda row: events.append(row["event"]), queue=lambda *_: state,
        ready=lambda *_: {"backend": {}}, heartbeat=lambda _: heartbeat)
    core = SimpleNamespace(_queue_state=lambda *_: None)

    def release(_runtime, actual_job, pids):
        assert actual_job == job and pids == [11]
        events.append("release checked")
        return {"verified": True, "runtime_overrun": False, "GPU_seconds": 1.0}

    monkeypatch.setattr(queue, "fenced_release", release)
    with pytest.raises(RuntimeError, match="episode"):
        with runner._lease(runtime, core, job, data) as (_service, guard):
            guard()
            raise RuntimeError("episode")
    assert events[-3:] == ["heartbeat stopped", "release checked", "GPU_released"]
    assert data["costs"]["GPU_jobs"][0]["release"]["verified"] is True
