from types import SimpleNamespace

import pytest

from scripts import run_native_state_recovery_transition_utility_003 as utility


@pytest.mark.parametrize("same_job", [True, False])
def test_registry_has_backend_contract_and_release_uses_a_fresh_status(monkeypatch, same_job):
    registry = SimpleNamespace(load_registration=lambda *_args, **_kwargs: {}, REQUIRED_NEW_SOURCES=set())
    old = {"job": {"status": "finished"}, "epoch": 100}
    fresh = {"job": old["job"] if same_job else {"status": "replaced"}, "epoch": 101,
             "owned": True, "monitor_live": True, "admission_cancelled": True, "released": True, "model_pids": []}
    operations = []

    class Runtime:
        def now(self):
            return 10 + len(operations)

        def queue(self, operation, job, pids):
            operations.append((operation, job, pids))
            return fresh

        def release(self, _job, _pids):
            operations.append(("fence",))
            return {"verified": True, "state": old, "release_wall_seconds": 1}

    class Source:
        def bindings(self):
            return [SimpleNamespace(static_backend=lambda value: value["static"])]

    monkeypatch.setattr(utility.base, "bindings", lambda _: (registry, None, SimpleNamespace(_Runtime=Runtime)))
    monkeypatch.setattr(utility, "LINEAGE", SimpleNamespace(source=Source()))
    utility.bindings.cache_clear()
    try:
        reg, _, runner = utility.bindings()
        assert reg._same(reg.static_backend({"static": {"arm": "B3"}}), {"arm": "B3"})
        if same_job:
            result = runner._Runtime().release("job", [])
            assert result["state"] == fresh and result["release_wall_seconds"] == 2
        else:
            with pytest.raises(ValueError):
                runner._Runtime().release("job", [])
        assert operations == [("fence",), ("status", "job", [])]
    finally:
        utility.bindings.cache_clear()
