from types import SimpleNamespace

import pytest

from scripts import advance_native_state_recovery_001 as advance
from scripts.native_appworld_worker_client import _encode


def test_staging_preserves_existing_content(tmp_path):
    (tmp_path / "artifacts").mkdir()
    path = "artifacts/sample.json"
    advance._stage(tmp_path, path, b"original")
    advance._stage(tmp_path, path, b"original")
    with pytest.raises(ValueError):
        advance._stage(tmp_path, path, b"changed")
    assert (tmp_path / path).read_bytes() == b"original"


def test_handoff_requires_full_evaluable_natural_source(monkeypatch, capsys):
    observations = {"gate": {"status": "not_evaluable"}, "audited_scored_and_released": 9}
    monkeypatch.setattr(advance.audit, "report", lambda **_: observations)
    monkeypatch.setattr(advance, "_route", lambda *_: pytest.fail("must not route"))
    assert advance.handoff("a" * 64) == 0
    assert "next_stage=none" in capsys.readouterr().out
    observations["gate"]["status"] = "insufficient"
    with pytest.raises(ValueError):
        advance.handoff("a" * 64)


def test_wait_repolls_original_handle_and_rejects_replacement(monkeypatch):
    unit = "wll-native-source-pilot-" + "a" * 32 + ".service"
    intent = {"unit": unit, "plan_sha256": advance.audit.PLAN_SHA256}
    terminal = b"terminal-synthetic"
    core = SimpleNamespace(_read_private=lambda path, **_: _encode(intent)
                           if path.name == "parent-launch-intent.json" else terminal)
    episode = SimpleNamespace(bind=lambda *_: SimpleNamespace(core=core), MAX_DOCUMENT=1000)
    reg = SimpleNamespace(PRIVATE_RUN_ROOT="/synthetic")
    monkeypatch.setattr(advance.source, "bindings", lambda: (reg, None, episode, None))
    monkeypatch.setattr(advance.time, "sleep", lambda _: None)
    replies = iter([ValueError(), 123, 0])

    def pid(actual):
        assert actual == unit
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(advance, "_main_pid", pid)
    assert advance._terminal(123) == advance._digest(terminal)
    monkeypatch.setattr(advance, "_main_pid", lambda _: 456)
    with pytest.raises(ValueError):
        advance._terminal(123)
