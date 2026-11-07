import pytest

from scripts import report_native_state_recovery_transition_utility_003 as audit


def test_terminal_wait_repolls_only_recorded_handle(monkeypatch):
    replies = iter([ValueError(), 123, 0])
    observed = []

    def pid(unit):
        observed.append(unit)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(audit, "_main_pid", pid)
    monkeypatch.setattr(audit.time, "sleep", lambda _: None)
    audit._wait("recorded.service", 123, 10)
    assert observed == ["recorded.service"] * 3


def test_terminal_wait_rejects_replacement(monkeypatch):
    monkeypatch.setattr(audit, "_main_pid", lambda _: 456)
    with pytest.raises(ValueError):
        audit._wait("recorded.service", 123, 10)
