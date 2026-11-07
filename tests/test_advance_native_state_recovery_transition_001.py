from types import SimpleNamespace

import pytest

from scripts import advance_native_state_recovery_transition_001 as advance


@pytest.mark.parametrize("status", ["abstained", "rejected", "plausible_pending_utility"])
def test_transition_handoff_requires_closed_exact_review_before_start(monkeypatch, status):
    reader, launch = object(), {"plan_sha256": "a" * 64}
    manifest = {"synthetic": "private"}
    registry = SimpleNamespace(PRIVATE_RUN_ROOT="/synthetic")
    downstream = SimpleNamespace(ID="synthetic-utility", QUALIFICATION_PROOF="qualification")
    monkeypatch.setattr(advance.writer, "bindings", lambda: (registry, None, None))
    monkeypatch.setattr(advance.writer, "core_for", lambda _: SimpleNamespace(episode=reader))
    monkeypatch.setattr(advance.handoff, "_intent", lambda actual, _: launch if actual is reader else None)
    monkeypatch.setattr(advance.handoff, "_closed", lambda *_: "b" * 64)
    monkeypatch.setattr(advance.review, "report", lambda *_: {"status": status})
    monkeypatch.setattr(advance.review, "prepare_manifest", lambda *_: {"manifest": manifest})
    history = []

    def freeze(actual, _private, actual_manifest):
        assert actual is reader and actual_manifest is manifest
        history.append("freeze")
        return "c" * 64

    monkeypatch.setattr(advance.staging, "_stage", lambda _root, path, _raw: history.append(path))
    monkeypatch.setattr(advance.handoff, "_private_manifest", freeze)
    monkeypatch.setattr(advance.common, "qualification", lambda *_args, **_kwargs: {"public": True})
    monkeypatch.setattr(advance.common, "bindings", lambda _: (downstream,))
    monkeypatch.setattr(advance.utility, "build_registration", lambda *_: {"synthetic": True})
    monkeypatch.setattr(advance.handoff, "_seal", lambda *_: "d" * 64)
    monkeypatch.setattr(advance, "_preflight_deployment", lambda *_: history.append("deployment_cpu"))
    monkeypatch.setattr(advance.handoff, "_start", lambda *_: history.append("launch"))
    assert advance.advance(123) == 0
    assert history == [advance.REPORT] + (
        ["freeze", "qualification", "deployment_cpu", "launch"] if status == "plausible_pending_utility" else []
    )
