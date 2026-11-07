from types import SimpleNamespace

import pytest

from scripts import advance_native_state_recovery_fixture_001 as run


@pytest.mark.parametrize("gate", ["insufficient", "not_evaluable"])
def test_fixture_terminal_without_qualified_evidence_never_launches_writer(monkeypatch, gate):
    staged = []
    episode = SimpleNamespace(bind=lambda *_: SimpleNamespace(core=object()))
    reg = SimpleNamespace(PRIVATE_RUN_ROOT="/synthetic")
    monkeypatch.setattr(run.fixture, "bindings", lambda: (reg, None, episode, None))
    monkeypatch.setattr(run, "_intent", lambda *_: {"plan_sha256": "1" * 64})
    monkeypatch.setattr(run, "_closed", lambda *_: "2" * 64)
    monkeypatch.setattr(run.audit, "report", lambda *_: {"gate": {"status": gate}})
    monkeypatch.setattr(run.handoff, "_stage", lambda _root, name, _raw: staged.append(name))
    monkeypatch.setattr(run.writer, "qualify", lambda *_: pytest.fail("writer admission attempted"))
    assert run.advance(123) == 0
    assert staged == ["artifacts/native-state-recovery-fixture-terminal-001.json"]


@pytest.mark.parametrize("writer_status", ["abstained", "rejected", "plausible_pending_utility"])
def test_utility_starts_only_after_exact_review_is_frozen(monkeypatch, writer_status):
    starts, staged = [], []
    writer_sha, utility_sha = "3" * 64, "4" * 64
    fixture_reader, writer_reader = object(), object()
    episode = SimpleNamespace(bind=lambda *_: SimpleNamespace(core=fixture_reader))
    fixture_registry = SimpleNamespace(PRIVATE_RUN_ROOT="/fixture")
    writer_registry = SimpleNamespace(PRIVATE_RUN_ROOT="/writer", build=lambda _: {"stage": "writer"})
    writer_runner = object()
    utility_registry = SimpleNamespace(ID="utility", QUALIFICATION_PROOF="utility-proof")
    manifest = {"example": "private"}
    monkeypatch.setattr(run.fixture, "bindings", lambda: (fixture_registry, None, episode, None))
    def intent(core, path):
        is_writer = path.name == "writer"
        assert core is (writer_reader if is_writer else fixture_reader)
        return {"unit": "synthetic", "plan_sha256": writer_sha if is_writer else "1" * 64}

    def closed(core, path, launch, _pid):
        assert launch == intent(core, path)
        return "2" * 64

    monkeypatch.setattr(run, "_intent", intent)
    monkeypatch.setattr(run, "_closed", closed)
    monkeypatch.setattr(run.audit, "report", lambda *_: {"gate": {"status": "passed"}})
    monkeypatch.setattr(run.handoff, "_stage", lambda _root, name, raw: staged.append((name, raw)))
    monkeypatch.setattr(run.writer, "qualify", lambda *_: {"qualified": True})
    monkeypatch.setattr(run.writer, "bindings", lambda: (writer_registry, None, writer_runner))
    monkeypatch.setattr(run.writer, "core_for", lambda _runner: SimpleNamespace(episode=writer_reader))
    monkeypatch.setattr(run, "_seal", lambda reg, _plan: writer_sha if reg is writer_registry else utility_sha)
    monkeypatch.setattr(run, "_start", lambda _plan, sha, _module: starts.append(sha))
    monkeypatch.setattr(run, "_main_pid", lambda _: 456)
    monkeypatch.setattr(run.review, "report", lambda *_: {"status": writer_status})
    monkeypatch.setattr(run.review, "prepare_manifest", lambda *_: {"manifest": manifest})

    def freeze(core, private, prepared):
        assert starts == [writer_sha] and core is writer_reader and private.name == "writer" and prepared is manifest
        return "5" * 64

    monkeypatch.setattr(run, "_private_manifest", freeze)
    monkeypatch.setattr(run.common, "qualification", lambda *_args, **_kwargs: {"public": True})
    monkeypatch.setattr(run.common, "bindings", lambda _: (utility_registry,))
    monkeypatch.setattr(run.utility, "build_registration", lambda *_: {"stage": "utility"})
    assert run.advance(123) == 0
    assert starts == ([writer_sha, utility_sha] if writer_status == "plausible_pending_utility" else [writer_sha])
    assert all(b'"private"' not in raw for _path, raw in staged)
