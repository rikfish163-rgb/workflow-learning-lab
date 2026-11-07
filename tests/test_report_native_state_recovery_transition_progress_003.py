from types import SimpleNamespace

import pytest

from scripts import report_native_state_recovery_transition_progress_003 as progress
from scripts.native_appworld_worker_client import _encode
from wll.protocol import fingerprint


def test_live_finish_requires_matching_cancel_record_and_no_heartbeat(tmp_path, monkeypatch):
    queue = SimpleNamespace(REMOTE=tmp_path)
    job = {"id": "synthetic", "max_runtime_seconds": 100}
    info = {"status": "finished", "started_epoch": 100, "finished_epoch": 150}
    fence = {"boot_id": "boot", "cancel_epoch": 152}
    cancel = {"reason": "controller_stop", "job_sha256": fingerprint(job), "epoch": 151}
    monkeypatch.setattr(progress, "public_read", lambda *_: _encode(cancel))
    assert progress._durable_finish(queue, job, info, None, "boot") is False
    assert progress._durable_finish(queue, job, info, fence, "boot") is True
    cancel["epoch"] = 153
    with pytest.raises(ValueError):
        progress._durable_finish(queue, job, info, fence, "boot")
    cancel["epoch"] = 151
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/gpu-idle-heartbeat-synthetic").touch()
    with pytest.raises(ValueError):
        progress._durable_finish(queue, job, info, fence, "boot")


def test_stable_journal_read_retries_transient_append(monkeypatch):
    attempts = []

    def read(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise ValueError("changed during live read")
        return b"complete"

    monkeypatch.setattr(progress.time, "sleep", lambda _: None)
    assert progress._study_journal(read, "journal", bytes.decode) == "complete"
    assert attempts == ["journal", "journal"]
