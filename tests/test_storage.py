import io
from types import SimpleNamespace

import pytest

from wll.appworld import StderrTail
from wll.storage import atomic_json, require_space


def test_low_disk_space_fails_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr("wll.storage.shutil.disk_usage", lambda p: SimpleNamespace(free=1))
    output = tmp_path / "new" / "run"
    with pytest.raises(OSError, match="free space"):
        require_space(output)
    assert not output.exists()


def test_worker_stderr_is_bounded_in_memory_and_keeps_recent_error():
    reader = StderrTail(io.BytesIO(b"x" * 1_000_000 + b"recent-error"))
    reader.thread.join(timeout=2)
    assert not reader.thread.is_alive()
    assert len(reader.tail()) <= 65536
    assert reader.tail().endswith(b"recent-error")


def test_failed_manifest_update_preserves_previous_valid_copy(monkeypatch, tmp_path):
    output = tmp_path / "manifest.json"
    output.write_text('{"status":"running"}\n')

    def disk_full(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("disk full")

    monkeypatch.setattr("wll.storage.json.dump", disk_full)
    with pytest.raises(OSError, match="disk full"):
        atomic_json(output, {"status": "complete"})
    assert output.read_text() == '{"status":"running"}\n'
    assert list(tmp_path.iterdir()) == [output]
