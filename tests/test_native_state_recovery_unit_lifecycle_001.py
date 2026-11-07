import errno
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_run_native_experience_utility_001 import SHA, Runtime

from scripts import native_state_recovery_unit_lifecycle_001 as lifecycle
from scripts import run_native_experience_utility_001 as utility

BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
SPEC = {"unit": "example.service", "process_identity": {"pid": 71, "boot_id": BOOT}}
EXIT = {"Id": "example.service", "MainPID": "0", "ExecMainPID": "71", "LoadState": "loaded",
        "ActiveState": "active", "SubState": "exited", "Result": "success",
        "ExecMainCode": "1", "ExecMainStatus": "0", "InvocationID": "a" * 32}
STATUS = {"verified": True, "running": False, "success": True, "main_pid": 0, "exit_code": 0}


def _status(path="/proc/71/stat"):
    seen = []

    def original(spec):
        seen.append(spec)
        if len(seen) == 1:
            raise FileNotFoundError(errno.ENOENT, "synthetic procfs race", path)
        return STATUS

    return original, seen


def test_lost_verified_proc_can_be_reconciled_only_as_exact_terminal(monkeypatch):
    original, seen = _status()
    monkeypatch.setattr(lifecycle, "_snapshot", lambda unit: {**EXIT, "Id": unit})
    assert lifecycle.unit_status_with_exit_recheck(original, SPEC) == STATUS
    assert seen == [SPEC, SPEC]


@pytest.mark.parametrize("change", [{"ExecMainPID": "72"}, {"MainPID": "71"},
                                    {"Result": "exit-code"}, {"InvocationID": "0" * 32}])
def test_replacement_and_uncertain_close_remain_errors(monkeypatch, change):
    original, seen = _status()
    monkeypatch.setattr(lifecycle, "_snapshot", lambda _: {**EXIT, **change})
    with pytest.raises(FileNotFoundError):
        lifecycle.unit_status_with_exit_recheck(original, SPEC)
    assert seen == [SPEC]


def test_unrelated_missing_file_cannot_pass_reconciliation(monkeypatch):
    original, seen = _status("/proc/72/stat")
    monkeypatch.setattr(lifecycle, "_snapshot", lambda _: pytest.fail("must not query an unrelated child"))
    with pytest.raises(FileNotFoundError):
        lifecycle.unit_status_with_exit_recheck(original, SPEC)
    assert seen == [SPEC]


def test_invocation_cannot_change_between_terminal_reads(monkeypatch):
    original, _ = _status()
    snapshots = iter([EXIT, {**EXIT, "InvocationID": "b" * 32}])
    monkeypatch.setattr(lifecycle, "_snapshot", lambda _: next(snapshots))
    with pytest.raises(ValueError):
        lifecycle.unit_status_with_exit_recheck(original, SPEC)


def test_prospective_binding_records_error_kind_without_private_message(monkeypatch):
    events = []
    original, _ = _status()
    runtime = SimpleNamespace(unit_status=original, record=events.append)
    monkeypatch.setattr(lifecycle, "_snapshot", lambda _: EXIT)

    def fail(*_):
        raise RuntimeError("synthetic sensitive message")

    runner = SimpleNamespace(_services=lambda _: (runtime, "reader"), _run_arm=fail)
    lifecycle.bind_runtime(runner)
    actual, reader = runner._services("root")
    assert actual.unit_status(SPEC) == STATUS and reader == "reader"
    with pytest.raises(RuntimeError, match="synthetic sensitive message"):
        runner._run_arm(actual, None, None, None, None, None, 2, "control", None, None, None, None)
    assert events == [{"event": "branch_error_observed", "pair_index": 2, "arm": "control",
                       "error_type": "RuntimeError", "function": "fail"}]


@pytest.mark.parametrize(("exec_pid", "expected_scores", "expected_releases"), [(71, 54, 14), (72, 34, 9)])
def test_full_synthetic_schedule_rechecks_only_the_verified_exit(monkeypatch, exec_pid, expected_scores, expected_releases):
    fake = Runtime()
    status = fake.unit_status
    injected = False

    def probe(spec):
        nonlocal injected
        if not injected and spec.get("index") == 17 and spec.get("arm") == "experience":
            injected = True
            spec["process_identity"] = {"pid": 71, "boot_id": BOOT}
            raise FileNotFoundError(errno.ENOENT, "synthetic process exit", "/proc/71/stat")
        return status(spec)

    fake.unit_status = probe
    monkeypatch.setattr(lifecycle, "_snapshot", lambda unit: {**EXIT, "Id": unit, "ExecMainPID": str(exec_pid)})
    monkeypatch.setattr(utility, "_services", lambda _: (fake, None))
    monkeypatch.setattr(utility, "_run_arm", utility._run_arm)
    monkeypatch.setattr(utility, "_make_job", lambda batch, owner, _: {"batch_index": batch, "owner": owner})
    lifecycle.bind_runtime(utility)
    runtime, _reader = utility._services("synthetic")
    result = utility.run_utility(SHA, _runtime=runtime)
    assert injected and len(result["executions"]) == expected_scores
    assert len(result["costs"]["GPU_jobs"]) == expected_releases
    errors = [row[1] for row in fake.calls if row[0] == "record" and row[1].get("event") == "branch_error_observed"]
    if exec_pid == 71:
        assert result["status"] == "complete" and not errors
    else:
        assert result["status"] == "unknown" and result["utility"]["statistics"] is None
        assert errors == [{"event": "branch_error_observed", "pair_index": 17, "arm": "experience",
                           "error_type": "FileNotFoundError", "function": "probe"}]
