"""Reconcile a verified child exiting between systemd and procfs reads."""

import errno
import re
import subprocess
from pathlib import Path

_PROPERTIES = ("Id", "MainPID", "ExecMainPID", "InvocationID", "LoadState", "ActiveState", "SubState",
               "Result", "ExecMainCode", "ExecMainStatus")
_PROC_FILES = ("stat", "cmdline", "environ", "status", "cgroup")
ERROR = "native_state_recovery_child_exit_unverified"


def _snapshot(unit):
    result = subprocess.run(
        ["/usr/bin/systemctl", "--user", "show", unit, *["--property=" + name for name in _PROPERTIES]],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=True,
    )
    if len(result.stdout) > 8192:
        raise ValueError(ERROR)
    items = [line.split("=", 1) for line in result.stdout.decode().splitlines()]
    values = dict(items)
    if len(items) != len(_PROPERTIES) or set(values) != set(_PROPERTIES):
        raise ValueError(ERROR)
    return values


def _same_closed_child(unit, pid, snapshot):
    expected = {"Id": unit, "MainPID": "0", "ExecMainPID": str(pid), "LoadState": "loaded",
                "ActiveState": "active", "SubState": "exited", "Result": "success",
                "ExecMainCode": "1", "ExecMainStatus": "0"}
    return (all(snapshot[key] == value for key, value in expected.items())
            and re.fullmatch("[0-9a-f]{32}", snapshot["InvocationID"]) is not None
            and snapshot["InvocationID"] != "0" * 32)


def unit_status_with_exit_recheck(original, spec):
    """Retry one verified process disappearing, then require the exact unit's clean exit."""
    try:
        return original(spec)
    except FileNotFoundError as error:
        identity = spec.get("process_identity")
        if type(identity) is not dict or type(identity.get("pid")) is not int or identity["pid"] <= 0:
            raise
        pid = identity["pid"]
        proc = Path("/proc") / str(pid)
        if error.errno != errno.ENOENT or error.filename not in {str(proc / name) for name in _PROC_FILES}:
            raise
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot != identity["boot_id"]:
            raise
        unit = spec["unit"]
        snapshot = _snapshot(unit)
        if not _same_closed_child(unit, pid, snapshot):
            raise
        # The original terminal path still checks all sandbox properties and
        # requires the recorded control group to be empty.
        checked = original(spec)
        if (checked.get("verified") is not True or checked.get("running") is not False
                or checked.get("success") is not True or checked.get("main_pid") != 0
                or checked.get("exit_code") != 0 or _snapshot(unit) != snapshot):
            raise ValueError(ERROR) from None
        return checked


def bind_runtime(runner):
    """Apply the recheck and safe branch diagnostics to a new isolated runner."""
    services, run_arm = runner._services, runner._run_arm

    def checked_services(root):
        runtime, reader = services(root)
        status = runtime.unit_status
        runtime.unit_status = lambda spec: unit_status_with_exit_recheck(status, spec)
        return runtime, reader

    def audited_arm(runtime, plan, sha, bundle, job, service, index, arm, row, guard, report, expected_metadata):
        try:
            return run_arm(runtime, plan, sha, bundle, job, service, index, arm, row, guard, report, expected_metadata)
        except Exception as error:
            first = error
            for _ in range(8):
                if first.__context__ is None:
                    break
                first = first.__context__
            frame = first.__traceback__
            while frame is not None and frame.tb_next is not None:
                frame = frame.tb_next
            failure_function = frame.tb_frame.f_code.co_name if frame is not None else "unknown"
            try:
                runtime.record({"event": "branch_error_observed", "pair_index": index, "arm": arm,
                                "error_type": type(first).__name__, "function": failure_function})
            except Exception:
                pass
            raise

    runner._services, runner._run_arm = checked_services, audited_arm
