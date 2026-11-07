"""Wait for one recorded V27 coordinator and persist its independent audit."""

import argparse
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from scripts import advance_native_state_recovery_001 as staging
from scripts import run_native_state_recovery_transition_utility_003 as utility
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import _sha

REPORT = "artifacts/native-state-recovery-transition-utility-terminal-003.json"
ERROR = "native_state_recovery_transition_utility_terminal_audit_failed"


def _wait(unit, expected_pid, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            pid = _main_pid(unit)
        except (ValueError, subprocess.TimeoutExpired):
            time.sleep(30)
            continue
        if pid == 0:
            return
        if pid != expected_pid:
            raise ValueError(ERROR)
        time.sleep(30)
    raise ValueError(ERROR)


def finalize(expected_pid):
    registry, _queue, runner = utility.bindings()
    reader = runner._services(WORKSPACE)[1]
    root = Path(registry.PRIVATE_RUN_ROOT)

    def read(name):
        return reader._read_private(root / name, maximum=64_000_000)

    launch = _decode(read("parent-launch-intent.json"))
    unit, sha = launch["unit"], launch["plan_sha256"]
    if not (_sha(sha) and re.fullmatch(r"wll-native-source-experience-utility-[a-f0-9]{32}\.service", unit)):
        raise ValueError(ERROR)
    registry.load_registration(WORKSPACE, sha, private=False)
    _wait(unit, expected_pid, runner.PARENT_SECONDS + 120)
    if not _same(launch, _decode(read("parent-launch-intent.json"))):
        raise ValueError(ERROR)
    result = staging._digest(read("study-result.json"))
    outcome = utility.report(sha, result)
    staging._stage(WORKSPACE, REPORT, _encode(outcome))
    print(f"utility_audit_written={REPORT}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    if not (args.wait_pid > 0 and socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""):
        raise ValueError(ERROR)
    return finalize(args.wait_pid)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(ERROR) from None
