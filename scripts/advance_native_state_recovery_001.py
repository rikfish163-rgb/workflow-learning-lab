"""Hand off the original source cohort to its registered next stage."""

import argparse
import hashlib
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from scripts import native_source_registration as public
from scripts import report_native_state_recovery_source_003 as audit
from scripts import run_native_state_recovery_fixture_001 as fixture
from scripts import run_native_state_recovery_source_003 as source
from scripts import run_native_state_recovery_writer_001 as writer
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same

DEPLOYMENT = Path("/home/hts/workflow-learning-lab")
WATCHER = "wll-gpu-idle-watch-b3-state-recovery.service"
WAIT_SECONDS = 6 * 60 * 60
ERROR = "native_state_recovery_handoff_not_admitted"


def _require(condition):
    if not condition:
        raise ValueError(ERROR)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _matches_or_missing(root, name, data):
    path = root / name
    _require(path.parent.resolve() == path.parent and path.parent.is_dir())
    _require(not os.path.lexists(path) or public.public_read(root, name) == data)


def _stage(root, name, data):
    _matches_or_missing(root, name, data)
    try:
        with (root / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    _require(public.public_read(root, name) == data)


def _terminal(expected_pid):
    reg, queue, episode, _ = source.bindings()
    core = episode.bind(reg, queue).core
    private = Path(reg.PRIVATE_RUN_ROOT)
    launch = _decode(core._read_private(private / "parent-launch-intent.json", maximum=64_000))
    unit = launch["unit"]
    _require(launch["plan_sha256"] == audit.PLAN_SHA256 and re.fullmatch(
        r"wll-native-source-pilot-[a-f0-9]{32}\.service", unit,
    ))
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            pid = _main_pid(unit)
        except (ValueError, subprocess.TimeoutExpired):
            time.sleep(15)
            continue
        if pid == 0:
            data = core._read_private(private / "study-result.json", maximum=episode.MAX_DOCUMENT)
            return _digest(data)
        _require(pid == expected_pid)
        time.sleep(15)
    raise ValueError(ERROR)


def _route(result_sha, gate):
    if gate == "insufficient":
        return fixture, fixture.PREDECESSOR, fixture.qualify(result_sha)
    _require(gate == "passed")
    return writer, writer.QUALIFICATION, writer.qualify(result_sha)


def handoff(result_sha):
    report = audit.report(terminal_sha256=result_sha)
    gate = report["gate"]["status"]
    if gate == "not_evaluable":
        print("source_gate=not_evaluable next_stage=none", flush=True)
        return 0
    _require(report["audited_scored_and_released"] == 10 and _main_pid(WATCHER) > 0)
    route, proof_path, proof = _route(result_sha, gate)
    private = Path(route.PRIVATE_RUN_ROOT)
    _require(private.parent == Path(public.PRIVATE_STATE_ROOT)
             and not os.path.lexists(private / "parent-launch-intent.json"))
    proof_bytes = _encode(proof)
    for root in (WORKSPACE, DEPLOYMENT):
        _matches_or_missing(root, proof_path, proof_bytes)
    _stage(WORKSPACE, proof_path, proof_bytes)
    reg = route.bindings()[0]
    plan = reg.build(WORKSPACE)
    _require(plan["private_run_root"] == str(private))
    files = plan["source_sha256"] | plan["input_sha256"]
    _require(all(public._public_hash(WORKSPACE, path) == digest for path, digest in files.items()))
    staged = {path: public.public_read(WORKSPACE, path) for path in files}
    plan_bytes = _encode(plan)
    plan_sha = _digest(plan_bytes)
    _matches_or_missing(WORKSPACE, route.PLAN, plan_bytes)
    _matches_or_missing(DEPLOYMENT, route.PLAN, plan_bytes)
    for path, data in staged.items():
        _matches_or_missing(DEPLOYMENT, path, data)
    for path, data in staged.items():
        _stage(DEPLOYMENT, path, data)
    for root in (WORKSPACE, DEPLOYMENT):
        _stage(root, route.PLAN, plan_bytes)
        _require(_same(reg.load_registration(root, plan_sha, private=False), plan))
    private.mkdir(mode=0o700, exist_ok=True)
    public._private_directory(private)
    _require(not any(private.iterdir()))
    reg.load_registration(WORKSPACE, plan_sha, private=True)
    _require(_main_pid(WATCHER) > 0)
    launched = subprocess.run(
        [plan["python"], "-B", "-m", route.MODULE, "--launch", "--plan-sha256", plan_sha],
        check=True, capture_output=True, text=True, timeout=300,
    )
    unit = launched.stdout.strip()
    _require(re.fullmatch(r"wll-native-source-pilot-[a-f0-9]{32}\.service", unit))
    print(f"source_gate={gate} next_stage={route.ID} plan_sha256={plan_sha} unit={unit}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-pid", required=True, type=int)
    args = parser.parse_args()
    _require(args.source_pid > 0 and socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
             and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and os.environ.get("TMPDIR") == public.PRIVATE_STATE_ROOT + "/runtime-tmp"
             and DEPLOYMENT.resolve() == DEPLOYMENT and DEPLOYMENT != WORKSPACE
             and shutil.disk_usage(DEPLOYMENT).free >= public.MINIMUM_FREE_BYTES)
    return handoff(_terminal(args.source_pid))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(ERROR) from None
