"""Advance the authored workflow and writer only after their verified gates."""

import argparse
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from scripts import advance_native_state_recovery_001 as handoff
from scripts import native_source_registration as public
from scripts import native_state_recovery_fixture_review_001 as review
from scripts import report_native_state_recovery_fixture_001 as audit
from scripts import run_native_state_recovery_fixture_001 as fixture
from scripts import run_native_state_recovery_fixture_utility_001 as utility
from scripts import run_native_state_recovery_fixture_writer_001 as writer
from scripts import run_native_state_recovery_utility_001 as common
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _exclusive, _same

ERROR = "native_state_recovery_fixture_handoff_not_admitted"


def _require(condition):
    if not condition:
        raise ValueError(ERROR)


def _intent(core, private):
    launch = _decode(core._read_private(private / "parent-launch-intent.json", maximum=64_000))
    _require(public._sha(launch["plan_sha256"])
             and re.fullmatch(r"wll-native-source-pilot-[a-f0-9]{32}\.service", launch["unit"]))
    return launch


def _closed(core, private, launch, expected_pid):
    deadline = time.monotonic() + handoff.WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            pid = _main_pid(launch["unit"])
        except (ValueError, subprocess.TimeoutExpired):
            time.sleep(15)
            continue
        if pid == 0:
            _require(_same(_intent(core, private), launch))
            return handoff._digest(core._read_private(private / "study-result.json", maximum=64_000_000))
        _require(pid == expected_pid)
        time.sleep(15)
    raise ValueError(ERROR)


def _seal(registry, plan):
    root, deployment = WORKSPACE, handoff.DEPLOYMENT
    private = Path(registry.PRIVATE_RUN_ROOT)
    _require(plan["private_run_root"] == str(private) and private.parent == Path(public.PRIVATE_STATE_ROOT)
             and not os.path.lexists(private / "parent-launch-intent.json"))
    bindings = plan["source_sha256"] | plan["input_sha256"]
    staged = {name: public.public_read(root, name) for name in bindings}
    _require(all(handoff._digest(data) == bindings[name] for name, data in staged.items()))
    plan_bytes = _encode(plan)
    sha = handoff._digest(plan_bytes)
    for target in (root, deployment):
        handoff._matches_or_missing(target, registry.PLAN, plan_bytes)
    for name, data in staged.items():
        handoff._matches_or_missing(deployment, name, data)
    for name, data in staged.items():
        handoff._stage(deployment, name, data)
    for target in (root, deployment):
        handoff._stage(target, registry.PLAN, plan_bytes)
        _require(_same(registry.load_registration(target, sha, private=False), plan))
    private.mkdir(mode=0o700, exist_ok=True)
    public._private_directory(private)
    _require(not any(private.iterdir()))
    registry.load_registration(root, sha, private=True)
    return sha


def _start(plan, sha, module):
    _require(_main_pid(handoff.WATCHER) > 0)
    subprocess.run([plan["python"], "-B", "-m", module, "--launch", "--plan-sha256", sha],
                   check=True, capture_output=True, timeout=300)


def _private_manifest(core, private, manifest):
    raw = _encode(manifest)
    path = private / review.MANIFEST
    fd = core._private_directory(private)
    try:
        try:
            with _exclusive(fd, review.MANIFEST) as stream:
                stream.write(raw)
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    finally:
        os.close(fd)
    _require(core._read_private(path, maximum=64_000_000) == raw)
    return handoff._digest(raw)


def advance(fixture_pid):
    reg, queue, episode, _runner = fixture.bindings()
    core = episode.bind(reg, queue).core
    private = Path(reg.PRIVATE_RUN_ROOT)
    launch = _intent(core, private)
    fixture_sha = launch["plan_sha256"]
    result_sha = _closed(core, private, launch, fixture_pid)
    outcome = audit.report(fixture_sha, result_sha)
    handoff._stage(WORKSPACE, "artifacts/native-state-recovery-fixture-terminal-001.json", _encode(outcome))
    gate = outcome["gate"]["status"]
    if gate != "passed":
        _require(gate in {"insufficient", "not_evaluable"})
        print(f"fixture_gate={gate} next_stage=none", flush=True)
        return 0
    proof = writer.qualify(fixture_sha, result_sha)
    handoff._stage(WORKSPACE, writer.QUALIFICATION, _encode(proof))
    registry, _writer_queue, writer_runner = writer.bindings()
    writer_reader = writer.core_for(writer_runner).episode
    plan = registry.build(WORKSPACE)
    writer_sha = _seal(registry, plan)
    _start(plan, writer_sha, writer.MODULE)
    private = Path(registry.PRIVATE_RUN_ROOT)
    launch = _intent(writer_reader, private)
    _require(launch["plan_sha256"] == writer_sha)
    print(f"fixture_gate=passed next_stage={writer.ID} plan_sha256={writer_sha} unit={launch['unit']}", flush=True)
    writer_result = _closed(writer_reader, private, launch, _main_pid(launch["unit"]))
    outcome = review.report(writer_sha, writer_result)
    handoff._stage(WORKSPACE, "artifacts/native-state-recovery-fixture-writer-terminal-001.json", _encode(outcome))
    if outcome["status"] != "plausible_pending_utility":
        print(f"writer_status={outcome['status']} next_stage=none", flush=True)
        return 0
    manifest = review.prepare_manifest(writer_sha, writer_result)["manifest"]
    manifest_sha = _private_manifest(writer_reader, private, manifest)
    proof = common.qualification(manifest, manifest_sha, lineage=utility.LINEAGE)
    registry = common.bindings(utility.LINEAGE)[0]
    raw = _encode(proof)
    handoff._stage(WORKSPACE, registry.QUALIFICATION_PROOF, raw)
    plan = utility.build_registration(WORKSPACE, handoff._digest(raw))
    utility_sha = _seal(registry, plan)
    _start(plan, utility_sha, utility.LINEAGE.module)
    print(f"writer_status=plausible_pending_utility next_stage={registry.ID} plan_sha256={utility_sha}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fixture-pid", type=int, required=True)
    args = parser.parse_args()
    _require(args.fixture_pid > 0 and socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
             and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and os.environ.get("TMPDIR") == public.PRIVATE_STATE_ROOT + "/runtime-tmp"
             and handoff.DEPLOYMENT.resolve() == handoff.DEPLOYMENT)
    return advance(args.fixture_pid)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(ERROR) from None
