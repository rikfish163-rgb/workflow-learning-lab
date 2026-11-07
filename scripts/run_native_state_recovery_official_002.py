"""Start the frozen official panel after the pre-actor port mismatch."""

import argparse
import os
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts import run_native_state_recovery_official_001 as previous
from scripts.native_appworld_scorer import _main_pid, _stable_read
from scripts.native_appworld_worker_client import (
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _private_tree,
)
from scripts.native_source_registration import public_read

ID = "qwen38-native-state-recovery-official-002"
PLAN = "artifacts/qwen38-native-state-recovery-official-model-plan-002.json"
PRIVATE = RUNTIME_TMP / "official-test-normal-002"
MODULE = "scripts.run_native_state_recovery_official_002"
SOURCE = MODULE.replace(".", "/") + ".py"
ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "scripts/run_native_state_recovery_official_001.py"
PREVIOUS_SHA = "faa9b3b9dde998f51ef3b5beb7977d635119c544dfddec5c53740ec2a87b20aa"
FIRST_REPORT = "artifacts/native-state-recovery-official-terminal-001.json"
FIRST_REPORT_SHA = "1db933aad0ba761734cb9c705ebefb48c39f5ebe82368d46c3e0c11ef7e0b1fa"
FIRST_UNIT = "wll-native-source-official-d661a85ed39e6873855be57d5ede39ba.service"
REPORT = "artifacts/native-state-recovery-official-terminal-002.json"
_require = previous._require
_hash = previous._hash


def _prove_unexposed():
    raw = public_read(WORKSPACE, FIRST_REPORT)
    _require(_hash(raw) == FIRST_REPORT_SHA and _main_pid(FIRST_UNIT) == 0)
    terminal = _decode(raw)
    fd = _private_tree(previous.PRIVATE)
    try:
        raw = _stable_read(fd, "study-result.json", previous.frozen.MAX_DOCUMENT)
        journal = previous.frozen._core()._journal(
            _stable_read(fd, "study-journal.jsonl", previous.frozen.MAX_DOCUMENT))
        files = os.listdir(fd)
    finally:
        os.close(fd)
    result = _decode(raw)
    _require(_hash(raw) == terminal["result_sha256"] and result["plan_sha256"] == terminal["plan_sha256"]
             and terminal["parent_terminal"] == "failed" and terminal["audited_finished_leases"] == 1
             and terminal["audited_scores"] == result["dev_or_test_calls"] == 0
             and terminal["audited_actor_cost_subtotal"]["generations"] == 0
             and result["executions"] == result["costs"]["episodes"] == result["costs"]["scorers"] == []
             and len(result["summaries"]) == 54 and all(row["status"] == "unstarted" for row in result["summaries"])
             and not any(event["event"] in ("GPU_ready", "unit_launch_intent") for event in journal)
             and not any(name.startswith(("unit-", "score-", "config-")) for name in files))


@cache
def bindings():
    raw = public_read(ROOT, PREVIOUS)
    _require(_hash(raw) == PREVIOUS_SHA)
    code = raw.decode()
    for before, after, count in (
        (previous.ID, ID, 1),
        (previous.PLAN, PLAN, 1),
        (previous.MODULE.removeprefix("scripts."), MODULE.removeprefix("scripts."), 2),
        ("PRIVATE = freeze.PRIVATE", 'PRIVATE = RUNTIME_TMP / "official-test-normal-002"', 1),
        ("native-official-001-", "native-official-002-", 1),
        ("PORT=queue.PORT,", 'PORT=_prior()["port"],', 1),
        ("queue = study._queue()", "study = study.bindings()\nqueue = study._queue()", 1),
        ("prior = study._prior()", "prior = study._prior()\n"
         "assert queue.serve.__globals__['registration'].PORT == study._registry().PORT == prior['port']", 1),
    ):
        _require(code.count(before) == count)
        code = code.replace(before, after)
    core = ModuleType("scripts._official_002_execution")
    core.__file__ = str(ROOT / SOURCE)
    exec(compile(code, str(ROOT / PREVIOUS), "exec", dont_inherit=True), vars(core))
    core.SOURCES |= {PREVIOUS}
    _require(core.PRIVATE == PRIVATE and core.ID == ID and core.PLAN == PLAN)
    return core


def report(plan_sha, result_sha):
    from scripts import report_native_state_recovery_official_001 as audit

    core = bindings()
    scores = previous._bind(audit._scores, study=core)
    return previous._bind(audit.report, study=core, _scores=scores)(plan_sha, result_sha)


def _launch_and_audit(sha):
    from scripts.finalize_native_state_recovery_comparisons_001 import (
        WAIT_SECONDS,
        _result_sha,
        _wait,
    )
    from scripts.register_native_state_recovery_authored_study_001 import _publish

    _prove_unexposed()
    unit = bindings().launch(sha)
    print(f"official_unit={unit}", flush=True)
    if pid := _main_pid(unit):
        _wait(unit, pid, WAIT_SECONDS)
    outcome = report(sha, _result_sha(PRIVATE / "study-result.json"))
    _publish(WORKSPACE / REPORT, _encode(outcome) + b"\n", 0o644)
    print(f"official_audit_written={REPORT}", flush=True)


def main():
    core = bindings()
    if sys.argv[1:] == ["--register"]:
        _prove_unexposed()
        PRIVATE.mkdir(mode=0o700, exist_ok=True)
        os.close(_private_tree(PRIVATE))
        print(core.register())
        return 0
    if sys.argv[1:2] == ["--launch-and-audit"]:
        parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
        parser.add_argument("--launch-and-audit", action="store_true", required=True)
        parser.add_argument("--plan-sha256", required=True)
        args = parser.parse_args()
        _launch_and_audit(args.plan_sha256)
        return 0
    if "--launch" in sys.argv[1:]:
        _prove_unexposed()
    return core.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("official_panel_successor_failed") from None
