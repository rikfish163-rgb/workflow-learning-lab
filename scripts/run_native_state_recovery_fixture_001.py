"""Run four authored workflows after a verified insufficient natural source study."""

import argparse
import hashlib
import signal
import sys
from functools import cache
from pathlib import Path

from scripts import native_state_recovery_fixture_episode_001 as fixture
from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts import report_native_state_recovery_source_003 as natural_report
from scripts import run_native_state_recovery_source_003 as natural
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _same
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _sha, public_read

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-state-recovery-fixture-001"
PLAN = "artifacts/qwen38-native-state-recovery-fixture-plan-001.json"
MODULE = "scripts.run_native_state_recovery_fixture_001"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-fixture-001"
DESIGN = "artifacts/native-state-recovery-fixture-contexts-CPU-001.json"
DESIGN_SHA256 = "b80ca80e49afffaf83126b49329e5fbd198768df60af0a54648c21bc4122778a"
CPU_AUDIT = "artifacts/native-state-recovery-fixture-actor-audit-CPU-001.json"
CPU_AUDIT_SHA256 = "7d75078e5c317b6cbbd1b54553f809a26b7522962b32a4e07fa97f2ff835cbb4"
PREDECESSOR = "artifacts/native-state-recovery-fixture-activation-001.json"
PREDECESSOR_ID = "native-state-recovery-fixture-activation-001"
SOURCE_SHA256 = natural_report.PLAN_SHA256
EXPERIMENT = "native-state-recovery-fixture-001-"
SOURCES = {"scripts/run_native_state_recovery_fixture_001.py", "scripts/native_state_recovery_fixture_episode_001.py",
           "scripts/native_state_recovery_fixture_protocol_001.py", "scripts/report_native_state_recovery_source_003.py",
           "scripts/native_state_recovery_writer_evidence_001.py", "tests/test_native_state_recovery_fixture_episode_001.py",
           "scripts/audit_native_state_recovery_fixture_actor_001.py", "scripts/probe_native_state_recovery_fixture_actor_001.py",
           "scripts/report_native_state_recovery_fixture_001.py"}


def _require(value):
    if not value:
        raise ValueError("native_state_recovery_fixture_not_admitted")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def qualify(result_sha256):
    """Prove the original cohort has ended with no eligible experience input."""
    _require(_sha(result_sha256))
    report = natural_report.report(terminal_sha256=result_sha256)
    gate = report["gate"]
    _require(report["coordinator_running"] is False and report["audited_scored_and_released"] == 10
             and gate["status"] == "insufficient" and type(gate["witness_scenarios"]) is int
             and 0 <= gate["witness_scenarios"] < 2)
    return {"id": PREDECESSOR_ID, "source_plan_sha256": SOURCE_SHA256, "source_result_sha256": result_sha256,
            "source_denominator": 10, "source_gate_status": "insufficient", "witness_scenarios": gate["witness_scenarios"]}


def _valid_predecessor(value):
    _require(type(value) is dict and set(value) == {"id", "source_plan_sha256", "source_result_sha256",
        "source_denominator", "source_gate_status", "witness_scenarios"}
        and value["id"] == PREDECESSOR_ID and value["source_plan_sha256"] == SOURCE_SHA256
        and _sha(value["source_result_sha256"]) and value["source_denominator"] == 10
        and value["source_gate_status"] == "insufficient" and type(value["witness_scenarios"]) is int
        and 0 <= value["witness_scenarios"] < 2)
    return value


def _design(root):
    raw = public_read(root, DESIGN)
    _require(_hash(raw) == DESIGN_SHA256)
    design = _decode(raw)
    _require(design["source_denominator"] == len(protocol.CASES) == 4 and design["minimum_qualified_scenarios"] == 2
             and design["status"] == "prospective_not_registered" and design["original_task_success_scored"] is False
             and design["engineering_probe_sha256"] == protocol.PROBE_SHA256
             and design["protocol_source_sha256"] == _public_hash(root, "scripts/native_state_recovery_fixture_protocol_001.py")
             and all(type(design[key]) is int and design[key] == 0 for key in ("GPU_jobs", "model_calls", "ground_truth_load_attempts")))
    rows = design["rows"]
    fixture._bound_rows(rows)
    _require(all(all(_sha(row[key]) for key in ("worker_metadata_sha256", "actor_metadata_sha256", "messages_sha256"))
        and type(row["input_tokens"]) is int and 1 <= row["input_tokens"] <= 15360 for row in rows))
    return design


@cache
def bindings():
    base, queue, _episode, _runner = natural.bindings.__wrapped__()
    rows = _design(ROOT)["rows"]
    template_ids = tuple(row["template_task_id"] for row in rows)
    prefix = ("native-state-recovery-001-", EXPERIMENT)
    episode = natural.prior._load("native_state_recovery_episode_001", ((*prefix, 2),), ID=ID, PLAN=PLAN,
                                 DESIGN=DESIGN, DESIGN_SHA256=DESIGN_SHA256, TASK_IDS=template_ids)
    fixture.install(episode, rows)
    # Keep the authenticated natural registration accessible as an independent
    # lineage. Rebind a separate registration module for the fixture study.
    reg = natural.prior._load("native_state_recovery_registration_001", episode=episode, ID=ID, PLAN=PLAN,
                             TASK_IDS=template_ids, PRIVATE_RUN_ROOT=PRIVATE_RUN_ROOT, PORT=base.PORT)

    def build(root):
        previous = base.load_registration(root, SOURCE_SHA256, private=False)
        design = _design(root)
        _require(_same(design["rows"], rows))
        roles, _ = reg.public._cohort(root)
        origins = {task[:-2] for task in template_ids}
        _require(len(origins) == len(template_ids) and set(template_ids) <= set(roles["origin"])
                 and not set(template_ids) & set(previous["task_ids"])
                 and not origins & {task[:-2] for task in roles["utility"]})
        proof_raw = reg.public_read(root, PREDECESSOR)
        proof = _valid_predecessor(_decode(proof_raw))
        audit_raw = reg.public_read(root, CPU_AUDIT)
        _require(_hash(audit_raw) == CPU_AUDIT_SHA256)
        cpu_audit = _decode(audit_raw)
        _require(cpu_audit["status"] == "passed" and cpu_audit["actor_policy_episode"] is False
                 and cpu_audit["metadata_substitution_verified"] is True and cpu_audit["fixture_target_verified"] is True
                 and cpu_audit["worker_release_verified"] is True
                 and all(cpu_audit[key] == 0 for key in ("model_calls", "GPU_jobs", "official_scorer_calls")))
        sources = previous["source_sha256"] | {name: _public_hash(root, name) for name in SOURCES}
        _require(sources["scripts/probe_native_state_recovery_fixture_actor_001.py"] == cpu_audit["probe_source_sha256"]
                 and sources["scripts/audit_native_state_recovery_fixture_actor_001.py"] == cpu_audit["audit_source_sha256"])
        inputs = previous["input_sha256"] | {DESIGN: DESIGN_SHA256, PREDECESSOR: _hash(proof_raw),
            natural.PLAN: SOURCE_SHA256, protocol.PROBE: protocol.PROBE_SHA256, CPU_AUDIT: CPU_AUDIT_SHA256}
        _require(all(_public_hash(root, name) == sha for name, sha in inputs.items()))
        return {**previous, "id": ID, "design": design, "task_ids": list(template_ids),
                "batch_task_ids": [[task] for task in template_ids], "private_run_root": PRIVATE_RUN_ROOT,
                "total_gpu_seconds": previous["gpu_seconds"] * len(template_ids), "source_sha256": sources,
                "input_sha256": inputs, "context_proof": {"path": DESIGN, "sha256": DESIGN_SHA256},
                "fixture_activation": proof}

    reg.build = build
    queue.ID, queue.PLAN, queue.registration = ID, PLAN, reg
    queue.GUARD_NAME = "native-state-recovery-fixture-operations-001.lock"
    make_job = queue.make_job

    def job(index, owner, sha):
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    runner = natural.prior._load("run_native_state_recovery_source_001", (
        (*prefix, 1), ("scripts.run_native_state_recovery_source_001", MODULE, 2),
    ), ID=ID, registration=reg, queue=queue, episode=episode)
    runner.PARENT_SECONDS = len(rows) * (86400 + 7200) + 600
    return reg, queue, episode, runner


def gate(report, tasks):
    slots = report["slots"]
    costs = report["costs"]
    known = (report["status"] == "complete" and [row["template_task_id"] for row in slots] == list(tasks)
             and len(costs["episodes"]) == len(costs["GPU_jobs"]) == len(tasks) == 4
             and [row["source_index"] for row in costs["episodes"]] == list(range(4))
             and [row["job"]["batch_index"] for row in costs["GPU_jobs"]] == list(range(4))
             and all(row["status"] == "known" and row["witness"]["status"] in {"qualified", "ineligible"} for row in slots)
             and all(row["counts_complete"] is True and row["unknown_cost"] is False for row in costs["episodes"])
             and all(row["release"]["verified"] is True and row["release"]["runtime_overrun"] is False
                     for row in costs["GPU_jobs"]))
    n = len({row["template_task_id"][:-2] for row in slots if row.get("fixture_target_verified") is True
             and row["witness"]["status"] == "qualified"})
    return {"status": ("passed" if n >= 2 else "insufficient") if known else "not_evaluable",
            "witness_scenarios": n, "source_denominator": 4}


def _run_slot(runtime, core, episode, plan, sha, index, context_row, job, service, guard, report):
    task = context_row["template_task_id"]
    owner = job["owner"]
    run_dir = Path(plan["private_run_root"]) / f"unit-{index:02d}-first_attempt-{owner}"
    experiment = f"{EXPERIMENT}{index:02d}-first_attempt-{owner}"
    unit = f"wll-native-source-{index:02d}-first_attempt-{owner}.service"
    context = fixture._case_context(context_row, protocol.CASES[index])
    binding = runtime.write(f"config-{index:02d}-first_attempt.json", {"plan_sha256": sha, "source_index": index,
        "phase": "first_attempt", "run_dir": str(run_dir), "job": job, "expected_backend": service["backend"],
        "prefix_binding": None, "initial_context": context})
    parent = None
    started = runtime.now()
    cost = {"source_index": index, "parent_wall_seconds": None, "counts_complete": False, "unknown_cost": True}
    try:
        parent, _wall = core._unit(runtime, "episode", unit, ["--config", binding["path"],
            "--config-sha256", binding["sha256"]], guard=guard)
        data = runtime.artifacts(run_dir)
        receipt = data["receipt"]
        recomputed = fixture.collect(episode, core.episode, run_dir, receipt, context)
        raw = runtime.read(run_dir / episode.EVIDENCE_NAME)
        _require(_same(_decode(raw), recomputed) and _main_pid(unit) == _main_pid(receipt["snapshot"]["unit_name"]) == 0)
        cost.update(core._costs(receipt, data["journal"]))
        _require(cost["counts_complete"] is True and cost["unknown_cost"] is False)
        return {"status": "known", "receipt_sha256": data["receipt_binding"]["sha256"],
                "witness_sha256": _hash(raw), "witness": recomputed["fixture"]["witness"],
                "fixture_target_verified": recomputed["fixture"]["fixture_target_verified"],
                "actor_stop_reason": receipt["actor_outcome"]["stop_reason"]}
    finally:
        if cost["unknown_cost"]:
            try:
                raw = runtime.read(run_dir / "controller.jsonl")
                rows = core._journal(raw[:raw.rfind(b"\n") + 1])
                cost.update(core._costs({}, rows), counts_complete=False, unknown_cost=True)
            except Exception:
                pass
        try:
            _require(runtime.cleanup_worker(run_dir, task, experiment)["verified"] is True)
        finally:
            if parent is not None:
                _require(runtime.stop_unit(parent)["verified"] is True)
            cost["parent_wall_seconds"] = runtime.now() - started
            report["costs"]["episodes"].append(cost)


def _preflight(plan):
    proof = _valid_predecessor(plan["fixture_activation"])
    _require(_same(qualify(proof["source_result_sha256"]), proof))


def run(plan_sha):
    reg, queue, episode, runner = bindings()
    core = runner._core()
    runtime = core._Runtime(WORKSPACE)
    plan = runtime.load_plan(plan_sha)
    _preflight(plan)
    rows = plan["design"]["rows"]
    slots = [{"source_index": i, "template_task_id": row["template_task_id"], "status": "unstarted",
              "witness": {"status": "unstarted"}} for i, row in enumerate(rows)]
    report = {"id": ID, "scope": "authored_workflows_on_real_AppWorld_tools", "status": "unknown",
              "plan_sha256": plan_sha, "slots": slots, "costs": {"episodes": [], "GPU_jobs": []},
              "official_scorer_calls": 0, "automatic_retry": False}
    runtime.begin(plan, plan_sha, slots)
    slot = None
    try:
        for index, (row, slot) in enumerate(zip(rows, slots, strict=True)):
            job = queue.make_job(index, runtime.owner(), plan_sha)
            with runner._lease(runtime, core, job, report) as (service, guard):
                slot["status"] = "running"
                slot.update(_run_slot(runtime, core, episode, plan, plan_sha, index, row, job, service, guard, report))
                runtime.record({"event": "fixture_finished", "slot": slot})
        _require(sum(row["release"]["GPU_seconds"] for row in report["costs"]["GPU_jobs"]) <= plan["total_gpu_seconds"])
        report["status"] = "complete"
    except BaseException as error:
        if slot is not None and slot["status"] == "running":
            slot.update(status="unknown", witness={"status": "unknown"})
        core._capture_initiating_exception(runtime, error)
    finally:
        report["source_gate"] = gate(report, reg.TASK_IDS)
        try:
            runtime.record({"event": "study_stop", "report": report})
            runtime.write("study-result.json", report)
        finally:
            runtime.close()
    return int(report["status"] != "complete")


def main():
    if "--serve" in sys.argv[1:]:
        return bindings()[1].main()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--config")
    parser.add_argument("--config-sha256")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    reg, queue, episode, runner = bindings()
    if args.config:
        _require(_sha(args.config_sha256) and args.plan_sha256 is None)
        receipt = episode.bind(reg, queue).run_episode(args.config, args.config_sha256)
        return int(receipt["status"] != "saved_and_closed")
    _require(_sha(args.plan_sha256) and args.config_sha256 is None)
    if args.launch:
        plan = reg.load_registration(WORKSPACE, args.plan_sha256, private=True)
        _preflight(plan)
        print(runner._core().launch_pilot(args.plan_sha256, root=WORKSPACE)["unit"])
        return 0

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    return run(args.plan_sha256)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_fixture_not_admitted\n")
        raise SystemExit(2) from None
