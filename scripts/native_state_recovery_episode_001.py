"""One fresh native state-recovery episode and post-close evidence binding.

No registry, queue job, model, worker, world, scorer, or private file is opened
at import. A separately frozen registration and queue must be supplied to
``bind``. The unchanged controller receipt remains the execution artifact;
authenticated local witness evidence is written separately after worker exit.
No official score or ground truth is loaded or used for witness selection.
"""

import ast
import hashlib
import os
import re
from dataclasses import asdict
from pathlib import Path

from scripts import native_appworld_actor_004 as actor
from scripts import native_appworld_controller_006 as controller
from scripts import native_appworld_model_client_004 as model
from scripts import native_source_episode_006 as source6
from scripts import native_state_recovery_evidence_001 as witness
from scripts import native_state_refusal_client_001 as refusal_client
from scripts.native_appworld_actor import _validate_reply
from scripts.native_appworld_worker_client import (
    ASSET_ROOT,
    PRIVATE_STATE_ROOT,
    PYTHON,
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
    _same,
)
from scripts.native_source_registration import public_read
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = Path("/home/hts/workflow-learning-lab")
ID = "qwen38-native-state-recovery-source-001"
PLAN = "artifacts/qwen38-native-state-recovery-source-plan-001.json"
DESIGN = "artifacts/native-state-recovery-source-design-001.json"
DESIGN_SHA256 = "a06bc3bd325224a72c034fae5dda7c2355bfb330196bbc21457e031f7df36d44"
TASK_IDS = ("22cc237_1", "27e1026_1", "287e338_1", "60d0b5b_1", "692c77d_1", "82e2fac_1",
            "b0a8eae_1", "c901732_1", "cf6abd2_1", "d0b1f43_1")
MODEL_ID = "qwen38-b3-matched-b3-001"
SOURCE006_SHA256 = "64769c575a6160806b84579daf55abc547b90fdfbd09eb729f49c21ce6ae6ebb"
LEGACY_SHA256 = "6b034e2b0c53050832c43384b1e3c4bc5853f5c8e940bbe52c348a0d3ce36e1b"
PILOT004_SHA256 = "76e3ae6db7607a28744369e3eb3b8d8e4deab5850e98881964147d4e8d3cc6aa"
REFUSAL_CLIENT_SHA256 = "e456059a5758321a95f870c713ba7e0a3d1b396bff4b570aaa3b7b57fb6b2d30"
WITNESS_SHA256 = "c1f2628979d913d3e573e042dfb77f103f3172064e3cfc6f319c739b10248987"
SOURCE_PINS = {
    "scripts/native_source_episode_006.py": SOURCE006_SHA256,
    "scripts/native_source_episode.py": LEGACY_SHA256,
    "scripts/run_native_source_pilot_004.py": PILOT004_SHA256,
    "scripts/native_state_refusal_client_001.py": REFUSAL_CLIENT_SHA256,
    "scripts/native_state_recovery_evidence_001.py": WITNESS_SHA256,
    "scripts/native_appworld_actor_004.py": "1fe2877b59a08484a4323cff3ef7e5e18cb5a06f41eb578e67c01ca2a55f7266",
    "scripts/native_appworld_controller_006.py": "7b8ff449823c79aeee6279616846c4a751a8b35008cb3fc9b428d2387635b4e1",
    "scripts/native_appworld_model_client_004.py": "ef4fc8817a8e7bfa0efb4168031e6c700ebb7633202017d42fc10e4761f6583a",
    **refusal_client.SOURCE_SHA256,
}
SELF_SOURCES = ("scripts/native_state_recovery_episode_001.py", "tests/test_native_state_recovery_episode_001.py")
EVIDENCE_NAME = "state-recovery-evidence.json"
MAX_DOCUMENT = 64_000_000
_ERROR = "native_state_recovery_episode_invalid"


class EvidenceError(ValueError):
    """Content-free evidence, configuration, or source binding failure."""


def _require(value):
    if not value:
        raise EvidenceError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def limits():
    return actor.ActorLimits(max_generations=60, max_actions=60, max_api_requests=60,
        max_input_tokens=15360, max_new_tokens=1024, max_prompt_tokens_total=921600,
        max_completion_tokens_total=61440, wall_seconds=600)


def _verified_sources():
    for name, expected in SOURCE_PINS.items():
        _require(_hash(public_read(ROOT, name)) == expected)
    for name, module in (("scripts/native_source_episode_006.py", source6),
                         ("scripts/native_appworld_actor_004.py", actor),
                         ("scripts/native_appworld_controller_006.py", controller),
                         ("scripts/native_appworld_model_client_004.py", model),
                         ("scripts/native_state_recovery_evidence_001.py", witness),
                         ("scripts/native_state_refusal_client_001.py", refusal_client)):
        _require(Path(module.__file__).resolve() == ROOT / name)
    refusal_client.verify_sources()


def _one(rows, event):
    found = [row for row in rows if type(row) is dict and row.get("event") == event]
    _require(len(found) == 1)
    return found[0]


def _actor_evidence(receipt, controller_events, initial_context):
    from scripts import run_native_source_pilot_004 as pilot

    _require(Path(pilot.__file__).resolve() == ROOT / "scripts/run_native_source_pilot_004.py"
             and _hash(public_read(ROOT, "scripts/run_native_source_pilot_004.py")) == PILOT004_SHA256)
    return pilot._get_core().actor_evidence(receipt, controller_events, expected_context=initial_context)


def _snapshot(receipt, run_dir):
    snap = receipt.get("snapshot")
    _require(type(snap) is dict and set(snap) == {"unit_name", "db_path", "files", "cost", "manifest_sha256"})
    digest = _hash(_encode({key: value for key, value in snap.items() if key != "manifest_sha256"}))
    _require(type(snap["manifest_sha256"]) is str and snap["manifest_sha256"] == digest
             and type(snap["files"]) is list and bool(snap["files"]))
    task, experiment = receipt["task_id"], receipt["experiment_name"]
    expected_db = ASSET_ROOT / "experiments" / "outputs" / experiment / "tasks" / task / "dbs"
    _require(snap["db_path"] == str(expected_db))
    worker_dir = Path(run_dir) / "worker"
    unit_digest = _hash((task + experiment + str(worker_dir)).encode())[:16]
    match = re.fullmatch(r"wll-native-worker-" + unit_digest + r"-([a-f0-9]{32})\.service", snap["unit_name"])
    _require(match is not None)
    for item in snap["files"]:
        _require(type(item) is dict and set(item) == {"path", "size", "sha256"}
                 and type(item["path"]) is str and item["path"] and os.path.normpath(item["path"]) == item["path"]
                 and not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts
                 and type(item["size"]) is int and item["size"] >= 0
                 and type(item["sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) is not None)
    _require([item["path"] for item in snap["files"]] == sorted(set(item["path"] for item in snap["files"])))
    return snap, match.group(1)


def _launch(launch, receipt, run_dir, unit, owner):
    _require(type(launch) is dict and set(launch) == {"event", "unit", "argv"}
             and launch.get("event") == "launch_intent" and launch.get("unit") == unit)
    argv = launch.get("argv")
    _require(type(argv) is list and all(type(item) is str for item in argv)
             and argv and argv[0] == "/usr/bin/systemd-run")
    for flag in ("--user", "--pipe", "--wait", "--collect", "--quiet", "--unit=" + unit,
                 "--description=wll-native-worker:" + owner, "--working-directory=" + str(WORKSPACE),
                 "--property=MemoryMax=3G", "--property=CPUQuota=200%", "--property=TasksMax=128",
                 "--property=NoNewPrivileges=yes", "--property=RestrictAddressFamilies=AF_UNIX",
                 "--property=RuntimeMaxSec=780", "--property=KillMode=control-group"):
        _require(argv.count(flag) == 1)
    bootstrap = ("import os,sys;assert sys.flags.no_user_site and sys.flags.safe_path;"
                 "assert not sys.flags.ignore_environment and os.environ.get('PYTHONHASHSEED')=='17';"
                 "sys.path.insert(0," + repr(str(WORKSPACE)) + ");"
                 "from scripts.native_state_refusal_worker_001 import main;raise SystemExit(main())")
    _require(argv.count("--") == 1 and argv[argv.index("--") + 1:argv.index("--") + 3] == ["/usr/bin/env", "-i"]
             and argv.count("-c") == 1 and argv[argv.index("-c") + 1:argv.index("-c") + 2] == [bootstrap]
             and argv[argv.index("-c") - 4:argv.index("-c")] == [PYTHON, "-s", "-P", "-B"]
             and all(argv.count(value) == 1 for value in ("CUDA_VISIBLE_DEVICES=", "TMPDIR=" + str(RUNTIME_TMP),
                                                          "PYTHONHASHSEED=17", "PYTHONDONTWRITEBYTECODE=1")))
    for flag, expected in (("--task-id", receipt["task_id"]), ("--experiment-name", receipt["experiment_name"]),
                           ("--run-dir", str(Path(run_dir) / "worker")), ("--max-actions", "60"),
                           ("--max-api-requests", "60")):
        _require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                 and argv[argv.index(flag) + 1] == expected)


def _journal(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT and raw.endswith(b"\n"))
    values = [_decode(line) for line in raw.splitlines()]
    _require(len(values) <= 100000 and all(type(row) is dict and type(row.get("event")) is str for row in values))
    return values


def authenticate_evidence(receipt, controller_events, worker_events, transport_events, worker_status,
                          fault_provenance, *, initial_context, run_dir):
    """Authenticate closed-worker journals and return a private local witness.

    This is a structural reconciliation, not an independent verifier of OS
    provenance: the caller must source these files from the closed owned worker
    using the registered private reader, source hashes, and resource controls.
    """
    try:
        _require(type(receipt) is dict and set(receipt) == {
                     "status", "task_id", "experiment_name", "is_policy_episode", "actor_outcome", "snapshot",
                     "worker_exit_code", "replay_outcome", "worker_profile", "new_GPU_jobs_started_by_controller",
                     "official_scores_computed", "elapsed_seconds"}
                 and receipt.get("status") == "saved_and_closed"
                 and receipt.get("is_policy_episode") is True and type(receipt.get("worker_exit_code")) is int
                 and receipt["worker_exit_code"] == 0 and receipt.get("replay_outcome") is None
                 and type(receipt.get("official_scores_computed")) is int and receipt["official_scores_computed"] == 0
                 and type(receipt.get("new_GPU_jobs_started_by_controller")) is int
                 and receipt["new_GPU_jobs_started_by_controller"] == 0
                 and type(receipt.get("task_id")) is str and receipt["task_id"] in TASK_IDS
                 and type(receipt.get("experiment_name")) is str and type(receipt.get("worker_profile")) is dict
                 and type(receipt.get("actor_outcome")) is dict and set(receipt["actor_outcome"]) == {
                     "status", "stop_reason", "cost", "outcome_unknown", "unknown_cost"}
                 and type(receipt["actor_outcome"].get("cost")) is dict and set(receipt["actor_outcome"]["cost"]) == {
                     "generations", "prompt_tokens", "completion_tokens", "action_attempts",
                     "api_invocations", "upstream_requests"}
                 and type(controller_events) is list and type(worker_events) is list and type(transport_events) is list
                 and type(initial_context) is dict and set(initial_context) == {
                     "metadata_sha256", "messages_sha256", "input_tokens"}
                 and all(type(initial_context[key]) is str and re.fullmatch(r"[a-f0-9]{64}", initial_context[key])
                         is not None for key in ("metadata_sha256", "messages_sha256"))
                 and type(initial_context["input_tokens"]) is int and 1 <= initial_context["input_tokens"] <= 15360)
        path = Path(run_dir)
        _require(path.is_absolute() and ".." not in path.parts and str(path) == os.fspath(run_dir))
        index = TASK_IDS.index(receipt["task_id"])
        match = re.fullmatch(rf"unit-{index:02d}-first_attempt-([a-f0-9]{{32}})", path.name)
        _require(match is not None and receipt["experiment_name"] ==
                 f"native-state-recovery-001-{index:02d}-first_attempt-{match.group(1)}")
        intent = _one(controller_events, "controller_intent")
        _require(intent.get("task_id") == receipt["task_id"] and intent.get("experiment_name") == receipt["experiment_name"]
                 and _same(intent.get("limits"), asdict(limits())) and type(intent.get("seed")) is int
                 and intent["seed"] == 17 and all(type(intent.get(key)) is int and intent[key] == expected
                     for key, expected in (("setup_seconds", 120), ("cleanup_seconds", 60), ("worker_wall_seconds", 780)))
                 and intent.get("is_policy_episode") is True and intent.get("review_hint_present") is False
                 and _same(_one(controller_events, "controller_stop").get("result"), receipt))
        start = _one(controller_events, "actor_start")
        _require(type(start.get("seed")) is int and start["seed"] == 17 and start.get("review_hint_index") is None)
        _metadata, records, generations = _actor_evidence(receipt, controller_events, initial_context)
        _require(type(records) is list and type(generations) is list and len(records) == len(generations) <= 60)
        intents = [row for row in controller_events if row.get("event") == "generation_intent"]
        _require(all(type(row.get("seed")) is int and row["seed"] == 17 + i
                     and type(row.get("max_new_tokens")) is int and row["max_new_tokens"] == 1024
                     and type(row.get("prompt_tokens")) is int and 1 <= row["prompt_tokens"] <= 15360
                     for i, row in enumerate(intents))
                 and (not intents or intents[0].get("prompt_tokens") == initial_context["input_tokens"]))
        n = len(records)
        outcome = receipt["actor_outcome"]
        _require(all(type(outcome["cost"].get(key)) is int and 0 <= outcome["cost"][key] <= maximum
                     for key, maximum in (("generations", 60), ("prompt_tokens", 921600),
                                          ("completion_tokens", 61440), ("action_attempts", 60),
                                          ("api_invocations", 60), ("upstream_requests", 60))))
        cost = {key: outcome["cost"][key] for key in ("action_attempts", "api_invocations", "upstream_requests")}
        _require(type(cost["action_attempts"]) is int and cost["action_attempts"] == n)
        snapshot, owner = _snapshot(receipt, path)
        _require(_same(snapshot["cost"], cost) and _same(_one(controller_events, "execution_snapshot").get("snapshot"), snapshot))
        expected_status = {"state": "closed", "task_id": receipt["task_id"],
            "experiment_name": receipt["experiment_name"], "max_actions": 60, "max_api_requests": 60,
            "cost": cost, "last_successful_save_through_attempt": n, "current_state_saved": True,
            "score_eligible": True, "outcome_unknown": False, "unknown_cost": False}
        _require(_same(worker_status, expected_status) and len(worker_events) == 2 * n + 2
                 and _same(worker_events[-2:], [{"event": "save_attempt"},
                                               {"event": "save_result", "success": True, "saved_through_attempt": n}]))
        previous = {"action_attempts": 0, "api_invocations": 0, "upstream_requests": 0}
        for i, (record, generation) in enumerate(zip(records, generations, strict=True)):
            attempt, result = worker_events[2 * i:2 * i + 2]
            _require(_same(attempt, {"event": "attempt", "attempt": i + 1, "action": record["action"]})
                     and type(result) is dict and set(result) == {"event", "attempt", "reply"}
                     and result["event"] == "result" and type(result["attempt"]) is int and result["attempt"] == i + 1
                     and _same(result["reply"], generation["reply"])
                     and _validate_reply(result["reply"], previous, i + 1, limits()))
            previous = result["reply"]["cost"]
        _require(_same(previous, cost))
        _require(len(transport_events) == 2 * n + 10)
        _launch(transport_events[0], receipt, path, snapshot["unit_name"], owner)
        _require(type(transport_events[1]) is dict and set(transport_events[1]) == {"event", "pid"}
                 and transport_events[1].get("event") == "launch_result"
                 and type(transport_events[1].get("pid")) is int and transport_events[1]["pid"] > 0)
        expected_rpc = [( {"op": "metadata"}, _one(controller_events, "metadata_result")["reply"] )]
        expected_rpc.extend(({"op": "dispatch", "action": record["action"]}, generation["reply"])
                            for record, generation in zip(records, generations, strict=True))
        expected_rpc += [({"op": "save"}, _one(controller_events, "save_result")["reply"]),
                         ({"op": "close"}, _one(controller_events, "close_result")["reply"])]
        for i, (request, reply) in enumerate(expected_rpc):
            _require(_same(transport_events[2 + 2 * i], {"event": "request_intent", "request": request})
                     and _same(transport_events[3 + 2 * i], {"event": "request_result", "reply": reply}))
        _require(_same(transport_events[-2], {"event": "exit_result", "returncode": 0})
                 and _same(transport_events[-1], {"event": "snapshot_result", "manifest": snapshot}))
        meta_reply = expected_rpc[0][1]
        _require(type(meta_reply) is dict and set(meta_reply) == {"ok", "result", "cost", "freshness"}
            and meta_reply.get("ok") is True and _same(meta_reply.get("result"), _metadata)
            and _same(meta_reply.get("cost"), {
            "action_attempts": 0, "api_invocations": 0, "upstream_requests": 0})
            and _same(meta_reply.get("freshness"), {"last_successful_save_through_attempt": None,
                "current_state_saved": False, "score_eligible": False, "outcome_unknown": False,
                "unknown_cost": False}))
        fresh = {"last_successful_save_through_attempt": n, "current_state_saved": True,
                 "score_eligible": False, "outcome_unknown": False, "unknown_cost": False}
        _require(_same(expected_rpc[-2][1], {"ok": True, "result": {"db_path": snapshot["db_path"]},
                                             "cost": cost, "freshness": fresh}))
        _require(_same(expected_rpc[-1][1], {"ok": True, "result": None, "cost": cost,
                                             "freshness": {**fresh, "score_eligible": True}}))
        decoded = witness._decode_records(records)
        if fault_provenance is None:
            _require(not any(value is not None and witness._target_write(value) is not None for value, _obs in decoded)
                     and not any(obs.get("message") == witness.FAULT_MESSAGE for _value, obs in decoded))
            fault_provenance = {"policy": witness.POLICY, "fault_index": None, "action_sha256": None,
                                "injected_count": 0, "backend_dispatches_at_refusal": None}
            origin = "authenticated_absence_after_close"
        else:
            _require(witness._valid_fault(fault_provenance) and fault_provenance["injected_count"] == 1)
            index = fault_provenance["fault_index"]
            _require(index < n and decoded[index][0] is not None and witness._target_write(decoded[index][0]) is not None
                     and fingerprint(decoded[index][0]) == fault_provenance["action_sha256"]
                     and _same(decoded[index][1], {"ok": False, "error_type": "ToolDispatchError",
                                                   "message": witness.FAULT_MESSAGE}))
            origin = "closed_worker_fault_receipt"
        result = witness.classify(records, fault_provenance)
        return {"id": "native-state-recovery-closed-evidence-001", "private": True,
                "fault_provenance_origin": origin, "fault_provenance": fault_provenance,
                "records_sha256": fingerprint(records), "witness": result}
    except Exception:
        raise EvidenceError(_ERROR) from None


class _IdentityProxy:
    def __init__(self, actual):
        self.actual = actual
        self.ID, self.PLAN = source6.ID, source6.PLAN

    def __getattr__(self, name):
        return getattr(self.actual, name)


def _configuration(core, registration, queue, config, root, loader):
    _require(type(config) is dict and set(config) == core._CONFIG_KEYS and core._sha(config["plan_sha256"])
             and type(config["source_index"]) is int and 0 <= config["source_index"] < len(TASK_IDS)
             and config["phase"] == "first_attempt" and config["prefix_binding"] is None)
    index = config["source_index"]
    plan = loader(root, config["plan_sha256"], private=False)
    raw = registration.public_read(root, DESIGN)
    _require(_hash(raw) == DESIGN_SHA256 and plan["input_sha256"].get(DESIGN) == DESIGN_SHA256
             and _same(plan.get("design"), _decode(raw)) and _same(plan.get("task_ids"), list(TASK_IDS))
             and plan.get("id") == ID and plan.get("status") == "registered"
             and plan.get("model_arm") == "B3" and plan.get("model_id") == MODEL_ID
             and type(plan.get("port")) is int and plan["port"] == registration.PORT
             and plan.get("python") == PYTHON and plan.get("tmp_dir") == str(RUNTIME_TMP)
             and plan.get("private_run_root") == str(core.PRIVATE_RUN_ROOT)
             and plan.get("automatic_retry") is False
             and all(plan.get(key) is False for key in ("writer_authorized", "training_authorized",
                                                        "utility_authorized", "dev_or_test_authorized"))
             and all(plan.get("source_sha256", {}).get(name) == digest for name, digest in SOURCE_PINS.items()))
    _require(all(plan["source_sha256"].get(name) == _hash(registration.public_read(root, name))
                 for name in SELF_SOURCES))
    job = config["job"]
    _require(type(job) is dict and _same(job, queue.make_job(index, job.get("owner"), config["plan_sha256"])))
    path = core._absolute(config["run_dir"])
    _require(path.parent == core.PRIVATE_RUN_ROOT
             and path.name == f"unit-{index:02d}-first_attempt-{job['owner']}")
    core._backend(plan, config["expected_backend"])
    context = config["initial_context"]
    _require(type(context) is dict and set(context) == core._CONTEXT_KEYS
             and core._sha(context["metadata_sha256"]) and core._sha(context["messages_sha256"])
             and type(context["input_tokens"]) is int and 1 <= context["input_tokens"] <= 15360)
    row = core._context_row(root, plan, index)
    _require(all(_same(context[key], row[key]) for key in core._CONTEXT_KEYS))
    return plan, limits(), path, None, None, None


def _legacy_run(core):
    raw = public_read(ROOT, "scripts/native_source_episode.py")
    _require(_hash(raw) == LEGACY_SHA256)
    source = raw.decode("utf-8")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_episode"]
    _require(len(functions) == 1 and not functions[0].decorator_list)
    text = ast.get_source_segment(source, functions[0])
    old, new = "native-source-001-", "native-state-recovery-001-"
    _require(type(text) is str and text.count(old) == 1)
    exec(compile(text.replace(old, new), str(ROOT / "scripts/native_source_episode.py"), "exec", dont_inherit=True),
         core.__dict__)


def _optional_private(core, path):
    directory = core._private_directory(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(directory)
    return core._read_private(path, maximum=4096)


def _collect_closed(core, run_dir, receipt, initial_context):
    path = Path(run_dir)
    names = {"execution_receipt": path / "execution-receipt.json", "controller_journal": path / "controller.jsonl",
             "worker_journal": path / "worker/actions.jsonl", "worker_status": path / "worker/status.json",
             "transport_journal": path / "worker/transport.jsonl"}
    raw = {key: core._read_private(value, maximum=MAX_DOCUMENT) for key, value in names.items()}
    fault_path = path / "worker/fault-provenance.json"
    fault_raw = _optional_private(core, fault_path)
    _require(raw["execution_receipt"].endswith(b"\n") and _same(_decode(raw["execution_receipt"]), receipt)
             and raw["worker_status"].endswith(b"\n") and (fault_raw is None or fault_raw.endswith(b"\n")))
    result = authenticate_evidence(receipt, _journal(raw["controller_journal"]), _journal(raw["worker_journal"]),
        _journal(raw["transport_journal"]), _decode(raw["worker_status"]),
        None if fault_raw is None else _decode(fault_raw), initial_context=initial_context, run_dir=path)
    for key, value in names.items():
        _require(core._read_private(value, maximum=MAX_DOCUMENT) == raw[key])
    _require(_optional_private(core, fault_path) == fault_raw)
    result["private_files_sha256"] = {str(names[key]): _hash(value) for key, value in raw.items()}
    if fault_raw is not None:
        result["private_files_sha256"][str(fault_path)] = _hash(fault_raw)
    directory = core._private_directory(path)
    try:
        with core._exclusive(directory, EVIDENCE_NAME) as stream:
            core._write(stream, result)
        os.fsync(directory)
    finally:
        os.close(directory)
    return result


class BoundEpisode:
    def __init__(self, core, registration, queue):
        self.core, self.registration, self.queue = core, registration, queue

    def run_episode(self, config_path, expected_config_sha256, *, root=WORKSPACE,
                    _load_plan=None, _model_factory=None, _run_task=None, _queue_status=None):
        _require(Path(root) == WORKSPACE and os.environ.get("TMPDIR") == str(RUNTIME_TMP))
        receipt = self.core.run_episode(config_path, expected_config_sha256, root=root, _load_plan=_load_plan,
            _model_factory=_model_factory, _run_task=_run_task, _queue_status=_queue_status)
        if receipt.get("status") == "saved_and_closed":
            raw = self.core._read_private(config_path, maximum=4_000_000)
            _require(_hash(raw) == expected_config_sha256)
            config = _decode(raw)
            _collect_closed(self.core, config["run_dir"], receipt, config["initial_context"])
        return receipt


def bind(registration, queue):
    """Bind a separately registered source and queue; starts no operations."""
    _verified_sources()
    private = Path(registration.PRIVATE_RUN_ROOT)
    _require(type(registration.PRIVATE_RUN_ROOT) is str and private.is_absolute()
             and str(private) == registration.PRIVATE_RUN_ROOT and ".." not in private.parts
             and PRIVATE_STATE_ROOT in private.parents and queue.REMOTE == DEPLOY_ROOT)
    _require(all(getattr(value, "ID", None) == ID and getattr(value, "PLAN", None) == PLAN
                 for value in (registration, queue))
             and tuple(registration.TASK_IDS) == TASK_IDS and registration.MODEL_ARM == "B3"
             and registration.MODEL_ID == MODEL_ID and type(registration.PORT) is int
             and 1024 <= registration.PORT <= 65535
             and callable(registration.load_registration) and callable(registration.public_read)
             and callable(registration.static_backend) and callable(queue.make_job)
             and callable(queue.queue_operation) and callable(queue.ready_path))
    modules = {"registration": _IdentityProxy(registration), "queue": _IdentityProxy(queue),
               "actor": actor, "controller": controller, "model": model}
    core = source6._isolated(_modules=modules)
    core.registration, core.queue, core.REMOTE = registration, queue, queue.REMOTE
    core.PRIVATE_RUN_ROOT = Path(registration.PRIVATE_RUN_ROOT)
    core.NativeWorkerClient = refusal_client.NativeWorkerClient
    core._configuration = lambda config, root, loader: _configuration(core, registration, queue, config, root, loader)
    _legacy_run(core)
    return BoundEpisode(core, registration, queue)
