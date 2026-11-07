"""Synthetic-only first-attempt and closed-worker evidence authentication."""

import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_appworld_actor_004 as actor
from scripts import native_state_recovery_episode_001 as episode
from scripts import native_state_recovery_evidence_001 as witness
from scripts.native_appworld_worker_client import (
    ASSET_ROOT,
    PYTHON,
    RUNTIME_TMP,
    WORKSPACE,
    _decode,
    _encode,
)
from scripts.verification_token_counter import _message_payload
from wll.protocol import fingerprint


def digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def synthetic_evidence(*, fault=True, prefix_invalid=False, empty=False, private_root=Path("/synthetic/private")):
    task, owner = episode.TASK_IDS[0], "a" * 32
    experiment = f"native-state-recovery-001-00-first_attempt-{owner}"
    run_dir = private_root / ("unit-00-first_attempt-" + owner)
    worker_dir = run_dir / "worker"
    unit_hash = hashlib.sha256((task + experiment + str(worker_dir)).encode()).hexdigest()[:16]
    unit = f"wll-native-worker-{unit_hash}-{'b' * 32}.service"
    metadata = {"instruction": "Synthetic task", "supervisor": {}, "app_descriptions": {}}
    messages = actor.build_actor_messages(metadata)
    initial = {"metadata_sha256": digest(metadata), "messages_sha256": _message_payload(messages)[1],
               "input_tokens": 100}
    cost = {"action_attempts": 0, "api_invocations": 0, "upstream_requests": 0}
    fresh = {"last_successful_save_through_attempt": None, "current_state_saved": False,
             "score_eligible": False, "outcome_unknown": False, "unknown_cost": False}
    metadata_reply = {"ok": True, "result": metadata, "cost": cost.copy(), "freshness": fresh.copy()}
    read = {"name": "todoist__show_task", "arguments": {"access_token": "synthetic", "task_id": 42}}
    write = {"name": "todoist__update_task", "arguments": {
        "access_token": "synthetic", "task_id": 42, "is_completed": True}}
    pairs = [(read, {"ok": True, "result": {"task_id": 42, "is_completed": False}})]
    if empty:
        pairs = []
    if prefix_invalid:
        pairs.insert(0, ("not JSON", {"ok": False, "error_type": "ToolDispatchError",
                                      "message": "synthetic predispatch rejection"}))
    provenance = None
    if fault:
        pairs += [(write, {"ok": False, "error_type": "ToolDispatchError", "message": witness.FAULT_MESSAGE}),
                  (write, {"ok": True, "result": {"message": "synthetic"}}),
                  (read, {"ok": True, "result": {"task_id": 42, "is_completed": True}})]
        provenance = {"policy": witness.POLICY, "fault_index": 2 if prefix_invalid else 1,
                      "action_sha256": fingerprint(write),
                      "injected_count": 1, "backend_dispatches_at_refusal": 0}
    controller = [
        {"event": "controller_intent", "task_id": task, "experiment_name": experiment,
         "limits": asdict(episode.limits()), "seed": 17, "setup_seconds": 120, "cleanup_seconds": 60,
         "worker_wall_seconds": 780, "is_policy_episode": True, "review_hint_present": False},
        {"event": "metadata_result", "reply": copy.deepcopy(metadata_reply)},
        {"event": "actor_start", "messages": copy.deepcopy(messages), "seed": 17,
         "limits": asdict(episode.limits()), "review_hint_index": None},
    ]
    worker, transport = [], []
    argv = ["/usr/bin/systemd-run", "--user", "--pipe", "--wait", "--collect", "--quiet", f"--unit={unit}",
            f"--description=wll-native-worker:{'b' * 32}", f"--working-directory={WORKSPACE}",
            "--property=MemoryMax=3G", "--property=CPUQuota=200%", "--property=TasksMax=128",
            "--property=NoNewPrivileges=yes", "--property=RestrictAddressFamilies=AF_UNIX",
            "--property=RuntimeMaxSec=780", "--property=KillMode=control-group", "--", "/usr/bin/env", "-i",
            "CUDA_VISIBLE_DEVICES=", f"TMPDIR={RUNTIME_TMP}", "PYTHONHASHSEED=17", "PYTHONDONTWRITEBYTECODE=1",
            PYTHON, "-s", "-P", "-B", "-c",
            ("import os,sys;assert sys.flags.no_user_site and sys.flags.safe_path;"
             "assert not sys.flags.ignore_environment and os.environ.get('PYTHONHASHSEED')=='17';"
             "sys.path.insert(0," + repr(str(WORKSPACE)) + ");"
             "from scripts.native_state_refusal_worker_001 import main;raise SystemExit(main())"),
            "--task-id", task, "--experiment-name", experiment, "--run-dir", str(worker_dir),
            "--max-actions", "60", "--max-api-requests", "60"]
    transport.extend([{"event": "launch_intent", "unit": unit, "argv": argv},
                      {"event": "launch_result", "pid": 111},
                      {"event": "request_intent", "request": {"op": "metadata"}},
                      {"event": "request_result", "reply": copy.deepcopy(metadata_reply)}])
    records = []
    for index, (value, observation) in enumerate(pairs):
        text = value if type(value) is str else json.dumps(value)
        cost["action_attempts"] += 1
        if observation.get("error_type") != "ToolDispatchError":
            cost["api_invocations"] += 1
            cost["upstream_requests"] += 1
        reply = {**observation, "cost": cost.copy(), "freshness": fresh.copy()}
        response = {"text": text, "prompt_tokens": 100, "completion_tokens": 1}
        controller.extend([
            {"event": "generation_intent", "messages": copy.deepcopy(messages), "seed": 17 + index,
             "prompt_tokens": 100, "max_new_tokens": 1024},
            {"event": "generation_result", "response": response},
            {"event": "dispatch_intent", "action": text}, {"event": "dispatch_result", "reply": reply},
        ])
        messages.extend([{"role": "assistant", "content": text},
                         {"role": "user", "content": _encode(observation).decode("ascii")}])
        worker += [{"event": "attempt", "attempt": index + 1, "action": text},
                   {"event": "result", "attempt": index + 1, "reply": copy.deepcopy(reply)}]
        transport += [{"event": "request_intent", "request": {"op": "dispatch", "action": text}},
                      {"event": "request_result", "reply": copy.deepcopy(reply)}]
        records.append({"action": text, "observation": observation})
    n = len(pairs)
    outcome = {"status": "incomplete", "stop_reason": "wall_seconds",
        "cost": {"generations": n, "prompt_tokens": 100 * n, "completion_tokens": n, **cost},
        "outcome_unknown": False, "unknown_cost": False}
    db_path = str(ASSET_ROOT / "experiments" / "outputs" / experiment / "tasks" / task / "dbs")
    snapshot = {"unit_name": unit, "db_path": db_path,
                "files": [{"path": "synthetic.db", "size": 1, "sha256": "a" * 64}], "cost": cost.copy()}
    snapshot["manifest_sha256"] = digest(snapshot)
    saved_fresh = {**fresh, "last_successful_save_through_attempt": n, "current_state_saved": True}
    closed_fresh = {**saved_fresh, "score_eligible": True}
    saved = {"ok": True, "result": {"db_path": db_path}, "cost": cost.copy(), "freshness": saved_fresh}
    closed = {"ok": True, "result": None, "cost": cost.copy(), "freshness": closed_fresh}
    worker += [{"event": "save_attempt"}, {"event": "save_result", "success": True, "saved_through_attempt": n}]
    transport += [{"event": "request_intent", "request": {"op": "save"}},
                  {"event": "request_result", "reply": saved},
                  {"event": "request_intent", "request": {"op": "close"}},
                  {"event": "request_result", "reply": closed},
                  {"event": "exit_result", "returncode": 0}, {"event": "snapshot_result", "manifest": snapshot}]
    receipt = {"status": "saved_and_closed", "task_id": task, "experiment_name": experiment,
               "is_policy_episode": True, "actor_outcome": outcome, "snapshot": snapshot,
               "worker_exit_code": 0, "replay_outcome": None, "worker_profile": {"synthetic": True},
               "elapsed_seconds": 30,
               "new_GPU_jobs_started_by_controller": 0, "official_scores_computed": 0}
    controller += [{"event": "actor_stop", "outcome": copy.deepcopy(outcome)},
                   {"event": "actor_result", "outcome": copy.deepcopy(outcome)},
                   {"event": "save_result", "reply": saved}, {"event": "close_result", "reply": closed},
                   {"event": "execution_snapshot", "snapshot": snapshot},
                   {"event": "controller_stop", "result": copy.deepcopy(receipt)}]
    status = {"state": "closed", "task_id": task, "experiment_name": experiment,
              "max_actions": 60, "max_api_requests": 60, "cost": cost.copy(), **closed_fresh}
    return SimpleNamespace(receipt=receipt, controller=controller, worker=worker, transport=transport,
                           status=status, fault=provenance, initial=initial, run_dir=run_dir, records=records)


def collect(value):
    return episode.authenticate_evidence(value.receipt, value.controller, value.worker, value.transport,
        value.status, value.fault, initial_context=value.initial, run_dir=value.run_dir)


def test_closed_evidence_authenticates_exact_records_without_official_score():
    value = synthetic_evidence()
    result = collect(value)
    assert result["private"] is True and result["fault_provenance"] == value.fault
    assert result["witness"]["status"] == "qualified"
    assert result["witness"]["private_evidence"]["indices"] == {
        "before": 0, "fault": 1, "correction": 2, "after": 3}
    assert result["records_sha256"] == fingerprint(value.records)
    assert "official_success" not in str(result) and "synthetic" not in str(result)


def test_absent_fault_is_known_only_without_any_hook_eligible_action():
    value = synthetic_evidence(fault=False)
    result = collect(value)
    assert result["witness"]["status"] == "ineligible" and result["witness"]["reason"] == "no_fault"
    value = synthetic_evidence()
    value.fault = None
    with pytest.raises(episode.EvidenceError):
        collect(value)


def test_closed_zero_action_budget_stop_is_known_no_fault():
    value = synthetic_evidence(fault=False, empty=True)
    result = collect(value)
    assert result["witness"]["status"] == "ineligible" and result["witness"]["reason"] == "no_fault"
    assert result["witness"]["counts"]["records"] == 0


def test_returned_generation_without_dispatch_preserves_known_no_fault():
    value = synthetic_evidence(fault=False)
    prior_intent = next(row for row in value.controller if row["event"] == "generation_intent")
    messages = copy.deepcopy(prior_intent["messages"])
    record = value.records[-1]
    messages.extend([{"role": "assistant", "content": record["action"]},
                     {"role": "user", "content": _encode(record["observation"]).decode("ascii")}])
    stop = next(i for i, row in enumerate(value.controller) if row["event"] == "actor_stop")
    value.controller[stop:stop] = [
        {"event": "generation_intent", "messages": messages, "seed": 18,
         "prompt_tokens": 100, "max_new_tokens": 1024},
        {"event": "generation_result", "response": {"text": "synthetic unused result",
         "prompt_tokens": 100, "completion_tokens": 1}},
    ]
    outcome = value.receipt["actor_outcome"]
    outcome["cost"]["generations"] += 1
    outcome["cost"]["prompt_tokens"] += 100
    outcome["cost"]["completion_tokens"] += 1
    for row in value.controller:
        if row["event"] in {"actor_stop", "actor_result"}:
            row["outcome"] = copy.deepcopy(outcome)
        if row["event"] == "controller_stop":
            row["result"] = copy.deepcopy(value.receipt)
    result = collect(value)
    assert result["witness"]["status"] == "ineligible" and result["witness"]["reason"] == "no_fault"
@pytest.mark.parametrize("fault", [False, True])
def test_rejected_invalid_json_still_consumes_a_bound_dispatcher_attempt(fault):
    value = synthetic_evidence(fault=fault, prefix_invalid=True)
    result = collect(value)
    assert result["witness"]["status"] == ("qualified" if fault else "ineligible")
    if fault:
        assert result["fault_provenance"]["fault_index"] == 2
        assert result["witness"]["private_evidence"]["indices"]["fault"] == 2


@pytest.mark.parametrize("change", ["unclosed", "worker_exit", "bool_exit", "bool_score", "hidden_score", "fault_hash", "fault_index", "worker_action",
                                    "worker_reply", "worker_gap", "extra_rpc", "worker_limit",
                                    "unbound_metadata", "changed_messages", "save_gap", "snapshot", "wrong_launch"])
def test_unclosed_tampered_or_gapped_evidence_is_rejected(change):
    value = synthetic_evidence()
    if change == "unclosed":
        value.receipt["status"] = "unknown"
    elif change == "worker_exit":
        value.receipt["worker_exit_code"] = 1
    elif change == "bool_exit":
        value.receipt["worker_exit_code"] = False
    elif change == "bool_score":
        value.receipt["official_scores_computed"] = False
    elif change == "hidden_score":
        value.receipt["official_success"] = True
    elif change == "fault_hash":
        value.fault["action_sha256"] = "0" * 64
    elif change == "fault_index":
        value.fault["fault_index"] = 2
    elif change == "worker_action":
        value.worker[2]["action"] = "synthetic changed"
    elif change == "worker_reply":
        value.worker[3]["reply"]["message"] = "synthetic changed"
    elif change == "worker_gap":
        value.worker[2]["attempt"] += 1
    elif change == "extra_rpc":
        value.transport.insert(2, {"event": "request_intent", "request": {"op": "invalid"}})
    elif change == "worker_limit":
        value.worker[3]["reply"]["error_type"] = "ActionLimitError"
    elif change == "unbound_metadata":
        value.initial["metadata_sha256"] = "0" * 64
    elif change == "changed_messages":
        value.controller[3]["messages"] = []
    elif change == "save_gap":
        value.worker.pop()
    elif change == "snapshot":
        value.receipt["snapshot"]["manifest_sha256"] = "0" * 64
    else:
        value.transport[0]["argv"][12] = "wrong worker"
    with pytest.raises(episode.EvidenceError):
        collect(value)


@pytest.fixture
def binding_fixture(tmp_path, monkeypatch):
    value = synthetic_evidence()
    design = {"prospective_limits": {"generation_seed": 17}}
    design_raw = _encode(design)
    monkeypatch.setattr(episode, "DESIGN_SHA256", hashlib.sha256(design_raw).hexdigest())
    proof = {"rows": [{"task_id": task, "initial_fits": True, **value.initial} for task in episode.TASK_IDS]}
    proof_raw, proof_path = _encode(proof), "artifacts/synthetic-context-proof.json"
    self_sources = {name: (episode.ROOT / name).read_bytes() for name in episode.SELF_SOURCES}
    gpu_uuid = "GPU-ecc8cd45-fdaa-b5ee-e54a-bf2ef7df8f82"
    backend = {"synthetic": "B3", "gpu_policy": {"hostname": "amax", "physical_gpu_index": 8,
        "gpu_uuid": gpu_uuid, "visible_device_count": 1, "policy_sha256": "c" * 64},
        "loading_peak_reserved_bytes": 1024}
    plan = {"id": episode.ID, "status": "registered", "task_ids": list(episode.TASK_IDS), "design": design,
            "input_sha256": {episode.DESIGN: episode.DESIGN_SHA256, proof_path: hashlib.sha256(proof_raw).hexdigest()},
            "source_sha256": {**episode.SOURCE_PINS, **{name: hashlib.sha256(raw).hexdigest()
                                                          for name, raw in self_sources.items()}}, "context_proof": {
                "path": proof_path, "sha256": hashlib.sha256(proof_raw).hexdigest()},
            "private_run_root": str(tmp_path), "model_arm": "B3", "model_id": episode.MODEL_ID, "port": 18999,
            "python": PYTHON, "tmp_dir": str(RUNTIME_TMP),
            "expected_backend_static": {"B3": {"synthetic": "B3"}}, "gpu_policy_sha256": "c" * 64,
            "gpu_policy": {"hostname": "amax", "allowed_gpus": {"8": gpu_uuid}},
            "automatic_retry": False, "writer_authorized": False, "training_authorized": False,
            "utility_authorized": False, "dev_or_test_authorized": False}

    def read(_root, name):
        return {episode.DESIGN: design_raw, proof_path: proof_raw, **self_sources}[name]

    reg = SimpleNamespace(ID=episode.ID, PLAN=episode.PLAN, TASK_IDS=episode.TASK_IDS, MODEL_ARM="B3",
        MODEL_ID=episode.MODEL_ID, PORT=18999, PRIVATE_RUN_ROOT=str(tmp_path), public_read=read,
        load_registration=lambda *_args, **_kwargs: plan,
        static_backend=lambda obj: {key: item for key, item in obj.items()
                                    if key not in {"gpu_policy", "loading_peak_reserved_bytes"}})
    queue = SimpleNamespace(ID=episode.ID, PLAN=episode.PLAN, REMOTE=episode.DEPLOY_ROOT,
        make_job=lambda index, owner, sha: {"index": index, "owner": owner, "plan_sha256": sha},
        queue_operation=lambda *_args, **_kwargs: None, ready_path=lambda *_args: None)
    owner, sha = "b" * 32, "d" * 64
    monkeypatch.setattr(episode, "PRIVATE_STATE_ROOT", tmp_path.parent)
    config = {"plan_sha256": sha, "source_index": 0, "phase": "first_attempt",
              "run_dir": str(tmp_path / ("unit-00-first_attempt-" + owner)),
              "job": queue.make_job(0, owner, sha), "expected_backend": backend,
              "prefix_binding": None, "initial_context": value.initial}
    return SimpleNamespace(reg=reg, queue=queue, plan=plan, config=config, metadata={
        "instruction": "Synthetic task", "supervisor": {}, "app_descriptions": {}}, backend=backend)


def test_binding_reuses_real_actor_controller_model_and_refusal_worker(binding_fixture):
    from scripts import native_appworld_controller_006 as controller
    from scripts import native_appworld_model_client_004 as model
    from scripts import native_state_refusal_client_001 as refusal_client

    f = binding_fixture
    bound = episode.bind(f.reg, f.queue)
    assert bound.core.actor is actor and bound.core.run_task is controller.run_task
    assert bound.core.NativeModel is model.NativeModel and bound.core.NativeWorkerClient is refusal_client.NativeWorkerClient
    plan, configured, path, metadata, records, hint = bound.core._configuration(
        f.config, Path("/synthetic"), f.reg.load_registration)
    assert plan == f.plan and configured == episode.limits() and str(path) == f.config["run_dir"]
    assert metadata is records is hint is None


@pytest.mark.parametrize("change", ["phase", "prefix", "index", "source_pin", "plan_task", "seed_design"])
def test_registered_configuration_rejects_changed_scope_or_replay(binding_fixture, change):
    f = binding_fixture
    bound = episode.bind(f.reg, f.queue)
    if change == "phase":
        f.config["phase"] = "control"
    elif change == "prefix":
        f.config["prefix_binding"] = {"path": "synthetic", "sha256": "0" * 64}
    elif change == "index":
        f.config["source_index"] = 10
    elif change == "source_pin":
        f.plan["source_sha256"]["scripts/native_source_episode_006.py"] = "0" * 64
    elif change == "plan_task":
        f.plan["task_ids"][0] = "0000000_1"
    else:
        f.plan["design"]["prospective_limits"]["generation_seed"] = 18
    with pytest.raises((episode.EvidenceError, ValueError)):
        bound.core._configuration(f.config, Path("/synthetic"), f.reg.load_registration)


def test_first_attempt_uses_bound_worker_and_collects_only_after_return(binding_fixture, monkeypatch):
    f = binding_fixture
    bound = episode.bind(f.reg, f.queue)
    core = bound.core
    config_path = str(Path(f.reg.PRIVATE_RUN_ROOT) / "synthetic-config.json")
    raw = _encode(f.config) + b"\n"
    monkeypatch.setattr(core, "_read_private", lambda path, **_kw: raw if str(path) == config_path else None)
    monkeypatch.setattr(core.socket, "gethostname", lambda: "amax")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TMPDIR", str(RUNTIME_TMP))
    monkeypatch.setattr(core, "_queue_binding", lambda *_args: ({}, {}, ("fixed",)))
    calls = {"closed": False}

    class Model:
        def __init__(self, *_args, **kwargs):
            self.max_tokens, self.deadline = kwargs["max_tokens"], kwargs["deadline"]
            calls["model"] = kwargs

        def check_health(self):
            return {"model": episode.MODEL_ID, "ready": True, "busy": False, "max_context": 16384,
                    "max_output_tokens": 1024, "backend": f.backend}

    class Worker:
        def __init__(self, *args, **kwargs):
            calls["worker"] = (args, kwargs)

        def request(self, request, **_kwargs):
            assert request == {"op": "metadata"}
            return {"ok": True, "result": f.metadata}

    monkeypatch.setattr(core, "NativeWorkerClient", Worker)

    def run_task(task, experiment, path, **kwargs):
        calls["controller"] = kwargs
        assert kwargs["source_metadata"] is kwargs["prefix_records"] is kwargs["review_hint"] is None
        worker = kwargs["_worker_factory"](task, experiment, path / "worker", max_actions=60,
            max_api_requests=60, rpc_timeout=60, wall_seconds=780)
        worker.request({"op": "metadata"})
        kwargs["model"].check_health()
        calls["closed"] = True
        return {"task_id": task, "status": "saved_and_closed"}

    def collected(_core, run_dir, receipt, initial):
        assert calls["closed"] is True
        calls["collection"] = (run_dir, receipt, initial)

    monkeypatch.setattr(episode, "_collect_closed", collected)
    result = bound.run_episode(config_path, hashlib.sha256(raw).hexdigest(), _model_factory=Model,
                               _run_task=run_task)
    assert result["status"] == "saved_and_closed" and calls["collection"][0] == f.config["run_dir"]
    assert calls["controller"]["seed"] == 17 and calls["controller"]["limits"] == episode.limits()
    assert calls["worker"][1] == {"max_actions": 60, "max_api_requests": 60,
                                  "rpc_timeout": 60, "wall_seconds": 780}
    assert calls["model"]["max_tokens"] == 1024 and calls["model"]["context_tokens"] == 16384


def test_unknown_episode_never_reads_or_exposes_a_fault_receipt(binding_fixture, monkeypatch):
    f = binding_fixture
    bound = episode.bind(f.reg, f.queue)
    core = bound.core
    config_path = str(Path(f.reg.PRIVATE_RUN_ROOT) / "synthetic-config.json")
    raw = _encode(f.config) + b"\n"
    monkeypatch.setattr(core, "_read_private", lambda path, **_kw: raw if str(path) == config_path else None)
    monkeypatch.setattr(core.socket, "gethostname", lambda: "amax")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TMPDIR", str(RUNTIME_TMP))
    monkeypatch.setattr(core, "_queue_binding", lambda *_args: ({}, {}, ("fixed",)))
    monkeypatch.setattr(episode, "_collect_closed", lambda *_args: pytest.fail("worker was not closed"))

    class Model:
        def __init__(self, *_args, **kwargs):
            self.max_tokens, self.deadline = kwargs["max_tokens"], kwargs["deadline"]

    result = bound.run_episode(config_path, hashlib.sha256(raw).hexdigest(), _model_factory=Model,
        _run_task=lambda task, _experiment, _path, **_kwargs: {"task_id": task, "status": "unknown"})
    assert result["status"] == "unknown"


def test_real_controller_actor_and_collector_reconcile_fake_closed_worker(binding_fixture, monkeypatch):
    """Exercise the inherited lifecycle with synthetic I/O and a real actor."""
    from scripts import native_appworld_controller_006 as controller
    from scripts import native_appworld_worker_client as worker_client

    f = binding_fixture
    private_root = Path(f.reg.PRIVATE_RUN_ROOT)
    value = synthetic_evidence(private_root=private_root)
    owner = "a" * 32
    f.config["job"] = f.queue.make_job(0, owner, f.config["plan_sha256"])
    f.config["run_dir"] = str(value.run_dir)
    bound = episode.bind(f.reg, f.queue)
    core = bound.core
    controller_core = controller._isolated()
    monkeypatch.setattr(controller_core, "PRIVATE_STATE_ROOT", private_root)
    monkeypatch.setattr(worker_client, "PRIVATE_STATE_ROOT", private_root)
    monkeypatch.setattr(core.socket, "gethostname", lambda: "amax")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TMPDIR", str(RUNTIME_TMP))
    monkeypatch.setattr(core, "_queue_binding", lambda *_args: ({}, {}, ("fixed",)))
    config_path = private_root / "config.json"
    raw = _encode(f.config) + b"\n"
    config_path.write_bytes(raw)
    config_path.chmod(0o600)
    calls = {"generated": 0, "dispatched": 0, "released": False}

    class Model:
        def __init__(self, *_args, **kwargs):
            self.max_tokens, self.deadline = kwargs["max_tokens"], kwargs["deadline"]

        def check_health(self):
            return {"model": episode.MODEL_ID, "ready": True, "busy": False, "max_context": 16384,
                    "max_output_tokens": 1024, "backend": f.backend}

        def count_input_tokens(self, _messages):
            return 100 if calls["generated"] < len(value.records) else 15361

        def generate(self, _messages, seed):
            index = calls["generated"]
            assert seed == 17 + index
            calls["generated"] += 1
            return {"text": value.records[index]["action"], "prompt_tokens": 100, "completion_tokens": 1}

    class Worker:
        def __init__(self, task, experiment, path, **_kwargs):
            assert task == value.receipt["task_id"] and experiment == value.receipt["experiment_name"]
            self.path = path
            self.unit_name = value.receipt["snapshot"]["unit_name"]

        def request(self, request, **_kwargs):
            if request == {"op": "metadata"}:
                return copy.deepcopy(value.controller[1]["reply"])
            if request["op"] == "dispatch":
                index = calls["dispatched"]
                assert request["action"] == value.records[index]["action"]
                calls["dispatched"] += 1
                return copy.deepcopy(value.worker[2 * index + 1]["reply"])
            assert request["op"] in {"save", "close"}
            return copy.deepcopy(next(row["reply"] for row in value.controller
                                      if row["event"] == request["op"] + "_result"))

        def wait(self, **_kwargs):
            return 0

        def snapshot(self, **_kwargs):
            def write(name, content):
                target = self.path / name
                target.write_bytes(content)
                target.chmod(0o600)

            def journal(rows):
                return b"".join(_encode(row) + b"\n" for row in rows)

            write("actions.jsonl", journal(value.worker))
            write("transport.jsonl", journal(value.transport))
            write("status.json", _encode(value.status) + b"\n")
            write("fault-provenance.json", _encode(value.fault) + b"\n")
            return copy.deepcopy(value.receipt["snapshot"])

        def release(self):
            calls["released"] = True

        def abort(self):
            pytest.fail("the synthetic worker should have closed")

    monkeypatch.setattr(core, "NativeWorkerClient", Worker)

    def run_controller(*args, **kwargs):
        return controller_core.run_task(*args, **kwargs, _inspect=lambda *_a, **_k: {"synthetic": True})

    result = bound.run_episode(config_path, hashlib.sha256(raw).hexdigest(), _model_factory=Model,
                               _run_task=run_controller)
    evidence = _decode((value.run_dir / episode.EVIDENCE_NAME).read_bytes())
    assert result["status"] == "saved_and_closed" and result["actor_outcome"]["stop_reason"] == "max_input_tokens"
    assert evidence["witness"]["status"] == "qualified" and calls == {
        "generated": 4, "dispatched": 4, "released": True}
    assert os.stat(value.run_dir / episode.EVIDENCE_NAME).st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fault", [False, True])
def test_closed_collection_writes_separate_private_receipt_after_reconciliation(binding_fixture, fault):
    f = binding_fixture
    bound = episode.bind(f.reg, f.queue)
    value = synthetic_evidence(fault=fault, private_root=Path(f.reg.PRIVATE_RUN_ROOT))
    value.run_dir.mkdir(mode=0o700)
    (value.run_dir / "worker").mkdir(mode=0o700)

    def write(path, content):
        path.write_bytes(content)
        path.chmod(0o600)

    def journal(rows):
        return b"".join(_encode(row) + b"\n" for row in rows)

    write(value.run_dir / "execution-receipt.json", _encode(value.receipt) + b"\n")
    write(value.run_dir / "controller.jsonl", journal(value.controller))
    write(value.run_dir / "worker/actions.jsonl", journal(value.worker))
    write(value.run_dir / "worker/transport.jsonl", journal(value.transport))
    write(value.run_dir / "worker/status.json", _encode(value.status) + b"\n")
    if value.fault is not None:
        write(value.run_dir / "worker/fault-provenance.json", _encode(value.fault) + b"\n")
    result = episode._collect_closed(bound.core, value.run_dir, value.receipt, value.initial)
    output = value.run_dir / episode.EVIDENCE_NAME
    assert output.stat().st_mode & 0o777 == 0o600 and _decode(output.read_bytes()) == result
    assert result["witness"]["status"] == ("qualified" if fault else "ineligible")
    assert result["fault_provenance_origin"] == (
        "closed_worker_fault_receipt" if fault else "authenticated_absence_after_close")
    with pytest.raises(FileExistsError):
        episode._collect_closed(bound.core, value.run_dir, value.receipt, value.initial)
