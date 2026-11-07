"""Exercise the future arm entry with fabricated lease and backend metadata."""

import copy
import os
import time
from types import SimpleNamespace

import pytest

from scripts import native_source_episode as native
from scripts import native_state_recovery_authored_episode_001 as episode
from scripts.native_appworld_worker_client import _encode

PLAN_SHA = "a" * 64
CONFIG_SHA = "b" * 64
OWNER = "d" * 32
TASK = "0000001_1"


class Queue:
    ID = episode.ID
    REMOTE = episode.DEPLOYMENT

    def make_job(self, batch, owner, sha):
        return {"id": f"{self.ID}-comparison-{batch}", "kind": "comparison",
                "batch_index": batch, "owner": owner, "plan_sha256": sha,
                "max_runtime_seconds": 7200}

    def queue_operation(self, *_args, **_kwargs):
        pytest.fail("The test must never access a real GPU queue.")


class Reader:
    def __init__(self, *, changed_at=None, age=30):
        self.calls = 0
        self.changed_at = changed_at
        self.age = age

    def _backend(self, prior, backend):
        native._backend(prior, backend)

    def _queue_binding(self, config, _status):
        self.calls += 1
        identity = "replacement" if self.changed_at is not None and self.calls >= self.changed_at else "owner"
        return {"job": {"started_epoch": time.time() - self.age}}, {}, (identity,)


class Backend:
    max_tokens = 1024
    deadline = 0

    def __init__(self, calls):
        self.calls = calls

    def check_health(self):
        self.calls.append("health")

    def count_input_tokens(self, _messages):
        self.calls.append("count")
        return 1

    def generate(self, _messages, *, seed):
        self.calls.append(("generate", seed))


@pytest.fixture
def prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(episode, "WORKSPACE", tmp_path)
    queue = Queue()
    draft = {
        "split": "authored", "scorer": "authored_terminal_state", "test_set_use_if_run": False,
        "expected_pairs": [[TASK, "0000001", 17]], "batches": [[0]],
        "arm_orders": [["control", "experience"]], "gpu_seconds_per_batch_ceiling": 7200,
        "initial_state_manifest_sha256": "c" * 64, "model_id": "scripted-backend", "model_arm": "B3",
        "source_sha256": {"scripts/synthetic.py": "f" * 64},
        "protocol": {"max_new_tokens": 1024, "context_tokens": 16384, "temperature": .7,
                     "request_window_seconds": 120, "worker_wall_seconds": 780,
                     "rpc_timeout_default_seconds": 60, "setup_seconds": 120,
                     "cleanup_seconds": 60, "shutdown_reserve_seconds": 150,
                     "episode_unit_seconds": 900, "scorer_seconds": 120},
    }
    plan = {"id": episode.ID, "status": "registered", "utility_authorized": True,
            "dev_or_test_authorized": False, "automatic_retry": False, "asset_root": str(tmp_path),
            "draft": draft, "episode_source_sha256": "f" * 64}
    report = _encode({"options": {"authored": {"draft_sha256": episode._hash(_encode(draft))}}})
    monkeypatch.setattr(episode, "public_read", lambda *_: report)
    monkeypatch.setattr(episode, "DRAFT_SUMMARY_SHA256", episode._hash(report))
    monkeypatch.setattr(episode, "_public_hash", lambda *_: "f" * 64)
    static = {"model_id": draft["model_id"]}
    devices = {"8": "GPU-ecc8cd45-fdaa-b5ee-e54a-bf2ef7df8f82"}
    prior = {"port": 18991, "model_arm": "B3", "model_id": draft["model_id"],
             "expected_backend_static": {"B3": static},
             "gpu_policy": {"hostname": "amax", "allowed_gpus": devices}, "gpu_policy_sha256": "e" * 64}
    monkeypatch.setattr(native.registration, "MODEL_ARM", "B3")
    backend = {**static, "loading_peak_reserved_bytes": 1,
               "gpu_policy": {"hostname": "amax", "physical_gpu_index": 8, "gpu_uuid": devices["8"],
                              "visible_device_count": 1, "policy_sha256": prior["gpu_policy_sha256"]}}
    configs = {}
    for arm in ("control", "experience"):
        configs[arm] = {"plan_sha256": PLAN_SHA, "pair_index": 0, "arm": arm,
                        "job": queue.make_job(0, OWNER, PLAN_SHA), "expected_backend": copy.deepcopy(backend)}
    monkeypatch.setattr(episode, "_prior", lambda: prior)
    monkeypatch.setattr(episode, "_bundle", lambda *_: "fixed reviewed text")
    monkeypatch.setattr(episode.scorer, "_main_pid", lambda _: os.getpid())
    monkeypatch.setattr(episode.assets, "verify_assets", lambda root, _: root)
    monkeypatch.setattr(episode.assets, "worker_factory", lambda *_: "trusted factory")
    monkeypatch.setattr(episode.utility, "_limits", lambda _: "fixed limits")
    monkeypatch.setattr(episode.utility, "_create_episode_directory", lambda *_: None)
    calls = []

    def read(path, sha):
        if path.name == episode.REGISTRATION:
            assert sha == PLAN_SHA
            return plan
        assert sha == CONFIG_SHA
        return next(config for arm, config in configs.items() if f"-{arm}-" in path.name)

    def model(endpoint, model_id, **_kwargs):
        assert endpoint == "http://127.0.0.1:18991/v1" and model_id == draft["model_id"]
        calls.append("construct")
        return Backend(calls)

    def controller(task, experiment, path, *, model, token_counter, review_hint, **kwargs):
        assert task == TASK and path.name == experiment and kwargs["is_policy_episode"] is True
        calls.append(("hint", review_hint))
        model.check_health()
        token_counter([])
        model.generate([], seed=17)
        return {"task_id": task, "experiment_name": experiment}

    monkeypatch.setattr(episode, "_read", read)

    def run(arm, reader):
        path = tmp_path / "configs" / f"pair-00-{arm}-{OWNER}.json"
        return episode.run_episode(tmp_path, path, CONFIG_SHA, PLAN_SHA, private_plan_sha=PLAN_SHA,
                                   queue=queue, _reader=reader,
                                   _model_factory=model, _controller=controller)

    return SimpleNamespace(plan=plan, configs=configs, calls=calls, run=run)


@pytest.mark.parametrize(("arm", "hint"), [("control", None), ("experience", "fixed reviewed text")])
def test_binding_reuses_one_lease_and_only_treatment_gets_hint(prepare, arm, hint):
    reader = Reader()
    assert prepare.run(arm, reader)["task_id"] == TASK
    assert prepare.calls == ["construct", ("hint", hint), "health", "count", ("generate", 17)]
    assert reader.calls == 6


@pytest.mark.parametrize("field", ["status", "utility_authorized", "dev_or_test_authorized"])
def test_unregistered_or_wrong_scope_cannot_construct_model(prepare, field):
    prepare.plan[field] = {"status": "draft", "utility_authorized": False, "dev_or_test_authorized": True}[field]
    reader = Reader()
    with pytest.raises(ValueError):
        prepare.run("control", reader)
    assert prepare.calls == [] and reader.calls == 0


@pytest.mark.parametrize("bad_binding", ["job", "backend"])
def test_mismatched_job_or_backend_is_rejected_before_any_lease_query(prepare, bad_binding):
    config = prepare.configs["control"]
    if bad_binding == "job":
        config["job"]["batch_index"] = 5
    else:
        config["expected_backend"]["gpu_policy"]["physical_gpu_index"] = 7
    reader = Reader()
    with pytest.raises((ValueError, native.NativeSourceEpisodeError)):
        prepare.run("control", reader)
    assert prepare.calls == [] and reader.calls == 0


def test_replacement_during_tool_loop_prevents_next_generation(prepare):
    with pytest.raises(ValueError):
        prepare.run("experience", Reader(changed_at=5))
    assert prepare.calls == ["construct", ("hint", "fixed reviewed text"), "health", "count"]


@pytest.mark.parametrize("age", [6050, 7051])
def test_near_expiry_lease_is_rejected_before_model_construction(prepare, age):
    with pytest.raises(ValueError):
        prepare.run("control", Reader(age=age))
    assert prepare.calls == []


def test_changed_episode_source_is_rejected_before_backend_binding(prepare):
    prepare.plan["episode_source_sha256"] = "1" * 64
    reader = Reader()
    with pytest.raises(ValueError):
        prepare.run("control", reader)
    assert prepare.calls == [] and reader.calls == 0
