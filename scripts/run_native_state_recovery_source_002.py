"""New launch namespace after source001 stopped before any task or generation.

Compare the backend against its registered JSON wire representation. The
cohort, experimental budgets, immutable predecessor, and GPU fence remain
bound. No import or help operation enqueues or loads a model.
"""

import hashlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts.native_appworld_worker_client import _decode, _encode
from scripts.native_source_registration import PRIVATE_STATE_ROOT, public_read

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-state-recovery-source-002"
PLAN = "artifacts/qwen38-native-state-recovery-source-plan-002.json"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-source-002"
PRIOR_PLAN = "artifacts/qwen38-native-state-recovery-source-plan-001.json"
PRIOR_TERMINAL = "artifacts/native-state-recovery-source-startup-terminal-001.json"
PRIOR_INPUTS = {
    PRIOR_PLAN: "ec65dab0d09d99e0f612e34760afd2c00237e0e4ee5275c4b2c67d2b0ef4ba0f",
    PRIOR_TERMINAL: "3c5fb191995dd81ad6885941d2111f2a5e8afdbac5a2c95a86d20574db93a396",
}
MODULE = "scripts.run_native_state_recovery_source_002"
SOURCES = {
    "native_state_recovery_episode_001": "fade63fc384e41e2de4038b2a8ab46e5bf690c670f12f61dfeca71c76bb067ca",
    "native_state_recovery_registration_001": "2a044b671d2b6fa608cf437986f8bab54964af7667abf6805990e7f1d5402c8f",
    "native_state_recovery_gpu_queue_001": "334daa20a24621c1a53eab895ab232671649c9797c9da18350a22b507da50453",
    "run_native_state_recovery_source_001": "aaffda8d9a4b09c4c438ff756eea3e67fa902c58b79256e016263942cf983222",
}


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_source_002_binding_invalid")


def _load(name, edits=(), **bindings):
    path = f"scripts/{name}.py"
    raw = public_read(ROOT, path)
    _require(hashlib.sha256(raw).hexdigest() == SOURCES[name])
    text = raw.decode("utf-8")
    for before, after, count in edits:
        _require(text.count(before) == count)
        text = text.replace(before, after)
    module = ModuleType("scripts._native_state_recovery_002_" + name)
    module.__file__, module.__package__ = str(ROOT / path), "scripts"
    exec(compile(text, module.__file__, "exec", dont_inherit=True), vars(module))
    vars(module).update(bindings)
    return module


def wire_backend(backend):
    """Normalize exactly as HTTP JSON does before the strict backend comparison."""
    return _decode(_encode(backend))


@cache
def bindings():
    experiment = ("native-state-recovery-001-", "native-state-recovery-002-")
    episode = _load("native_state_recovery_episode_001", ((*experiment, 2),), ID=ID, PLAN=PLAN)
    reg = _load("native_state_recovery_registration_001", episode=episode, ID=ID, PLAN=PLAN,
                PRIVATE_RUN_ROOT=PRIVATE_RUN_ROOT)
    reg.FIXED_INPUTS = reg.FIXED_INPUTS | PRIOR_INPUTS
    reg.NEW_SOURCES = reg.NEW_SOURCES | {f"scripts/{name}.py" for name in SOURCES} | {
        "scripts/run_native_state_recovery_source_002.py", "tests/test_native_state_recovery_source_002.py",
    }
    original_build = reg.build

    def build(root):
        plan = original_build(root)
        old = reg.public._read(root, PRIOR_TERMINAL)
        _require(old["status"] == "released_backend_metadata_type_mismatch_before_actor"
                 and old["source_plan_sha256"] == PRIOR_INPUTS[PRIOR_PLAN] and old["release_verified"] is True
                 and old["automatic_retry"] is False and all(type(old[key]) is int and old[key] == 0
                     for key in ("source_tasks_started", "model_generations", "official_scores", "writer_calls")))
        plan["startup_predecessor"] = {"path": PRIOR_TERMINAL, "sha256": PRIOR_INPUTS[PRIOR_TERMINAL]}
        return plan

    reg.build = build
    queue = _load("native_state_recovery_gpu_queue_001", (
        ("scripts.native_state_recovery_gpu_queue_001", MODULE, 1),
        ("registration.static_backend(engine.backend)", "registration.static_backend(wire_backend(engine.backend))", 1),
    ), ID=ID, PLAN=PLAN, GUARD_NAME="native-state-recovery-source-operations-002.lock",
        registration=reg, wire_backend=wire_backend)
    runner = _load("run_native_state_recovery_source_001", (
        (*experiment, 1), ("scripts.run_native_state_recovery_source_001", MODULE, 2),
    ), ID=ID, registration=reg, queue=queue, episode=episode)
    return reg, queue, episode, runner


def main():
    _reg, queue, _episode, runner = bindings()
    return queue.main() if "--serve" in sys.argv[1:] else runner.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_source_002_failed\n")
        raise SystemExit(2) from None
