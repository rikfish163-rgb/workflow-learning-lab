"""Fresh ten-task cohort after an authenticated synthetic generation passed."""

import hashlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts import probe_native_state_recovery_generation_001 as probe
from scripts import run_native_state_recovery_source_002 as prior
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _same, public_read

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-state-recovery-source-003"
PLAN = "artifacts/qwen38-native-state-recovery-source-plan-003.json"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-source-003"
MODULE = "scripts.run_native_state_recovery_source_003"
SCRIPT = "scripts/run_native_state_recovery_source_003.py"
DESIGN = "artifacts/native-state-recovery-source-design-003.json"
DESIGN_SHA256 = "2b06412687c99543b75255cc21a40d4d2fdfd495f0e08eda5e34b6b7442d4b1b"
CONTEXT = "artifacts/native-state-recovery-source-context-CPU-003.json"
READY = "artifacts/native-state-recovery-generation-probe-terminal-001.json"
READY_SHA256 = "ca0f6bae989f65f0bba824c9389fe2d5482775f241e913715f6b60f3414a3a3f"
PROBE_PLAN_SHA256 = "798d90ae3bc3a70235a73624e7364c88a7f6b6ee5ffc95344d4e70dc48251f15"
PREPARER = "scripts/prepare_native_state_recovery_contexts_001.py"
PREPARER_SHA256 = "980f9626c89884c9f6eef8481548cc3bc208a3a624e50c6cd1a4614b16d55873"
SOURCE002_SHA256 = "44a7eaa6ab4961faa3287a97141890064a21e90a02e0ebf1f4df46da939ec326"
EXPERIMENT_PREFIX = ("native-state-recovery-001-", "native-state-recovery-003-")


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_source_003_binding_invalid")


def episode_binding():
    _require(_public_hash(ROOT, prior.MODULE.replace(".", "/") + ".py") == SOURCE002_SHA256)
    episode = prior._load("native_state_recovery_episode_001", ((*EXPERIMENT_PREFIX, 2),), ID=ID, PLAN=PLAN,
                          DESIGN=DESIGN, DESIGN_SHA256=DESIGN_SHA256)
    _require(episode.TASK_IDS[0] == "22cc237_1")
    episode.TASK_IDS = ("22cc237_3", *episode.TASK_IDS[1:])
    return episode


def prepare_contexts():
    raw = public_read(ROOT, PREPARER)
    _require(hashlib.sha256(raw).hexdigest() == PREPARER_SHA256)
    text = raw.decode("utf-8")
    predicate = 'task.endswith("_1") and task in roles["origin"]'
    _require(text.count(predicate) == 1)
    module = ModuleType("scripts._native_state_recovery_contexts_003")
    module.__file__, module.__package__ = str(ROOT / PREPARER), "scripts"
    exec(compile(text.replace(predicate, 'task in roles["origin"]'), module.__file__, "exec", dont_inherit=True),
         vars(module))
    module.episode = episode_binding()
    module.ID, module.PROOF, module.PREPARER = Path(CONTEXT).stem, ROOT / CONTEXT, SCRIPT
    module.SOURCES = module.SOURCES | {SCRIPT}
    return module.main()


@cache
def bindings():
    reg, queue, _, _ = prior.bindings.__wrapped__()
    episode = episode_binding()
    reg.ID, reg.PLAN, reg.PRIVATE_RUN_ROOT = ID, PLAN, PRIVATE_RUN_ROOT
    reg.episode, reg.TASK_IDS, reg.CONTEXT = episode, episode.TASK_IDS, CONTEXT
    reg.BATCH_TASK_IDS = tuple((task,) for task in reg.TASK_IDS)
    reg.FIXED_INPUTS = reg.FIXED_INPUTS | {DESIGN: DESIGN_SHA256, CONTEXT: _public_hash(ROOT, CONTEXT),
        probe.FAILURE: probe.FAILURE_SHA256, prior.PLAN: probe.SOURCE_SHA256, READY: READY_SHA256,
        probe.PLAN: PROBE_PLAN_SHA256}
    reg.NEW_SOURCES = reg.NEW_SOURCES | {SCRIPT, probe.SOURCE_FILE}
    original_build = reg.build

    def build(root):
        plan = original_build(root)
        failure = reg.public._read(root, probe.FAILURE)
        readiness = reg.public._read(root, READY)
        _require(failure["source_plan_sha256"] == probe.SOURCE_SHA256 and failure["source_tasks_started"] == 1
                 and failure["model_generation_returns"] == failure["tool_actions"] == failure["official_scores"] == 0
                 and failure["release_verified"] is True and readiness["status"] == "passed"
                 and readiness["probe_plan_sha256"] == PROBE_PLAN_SHA256 and readiness["model_generation_returns"] == 1
                 and readiness["source_tasks_started"] == 0 and readiness["release_verified"] is True
                 and _same(readiness["compiled_header_binding"], probe.header_binding()))
        plan["source_002_startup_terminal"] = {"path": probe.FAILURE, "sha256": probe.FAILURE_SHA256}
        plan["readiness_probe"] = {"path": READY, "sha256": READY_SHA256}
        return plan

    reg.build = build
    queue.ID, queue.PLAN = ID, PLAN
    queue.GUARD_NAME = "native-state-recovery-source-operations-003.lock"
    make_job = queue.make_job

    def job(index, owner, sha):
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    runner = prior._load("run_native_state_recovery_source_001", (
        (*EXPERIMENT_PREFIX, 1), ("scripts.run_native_state_recovery_source_001", MODULE, 2),
    ), ID=ID, registration=reg, queue=queue, episode=episode)
    return reg, queue, episode, runner


def main():
    if sys.argv[1:] == ["--prepare-contexts"]:
        prepare_contexts()
        return 0
    _reg, queue, _episode, runner = bindings()
    return queue.main() if "--serve" in sys.argv[1:] else runner.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_source_003_failed\n")
        raise SystemExit(2) from None
