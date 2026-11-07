"""The full V27 comparison for an independently replayed state-recovery lesson."""

import hashlib
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts import native_source_registration_006 as roster
from scripts import native_state_recovery_review_001 as review
from scripts import run_native_state_recovery_writer_001 as writer
from scripts.native_appworld_worker_client import _encode, _same
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _public_hash, _sha, public_read

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-state-recovery-utility-001"
PLAN = "artifacts/qwen38-native-state-recovery-utility-plan-001.json"
MODULE = "scripts.run_native_state_recovery_utility_001"
SCRIPT = "scripts/run_native_state_recovery_utility_001.py"
QUALIFICATION_ID = "native-state-recovery-utility-qualification-001"
QUALIFICATION_PROOF = f"artifacts/{QUALIFICATION_ID}.json"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-utility-001"
REGISTRATION = "scripts/native_experience_utility_registration_001.py"
RUNNER = "scripts/run_native_experience_utility_001.py"
PINS = {
    REGISTRATION: "f6ec557278870026ebccf8dffdcbc2db7941228a0fcd67b21b23eb3ffa2667d9",
    RUNNER: "aa5e8184098900f7575c81e3ab1b39218f4e9b8b5db0df62588789c87aa02876",
}
ROSTER_SHA256 = "da4b4c4a4c8040a961ef0604ed19152bc5a3ea21790618afd19e694651c63ad3"
SOURCE_SHA256 = writer.evidence.SOURCE_PLAN_SHA256


@dataclass(frozen=True)
class UtilityLineage:
    slug: str
    module: str
    review: ModuleType
    source_scenarios: int
    fixed_source_sha256: str | None = None

    @property
    def source(self):
        return self.review.writer.source

    @property
    def writer(self):
        return self.review.writer

    def source_sha256(self, root):
        return self.fixed_source_sha256 or _public_hash(root, self.source.PLAN)


NATURAL = UtilityLineage("native-state-recovery-utility-001", MODULE, review, 10, SOURCE_SHA256)


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_utility_binding_invalid")


def _load(path, edits):
    raw = public_read(ROOT, path)
    _require(hashlib.sha256(raw).hexdigest() == PINS[path])
    text = raw.decode("utf-8")
    for before, after, count in edits:
        _require(text.count(before) == count)
        text = text.replace(before, after)
    module = ModuleType("scripts._native_state_recovery_utility_" + Path(path).stem)
    module.__file__, module.__package__ = str(ROOT / path), "scripts"
    exec(compile(text, module.__file__, "exec", dont_inherit=True), vars(module))
    return module


def _source_plan(root, lineage=NATURAL):
    return lineage.source.bindings()[0].load_registration(root, lineage.source_sha256(root), private=False)


def _writer_plan(root, sha, lineage=NATURAL):
    return lineage.writer.bindings()[0].load_registration(root, sha, private=False)


def _cohort(previous, lineage=NATURAL):
    frozen = roster.load_registration(ROOT, ROSTER_SHA256, private=False)["design"]["cohort"]
    tasks = frozen["utility_task_ids_reserved"]
    scenes = list(dict.fromkeys(task[:-2] for task in tasks))
    origins = sorted(task[:-2] for task in previous["task_ids"])
    _require(frozen["official_split"] == "train" and frozen["project_unseen"] is False
             and frozen["all_train_roles_already_exposed"] is True
             and len(origins) == len(set(origins)) == lineage.source_scenarios and len(scenes) == 9
             and not set(origins) & set(scenes)
             and (lineage.fixed_source_sha256 is None or scenes == previous["design"]["utility_scenarios"])
             and tasks == [f"{scene}_{variant}" for scene in scenes for variant in (1, 2, 3)])
    pairs = [[task, task[:-2], 17] for task in tasks]
    batches = [list(range(i, min(i + 2, 27))) for i in range(0, 27, 2)]
    orders = [["control", "experience"] if i % 2 == 0 else ["experience", "control"] for i in range(27)]
    return origins, pairs, batches, orders


def _validate_writer(candidate, previous, proof, lineage=NATURAL):
    bound = candidate["qualification"]
    _require(candidate["id"] == lineage.writer.ID and candidate["status"] == "registered"
             and candidate["scope"] == "writer_and_conditional_review_no_task" and candidate["writer_authorized"] is True
             and candidate["writer_max_calls"] == candidate["review_max_calls"] == 1
             and candidate["automatic_retry"] is False
             and all(candidate[key] is False for key in ("training_authorized", "utility_authorized", "dev_or_test_authorized"))
             and bound["source_plan_sha256"] == lineage.source_sha256(ROOT)
             and bound["source_result_sha256"] == proof["source_result_sha256"]
             and all(_same(candidate[k], previous[k]) for k in ("expected_backend_static", "model_id", "model_arm", "port",
                     "service_plan", "service_plan_sha256", "runtime_relocation", "gpu_policy", "gpu_policy_sha256", "python", "tmp_dir")))


@cache
def bindings(lineage=NATURAL):
    base, queue, _episode, _runner = lineage.source.bindings.__wrapped__()
    study_id = f"qwen38-{lineage.slug}"
    plan_path = f"artifacts/qwen38-{lineage.slug.removesuffix('-001')}-plan-001.json"
    qualification_id = f"{lineage.slug.removesuffix('-001')}-qualification-001"
    review = lineage.review
    writer = lineage.writer
    source = lineage.source
    registration = _load(REGISTRATION, (
        ("native-experience-001:", review.BUNDLE_PREFIX, 2),
        ('"python", "gpu_policy",', '"python", "runtime_relocation", "gpu_policy",', 1),
        ('and plan["python"] == PYTHON and plan["tmp_dir"] == TMPDIR',
         'and public._same(plan["runtime_relocation"], previous["runtime_relocation"])\n'
         '                 and plan["python"] == PYTHON and plan["tmp_dir"] == TMPDIR', 1),
        ('"runtime_sha256": copy.deepcopy(runtime_sha256), "tmp_dir": TMPDIR,',
         '"runtime_sha256": copy.deepcopy(runtime_sha256), "tmp_dir": TMPDIR,\n'
         '                "runtime_relocation": copy.deepcopy(previous["runtime_relocation"]),', 1),
    ))
    registration.ID, registration.PLAN, registration.PORT = study_id, plan_path, base.PORT
    registration.PRIVATE_RUN_ROOT = f"{PRIVATE_STATE_ROOT}/{lineage.slug}"
    registration.QUALIFICATION_ID = qualification_id
    registration.QUALIFICATION_PROOF = f"artifacts/{qualification_id}.json"
    registration.SOURCE_PLAN, registration.SOURCE_PLAN_SHA, registration.SOURCE_GATE_KIND = (
        source.PLAN, lineage.source_sha256(ROOT), review.SOURCE_GATE_KIND)
    registration.WRITER_ID, registration.WRITER_PLAN = writer.ID, writer.PLAN
    registration.SOURCE_RESULT_PATH = str(review.SOURCE_RESULT)
    registration.WRITER_RESULT_PATH = str(Path(writer.PRIVATE_RUN_ROOT) / "study-result.json")
    registration.REVIEW_MANIFEST_PATH = str(Path(writer.PRIVATE_RUN_ROOT) / review.MANIFEST)
    registration.REQUIRED_NEW_SOURCES |= {SCRIPT, "scripts/native_state_recovery_review_001.py",
        "scripts/report_native_state_recovery_utility_001.py", "scripts/report_paired_omission_utility_001.py",
        lineage.module.replace(".", "/") + ".py"}
    registration._source_plan = lambda root: _source_plan(root, lineage)
    registration._writer_plan = lambda root, sha: _writer_plan(root, sha, lineage)
    registration._cohort = lambda previous: _cohort(previous, lineage)
    registration._validate_previous = lambda candidate, previous, proof: _validate_writer(candidate, previous, proof, lineage)
    registration._load_reviewed_bundle = lambda _root, *hashes: review.load_reviewed_bundle(*hashes)

    queue.ID, queue.PLAN, queue.registration = study_id, plan_path, registration
    queue.GUARD_NAME = f"{lineage.slug.removesuffix('-001')}-operations-001.lock"

    def job(index, owner, sha):
        _require(type(index) is int and 0 <= index < 14 and type(owner) is str
                 and re.fullmatch(r"[a-f0-9]{32}", owner) and _sha(sha))
        return {"id": f"{study_id}-comparison-{index}", "kind": "comparison", "batch_index": index,
                "owner": owner, "plan_sha256": sha, "max_runtime_seconds": 7200,
                "argv": [base.PYTHON, "-B", "-m", lineage.module, "--serve", "--plan-sha256", sha,
                         "--batch", str(index), "--owner", owner]}

    queue.make_job = job
    runner = _load(RUNNER, (
        ("qwen38-native-experience-utility-001", study_id, 1),
        ("artifacts/qwen38-native-experience-utility-plan-001.json", plan_path, 1),
        ("scripts.run_native_experience_utility_001", lineage.module, 2),
        ("native-experience-utility-001-", lineage.slug + "-", 1),
        ("ordinary_control_recovery_gate", review.SOURCE_GATE_KIND, 1),
        ("native-experience-001:", review.BUNDLE_PREFIX, 1),
    ))
    runner._registry = lambda: registration
    runner._queue = lambda: queue
    return registration, queue, runner


def qualification(manifest, manifest_sha, *, lineage=NATURAL):
    """Create the public claim only from the already authenticated private manifest."""
    _require(_sha(manifest_sha) and manifest["id"] == lineage.review.LINEAGE.manifest_id
             and manifest["private"] is True and manifest["source_plan_sha256"] == lineage.source_sha256(ROOT)
             and manifest["source_gate_kind"] == lineage.review.SOURCE_GATE_KIND)
    fields = ("source_gate_kind", "source_plan_sha256", "source_result_sha256", "writer_plan_sha256",
              "writer_result_sha256", "experience_bundle_id", "source_scenario_ids")
    qualification_id = f"{lineage.slug.removesuffix('-001')}-qualification-001"
    return {"id": qualification_id, "status": "plausible_pending_utility", "review_manifest_sha256": manifest_sha,
            **{key: manifest[key] for key in fields}}


def build_registration(root, proof_sha, *, private=False, lineage=NATURAL):
    registration, _queue, _runner = bindings(lineage)
    proof_path = registration.QUALIFICATION_PROOF
    proof = registration.public._read(root, proof_path)
    prior = _writer_plan(root, proof["writer_plan_sha256"], lineage)
    sources = prior["source_sha256"] | {name: _public_hash(root, name) for name in registration.REQUIRED_NEW_SOURCES}
    _require(all(sources.get(name) == digest for name, digest in PINS.items()))
    inputs = prior["input_sha256"] | {proof_path: proof_sha, lineage.writer.PLAN: proof["writer_plan_sha256"],
                                     roster.PLAN: ROSTER_SHA256}
    return registration.build_registration(root, source_sha256=sources, input_sha256=inputs,
        runtime_sha256=prior["runtime_sha256"], qualification_proof_sha256=proof_sha, private=private)


def _verification_payload(spec):
    return bindings()[2]._verification_payload(spec)


def main(*, lineage=NATURAL):
    _reg, queue, runner = bindings(lineage)
    return queue.main() if "--serve" in sys.argv[1:] else runner.main()


__all__ = ["_encode"]


if __name__ == "__main__":
    raise SystemExit(main())
