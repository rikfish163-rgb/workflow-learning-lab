"""Register one reviewed paired-omission lesson against the fixed train V27.

Build and import are inert. Public qualification is metadata; the independent
private review loader must authenticate the writer, review, and their source
evidence before admission. No test or previously observed outcome is used to
select the cohort or stop the comparison. The legacy source_result_sha256 field
binds the writer's canonical combined evidence input, not a raw study result.
"""

import hashlib
import importlib.util
import os
import shutil
import socket
from pathlib import Path

from scripts import native_experience_utility_registration_001 as old
from scripts import native_source_registration as public
from scripts import native_source_registration_006 as source6

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = "scripts/native_experience_utility_registration_001.py"
BASE_SHA256 = "f6ec557278870026ebccf8dffdcbc2db7941228a0fcd67b21b23eb3ffa2667d9"
BASE_TEST = "tests/test_native_experience_utility_registration_001.py"
BASE_TEST_SHA256 = "392ccc1b7143475508f1138b9daca4fadac493221256c58f65d35880a1af3229"
SEALER_BASE = "scripts/prepare_codeact_within_utility_001.py"
SEALER_BASE_SHA256 = "fc3bdb15ad7e72453e2c03493952f5f4175c25c62e8667c3f5c18cc38e674027"
ID = "qwen38-paired-omission-utility-001"
PLAN = f"artifacts/{ID}-plan.json"
QUALIFICATION_ID = "paired-omission-utility-qualification-001"
QUALIFICATION_PROOF = "artifacts/paired-omission-utility-qualification-001.json"
SOURCE_PLAN = "artifacts/paired-omission-evidence-plan-001.json"
SOURCE_PLAN_SHA = "ae10ca458161d62c14e97623623418942d1448161f53845be0fffe60395405f5"
ROSTER_PLAN = "artifacts/qwen38-native-source-pilot-plan-006.json"
ROSTER_PLAN_SHA = "da4b4c4a4c8040a961ef0604ed19152bc5a3ea21790618afd19e694651c63ad3"
SOURCE_GATE_KIND = "paired_verified_write_omission_v1"
WRITER_ID = "qwen38-paired-omission-writer-003"
WRITER_PLAN = f"artifacts/{WRITER_ID}-plan.json"
BUNDLE_PREFIX = "paired-omission-experience-001:"
SOURCE_SCENARIOS = ("2a163ab", "b7a9ee9")
UTILITY_SCENARIOS = ("229360a", "34d9492", "6104387", "76f2c72", "7d7fbf6",
                     "aa8502b", "afc0fce", "ce359b5", "e7a10f8")
UTILITY_TASK_IDS = tuple(f"{scene}_{variant}" for scene in UTILITY_SCENARIOS for variant in (1, 2, 3))
PRIVATE_STATE_ROOT = old.PRIVATE_STATE_ROOT
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/paired-omission-utility-001"
WRITER_RESULT_PATH = PRIVATE_STATE_ROOT + "/paired-omission-writer-003/study-result.json"
REVIEW_MANIFEST_PATH = PRIVATE_STATE_ROOT + "/paired-omission-writer-003/review-manifest.json"
PORT = 18997
PYTHON, TMPDIR = old.PYTHON, old.TMPDIR
SERVICE_PLAN, SERVICE_SHA = old.SERVICE_PLAN, old.SERVICE_SHA
MODEL_ARM, MODEL_ID = old.MODEL_ARM, old.MODEL_ID
MINIMUM_FREE_BYTES, WAIT_SECONDS = old.MINIMUM_FREE_BYTES, old.WAIT_SECONDS
BATCH_GPU_SECONDS, N_BATCHES, TOTAL_GPU_SECONDS = old.BATCH_GPU_SECONDS, old.N_BATCHES, old.TOTAL_GPU_SECONDS
REQUIRED_NEW_SOURCES = old.REQUIRED_NEW_SOURCES | frozenset({
    BASE_SOURCE, BASE_TEST, SEALER_BASE, "scripts/native_experience_registration_001.py",
    "scripts/paired_omission_utility_registration_001.py",
    "scripts/paired_omission_utility_gpu_queue_001.py",
    "scripts/run_paired_omission_utility_001.py",
    "scripts/prepare_paired_omission_utility_001.py",
    "tests/test_paired_omission_utility_001.py",
    "tests/test_prepare_paired_omission_utility_001.py",
})
public_read = public.public_read
_public_hash = public._public_hash
_absolute_hash = public._absolute_hash
_private_directory = public._private_directory
_ERROR = "paired_omission_utility_registration_invalid"
_CORE = None


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_plan(root):
    """Use the authenticated native plan solely for its frozen train roster."""
    _require(_hash(public_read(root, ROSTER_PLAN)) == ROSTER_PLAN_SHA)
    return source6.load_registration(root, ROSTER_PLAN_SHA, private=False)


def _writer_plan(root, sha):
    from scripts import run_paired_omission_writer_003 as writer

    _require(writer.ID == WRITER_ID and writer.PLAN == WRITER_PLAN
             and public._sha(sha) and _hash(public_read(root, WRITER_PLAN)) == sha)
    return writer.load_registration(root, sha, private=False)


def _cohort(previous):
    cohort = previous["design"]["cohort"]
    _require(type(cohort) is dict and cohort.get("utility_scenarios") == 9
             and cohort.get("official_split") == "train" and cohort.get("project_unseen") is False
             and cohort.get("all_train_roles_already_exposed") is True
             and public._same(cohort.get("utility_task_ids_reserved"), list(UTILITY_TASK_IDS))
             and not set(SOURCE_SCENARIOS) & set(UTILITY_SCENARIOS))
    pairs = [[task, task[:-2], 17] for task in UTILITY_TASK_IDS]
    batches = [list(range(i, min(i + 2, len(pairs)))) for i in range(0, len(pairs), 2)]
    orders = [["control", "experience"] if i % 2 == 0 else ["experience", "control"]
              for i in range(len(pairs))]
    return list(SOURCE_SCENARIOS), pairs, batches, orders


def _validate_previous(writer, previous, proof):
    _require(type(writer) is dict and writer.get("id") == WRITER_ID and writer.get("status") == "registered"
             and writer.get("writer_authorized") is True
             and all(writer.get(k) is False for k in ("training_authorized", "utility_authorized", "dev_or_test_authorized"))
             and writer.get("model_arm") == MODEL_ARM and writer.get("model_id") == MODEL_ID
             and writer.get("port") == 18996 and writer.get("service_plan") == SERVICE_PLAN
             and writer.get("service_plan_sha256") == SERVICE_SHA and writer.get("python") == PYTHON
             and writer.get("tmp_dir") == TMPDIR
             and writer.get("source") == {"plan_sha256": SOURCE_PLAN_SHA,
                                           "result_sha256": proof["source_result_sha256"],
                                           "source_gate_kind": SOURCE_GATE_KIND}
             and writer.get("private_input_sha256") == proof["source_result_sha256"]
             and writer.get("registered_inputs", {}).get("origin_ids") == ["b7a9ee9", "2a163ab"]
             and writer.get("input_sha256", {}).get(SOURCE_PLAN) == SOURCE_PLAN_SHA
             and writer.get("input_sha256", {}).get(ROSTER_PLAN) == ROSTER_PLAN_SHA
             and public._same(writer.get("expected_backend_static"), previous["expected_backend_static"]))


def _private_assets():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and os.environ.get("TMPDIR") == TMPDIR)
    for path in (PRIVATE_STATE_ROOT, TMPDIR, PRIVATE_RUN_ROOT):
        _private_directory(path)
    _require(shutil.disk_usage(TMPDIR).free >= MINIMUM_FREE_BYTES
             and shutil.disk_usage(PRIVATE_RUN_ROOT).free >= MINIMUM_FREE_BYTES)


def _load_reviewed_bundle(root, writer_plan_sha, writer_result_sha, review_manifest_sha):
    from scripts.prepare_paired_omission_utility_001 import load_reviewed_bundle

    return load_reviewed_bundle(root, writer_plan_sha, writer_result_sha, review_manifest_sha)


def _private_qualification(root, plan, proof):
    """Recompute source and writer authentication; public claims alone are insufficient."""
    _private_assets()
    _require(all(_absolute_hash(name) == digest for name, digest in plan["runtime_sha256"].items()))
    reviewed = _load_reviewed_bundle(root, proof["writer_plan_sha256"], proof["writer_result_sha256"],
                                     proof["review_manifest_sha256"])
    _require(type(reviewed) is dict and set(reviewed) == old._PRIVATE_FIELDS and reviewed["private"] is True
             and reviewed["status"] == "plausible_pending_utility"
             and all(public._same(reviewed[k], proof[k]) for k in (
                 "source_gate_kind", "source_plan_sha256", "source_result_sha256", "writer_plan_sha256",
                 "writer_result_sha256", "review_manifest_sha256", "experience_bundle_id", "source_scenario_ids")))
    text = reviewed["insertion_text"]
    _require(type(text) is str and bool(text.strip()) and 1 <= len(text.encode("utf-8")) <= 65536
             and reviewed["experience_bundle_id"] == BUNDLE_PREFIX + _hash(text.encode("utf-8")))
    files = reviewed["private_files_sha256"]
    _require(type(files) is dict and bool(files) and files.get(WRITER_RESULT_PATH) == proof["writer_result_sha256"]
             and files.get(REVIEW_MANIFEST_PATH) == proof["review_manifest_sha256"])
    for name, digest in files.items():
        _require(type(name) is str and Path(name).is_absolute() and os.path.normpath(name) == name
                 and Path(PRIVATE_STATE_ROOT) in Path(name).parents and ".." not in Path(name).parts
                 and public._sha(digest))
    return reviewed


def _isolated(*, raw=None):
    raw = public.public_read(ROOT, BASE_SOURCE) if raw is None else raw
    _require(type(raw) is bytes and _hash(raw) == BASE_SHA256
             and _hash(public.public_read(ROOT, BASE_TEST)) == BASE_TEST_SHA256)
    text = raw.decode("utf-8")
    _require(text.count("native-experience-001:") == 2)
    text = text.replace("native-experience-001:", BUNDLE_PREFIX)
    spec = importlib.util.spec_from_file_location("scripts._paired_omission_utility_registration_base", ROOT / BASE_SOURCE)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    exec(compile(text, str(ROOT / BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    for name in ("ID", "PLAN", "QUALIFICATION_ID", "QUALIFICATION_PROOF", "SOURCE_PLAN", "SOURCE_PLAN_SHA",
                 "SOURCE_GATE_KIND", "WRITER_ID", "WRITER_PLAN", "PRIVATE_RUN_ROOT", "REQUIRED_NEW_SOURCES",
                 "PORT", "_ERROR"):
        setattr(core, name, globals()[name])
    for name in ("public_read", "_public_hash", "_absolute_hash", "_private_directory", "_source_plan",
                 "_writer_plan", "_cohort", "_validate_previous", "_private_assets", "_load_reviewed_bundle",
                 "_private_qualification"):
        setattr(core, name, lambda *args, _name=name, **kwargs: globals()[_name](*args, **kwargs))
    validate = core.validate_registration

    def validate_bound(root, plan, *, private=False):
        _require(type(plan) is dict and type(plan.get("source_sha256")) is dict
                 and plan["source_sha256"].get(BASE_SOURCE) == BASE_SHA256
                 and plan["source_sha256"].get(BASE_TEST) == BASE_TEST_SHA256
                 and plan["source_sha256"].get(SEALER_BASE) == SEALER_BASE_SHA256)
        return validate(root, plan, private=private)

    core.validate_registration = validate_bound
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def utility_policy():
    return _get_core().utility_policy()


def build_registration(root, *, source_sha256, input_sha256, runtime_sha256,
                       qualification_proof_sha256, private=False):
    return _get_core().build_registration(root, source_sha256=source_sha256, input_sha256=input_sha256,
        runtime_sha256=runtime_sha256, qualification_proof_sha256=qualification_proof_sha256, private=private)


def validate_registration(root, plan, *, private=False):
    return _get_core().validate_registration(root, plan, private=private)


def load_registration(root, sha, *, private=False):
    return _get_core().load_registration(root, sha, private=private)


def private_preflight(root, sha):
    return _get_core().private_preflight(root, sha)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)
