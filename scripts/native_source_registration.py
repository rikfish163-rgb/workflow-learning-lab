"""Plain, source-bound metadata for one independent native JSON source pilot.

The builder returns a validated value and never writes, queues, starts a world,
contacts a service, imports AppWorld, or loads a tokenizer/model. Public checks
read only portable source and public metadata. Explicit private preflight also
rehashes the copied assets and runtime and checks the private storage paths.
CPU fixtures and historical outcomes cannot supply source-pilot observations.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
from pathlib import Path, PurePosixPath

ID = "qwen38-native-source-pilot-001"
PLAN = "artifacts/qwen38-native-source-pilot-plan-001.json"
PORT = 18991
SERVICE_PLAN = "artifacts/qwen38-b3-service-plan-001.json"
SERVICE_SHA = "d7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5"
SERVICE_PARENT = "artifacts/qwen38-service-idle180-plan-001.json"
MODEL_ARM = "B3"
MODEL_ID = "qwen38-b3-matched-b3-001"
TASK_IDS = (
    "22cc237_2", "27e1026_2", "287e338_2", "2a163ab_2", "60d0b5b_2", "692c77d_2",
    "82e2fac_2", "b0a8eae_2", "b7a9ee9_2", "c901732_2", "cf6abd2_2", "d0b1f43_2",
)
BATCH_TASK_IDS = tuple(TASK_IDS[index:index + 4] for index in range(0, 12, 4))
BATCH_GPU_SECONDS = 7200
WAIT_SECONDS = 86400
GATE = "artifacts/qwen38-b3-inference-gate-integration-001.json"
GATE_B3 = "artifacts/b3-inference-gate-result-001/remote-artifacts/qwen38-b3-inference-gate-b3-001.json"
NATIVE_PROOF = "artifacts/native-appworld-replay-scorer-root-CPU-001.json"
LAUNCH_PROOF = "artifacts/native-source-launch-root-CPU-001.json"
PARTITION = "artifacts/partitions-v1/manifest.json"
PROVENANCE = "artifacts/scenario-provenance-audit-003.json"
ORIGIN_DESIGN = "artifacts/qwen38-origin-recovery-design-001.json"
CONTEXT_PROOF_ID = "native-source-initial-context-CPU-001"
MODEL_PATH = "/home/data4/hts/workflow-learning-lab/models/qwen38-27b-1d4bf0f2"
MODEL_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
PRIVATE_STATE_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-source-pilot-001"
TMPDIR = PRIVATE_STATE_ROOT + "/runtime-tmp"
ASSET_ROOT = PRIVATE_STATE_ROOT + "/native-cpu-assets-001"
ASSET_MANIFEST = ASSET_ROOT + "/manifests/remote-after.json"
ASSET_MANIFEST_SHA256 = "8f28a3f4c4f02442429f472f1c1a5cce2a2f91f4cbf8e86f927f249af165f885"
MINIMUM_FREE_BYTES = 5 * 1024**3
ASSET_FILE_COUNT = 27025
ASSET_DIRECTORY_COUNT = 4605
ASSET_BYTES = 322289155
TOKENIZER_SHA256 = {
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
REQUIRED_RUNTIME_PATHS = frozenset({
    "/usr/bin/python3.12",
    "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/pyvenv.cfg",
    "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/lib/python3.12/site-packages/wll_borrowed_dependencies.pth",
    *(MODEL_PATH + "/" + name for name in TOKENIZER_SHA256),
    MODEL_PATH + "/config.json",
})
NATIVE_SOURCES = frozenset({
    "scripts/native_appworld_actor.py", "scripts/native_appworld_controller.py",
    "scripts/native_appworld_model_client.py", "scripts/native_appworld_replay.py",
    "scripts/native_appworld_scorer.py", "scripts/native_appworld_tools.py",
    "scripts/native_appworld_worker.py", "scripts/native_appworld_worker_client.py",
    "scripts/native_verification_token_counter.py", "scripts/verification_token_counter.py",
})
NEW_SOURCES = frozenset({
    "scripts/native_source_registration.py", "tests/test_native_source_registration.py",
    "scripts/native_source_gpu_queue.py", "tests/test_native_source_gpu_queue.py",
    "scripts/native_source_episode.py", "tests/test_native_source_episode.py",
    "scripts/run_native_source_pilot.py", "tests/test_run_native_source_pilot.py",
})
PREPARATION_SOURCES = frozenset({
    "scripts/prepare_native_source_contexts.py", "tests/test_prepare_native_source_contexts.py",
})
REQUIRED_SOURCES = NEW_SOURCES | NATIVE_SOURCES | PREPARATION_SOURCES | {
    "tests/test_native_appworld_actor.py", "tests/test_native_appworld_controller.py",
    "tests/test_native_appworld_model_client.py", "tests/test_native_appworld_replay.py",
    "tests/test_native_appworld_scorer.py", "tests/test_native_appworld_tools.py",
    "tests/test_native_appworld_worker.py", "tests/test_native_appworld_worker_client.py",
    "tests/test_native_verification_token_counter.py", "tests/test_verification_token_counter.py",
    "scripts/gpu_idle_watch_b3.py", "config/gpu-policy.json",
    "scripts/run_foundation_pilot.py", "scripts/run_qwen38_b3_dev.py",
    "scripts/run_qwen38_capability.py", "scripts/run_timed_tasks_v3.py",
    "scripts/serve_qwen38_b3.py", "scripts/serve_qwen38.py", "scripts/probe_chunked_prefill.py",
    "wll/protocol.py", "wll/gpu_idle.py", "wll/gpu_policy.py", "wll/storage.py",
    "scripts/documentation_feedback.py", "scripts/owned_appworld_v3.py",
    "scripts/probe_observation_visibility.py", "scripts/report_timed_tasks.py",
    "scripts/run_queued_decoder_completion.py", "scripts/run_timed_tasks.py",
    "scripts/run_timed_tasks_v2.py", "wll/__init__.py", "wll/appworld.py",
    "wll/appworld_v2.py", "wll/cli.py", "wll/model.py", "wll/prompts.py",
    "wll/runner.py", "wll/training.py",
}
FROZEN_SOURCE_SHA256 = {
    "scripts/completion_gpu_queue.py": "928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8",
}
REQUIRED_SOURCES |= FROZEN_SOURCE_SHA256.keys()
FIXED_INPUT_SHA256 = {
    SERVICE_PLAN: SERVICE_SHA,
    SERVICE_PARENT: "040a6a4476d1b4c6b4ef1d78d2e76fe1e600783b47c121d7ef7cf7a1ab668889",
    GATE: "ae3e5c90a88a085360ddf2f1f9fa237fb2c0a0a1fbbc541eaa65ac8e8d8eed67",
    GATE_B3: "740ec1c024671228fe06084504f68c57c787167fc60b313ea907b2472e749016",
    NATIVE_PROOF: "2a8708a0065fd5444e0db8b6464b92eb7d85cfa69499f2e65e85199f48742221",
    PARTITION: "3732b02da71c4031ba3a831c727d1956bb43872552841db275a86900e0dcf4a6",
    "artifacts/partitions-v1/origin.txt": "c8d69221c9ba3967b08fedd1e7c6dff31e0c3817101c24638c3ffeb1dc062c46",
    "artifacts/partitions-v1/utility.txt": "d9b24521f14ad512f36328f9561789db2424bc0471a215045a889fc866ce6580",
    "artifacts/partitions-v1/distill.txt": "04934c1e3a3295c6613b84e91ddbc3af9acbd2bbfc953bb59a59531526c4c068",
    PROVENANCE: "8e5b44de014b95c5468a60971060d3eb20220af9329c988ec3c0f2c2a494e213",
    "artifacts/scenario-provenance-audit-root-CPU-003.json": "4e953e20a745fbd327335f04201cebdcdc0ed97502bea3a4f7920a154cedbe45",
    ORIGIN_DESIGN: "5563a02031871d1ac34a50e3b2c685485b3d1e4e45babc269021c814ac35e46e",
    "docs/native-cohort-registration-audit-001.md": "e54d30f010e4ce158bfc4033190b4b0a2e386b82f603b0a5749bff4823fec8f4",
    "docs/native-workflow-learning-protocol-draft-001.md": "09bd5cb33dd66c41d9103f5e6d74fa90d03ba02d28c8050f20b0d1617df4f069",
    "docs/native-appworld-replay-scorer-001.md": "2b2f4cbd70761bc6b7f51c0c2bd917c55097b8fb826142f48973e7a7beacbb8e",
    "docs/remote-native-CPU-runtime-2026-11-02.md": "9834d62f0196f87b0bcaaac356c8cf697980c85e01cbedc6145de04db528e42a",
    "artifacts/native-cpu-assets-data-verification-001.json": "c27d816115a663cc34ca5d46c29dcb576cba2daf96291eb4cf9bbcbdd3efb10e",
    "artifacts/native-token-counter-root-CPU-001.json": "d6fcf3e512c9972b0a7d9de6dca04b3816261c5ed6279e7d8df8c42734cf30f1",
    "config/gpu-policy.json": "22dec644b27956ef8eb379cdc6560fceafe89b9d5e0a45212f5420a267c0e166",
}
_ERROR = "native_source_registration_invalid"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_PRIVATE_NAMES = {
    "tasks.json", "tasks.jsonl", "task.json", "spec.json", "solution.py", "solution.json",
    "actor.json", "actor.jsonl", "episode.json", "episodes.jsonl", "request.jsonl",
    "responses.jsonl", "steps.jsonl", "events.jsonl", "worker-trace.jsonl",
    "credentials.json", "passwords.json", "ground_truth.json", "ground-truth.json",
}


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _values(value, depth=0):
    _require(depth <= 40)
    if type(value) is str:
        value.encode("utf-8", errors="strict")
    elif type(value) is float:
        _require(math.isfinite(value))
    elif type(value) is dict:
        for key, child in value.items():
            _require(type(key) is str)
            _values(key, depth + 1)
            _values(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _values(child, depth + 1)
    else:
        _require(value is None or type(value) in (bool, int))


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(v, right[k]) for k, v in left.items())
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result)
        result[key] = value
    return result


def _reject(_value):
    raise ValueError(_ERROR)


def _json(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject)
    _values(value)
    return value


def _relative(name, *, public=True):
    _require(type(name) is str and 0 < len(name) < 1024 and "\\" not in name
             and all(ord(char) > 32 and ord(char) != 127 for char in name))
    path = PurePosixPath(name)
    _require(not path.is_absolute() and all(part not in {".", ".."} for part in path.parts)
             and str(path) == name)
    if public:
        _require(path.parts[0] in {"scripts", "tests", "wll", "config", "docs", "artifacts"}
                 and not set(path.parts) & {"private-state", "tasks", "ground_truth", "secrets"}
                 and path.name not in _PRIVATE_NAMES)
    return path.parts


def _directory(path):
    path = Path(os.path.abspath(path))
    descriptor = os.open("/", _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_file(root, parts, *, maximum=32_000_000, return_raw=True):
    directory = _directory(root)
    file = None
    try:
        for component in parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
        file = os.open(parts[-1], _FILE_FLAGS, dir_fd=directory)
        before = os.fstat(file)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and 0 <= before.st_size <= maximum)
        digest, chunks, size = hashlib.sha256(), [], 0
        while True:
            block = os.read(file, min(1024 * 1024, maximum + 1 - size))
            if not block:
                break
            size += len(block)
            _require(size <= maximum)
            digest.update(block)
            if return_raw:
                chunks.append(block)
        _require(size == before.st_size and _stamp(before) == _stamp(os.fstat(file))
                 == _stamp(os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)))
        return b"".join(chunks) if return_raw else digest.hexdigest()
    finally:
        if file is not None:
            os.close(file)
        os.close(directory)


def public_read(root, name):
    """Read one regular public file with no symlink traversal or private fallback."""
    try:
        return _read_file(root, _relative(name))
    except Exception:
        raise ValueError(_ERROR) from None


def _read(root, name):
    return _json(public_read(root, name))


def _public_hash(root, name):
    return hashlib.sha256(public_read(root, name)).hexdigest()


def _absolute_hash(name):
    _require(type(name) is str and Path(name).is_absolute()
             and os.path.normpath(name) == name and ".." not in Path(name).parts)
    return _read_file("/", Path(name).parts[1:], maximum=1024**3, return_raw=False)


def static_backend(backend):
    """Use the existing service's static projection without mutating its proof."""
    _require(type(backend) is dict)
    return {key: copy.deepcopy(value) for key, value in backend.items()
            if key not in {"gpu_policy", "loading_peak_reserved_bytes"}}


def _cohort(root):
    for name in (PARTITION, PROVENANCE, ORIGIN_DESIGN):
        _require(_public_hash(root, name) == FIXED_INPUT_SHA256[name])
    partition, provenance, origin = (_read(root, name) for name in (PARTITION, PROVENANCE, ORIGIN_DESIGN))
    roles = partition["task_ids"]
    _require(type(roles) is dict and set(roles) == {"origin", "utility", "distill"})
    scenario_sets = {}
    for role, count in (("origin", 12), ("utility", 9), ("distill", 9)):
        ids = roles[role]
        _require(type(ids) is list and len(ids) == len(set(ids)) == count * 3
                 and all(type(value) is str and re.fullmatch(r"[a-f0-9]{7}_[123]", value) for value in ids))
        scenarios = list(dict.fromkeys(value[:-2] for value in ids))
        _require(len(scenarios) == count and ids == [s + "_" + str(v) for s in scenarios for v in (1, 2, 3)])
        scenario_sets[role] = set(scenarios)
        name = "artifacts/partitions-v1/" + role + ".txt"
        _require(_public_hash(root, name) == FIXED_INPUT_SHA256[name]
                 and public_read(root, name).decode("utf-8").splitlines() == ids)
    _require(all(not scenario_sets[a] & scenario_sets[b]
                 for a, b in (("origin", "utility"), ("origin", "distill"), ("utility", "distill"))))
    selected = [value for value in roles["origin"] if value.endswith("_2")]
    audit = provenance["original_audit_result"]
    train = audit["official_split_metadata"]["train"]
    _require(selected == list(TASK_IDS) == origin["selection"]["task_ids"]
             and origin["selection"]["project_unseen"] is False
             and set(train["task_ids_official_order"]) == set(sum(roles.values(), []))
             and train["task_count"] == 90 and train["scenario_count"] == 30)
    for role in roles:
        _require(set(audit["known_exposure_and_role"]["train_" + role + "_scenario_ids"])
                 == scenario_sets[role])
    return copy.deepcopy(roles), origin


def expected_design(root):
    """Return the complete fixed native protocol from public metadata only."""
    try:
        roles, origin = _cohort(root)
        hint = origin["protocol"]["hint"]
        _require(type(hint) is str and bool(hint))
        return {
            "interface": {"name": "native_JSON_public_API", "public_API_count": 473,
                          "action": "one_JSON_action", "full_task_and_documentation": True,
                          "policy_Python": False, "NoHintSpecificTask": True},
            "cohort": {"source_task_ids": list(TASK_IDS), "source_scenarios": 12,
                       "selection": "manifest_origin_order_variant_2", "official_split": "train",
                       "project_unseen": False, "all_train_roles_already_exposed": True,
                       "utility_task_ids_reserved": roles["utility"], "utility_scenarios": 9,
                       "distill_task_ids_reserved": roles["distill"], "distill_scenarios": 9,
                       "E": "unknown", "official_tests_reserved_unrun": True},
            "backbone": {"arm": MODEL_ARM, "model_id": MODEL_ID, "service_plan": SERVICE_PLAN,
                         "service_plan_sha256": SERVICE_SHA, "trained_on_distill_scenarios": 9,
                         "all_source_and_branch_initializations_identical": True,
                         "new_training": False, "learning_or_writer_between_items": False},
            "schedule": {"mode": "per_source_interleaved", "task_ids": list(TASK_IDS),
                         "per_source": ["first_attempt", "eligible_pair_once"],
                         "branch_orders": [["control", "review"] if index % 2 == 0
                                           else ["review", "control"] for index in range(12)],
                         "batch_task_ids": [list(batch) for batch in BATCH_TASK_IDS],
                         "environment_seed": 100, "generation_seed_base": 17,
                         "source_seed": "17 + generation_index_zero_based",
                         "suffix_seed": "17 + prefix_action_count + generation_index_zero_based",
                         "paired_generation_seeds_identical": True, "automatic_retry": False,
                         "import_old_results_or_retry_unknown": False},
            "boundary": {"trigger": "first_actor_visible_confirmed_ObservedAPIError_HTTP4xx",
                         "prefix_min": 1, "prefix_max": 54, "uses_gold_or_terminal_score": False,
                         "no_boundary": "known_ineligible", "completed_prefix": "known_ineligible",
                         "fresh_process_and_world_per_branch": True,
                         "replay": "full_prefix_exact_typed_observation_equality",
                         "mismatch": "stop_study_unknown_no_retry"},
            "review": {"hint": hint, "hint_source": ORIGIN_DESIGN + "#/protocol/hint",
                       "placement": "one_user_message_after_full_replayed_prefix",
                       "control_insertions": 0, "review_insertions": 1,
                       "task_specific_additions": False, "other_memory": False,
                       "student_projection": "remove_only_the_exact_fixed_hint_message",
                       "initial_suffix_contexts": "count_both_before_either_suffix_generation",
                       "either_initial_context_does_not_fit": "paired_known_ineligible",
                       "truncation_or_resampling": False},
            "budgets": {
                "max_input_tokens": 15360, "max_new_tokens": 1024,
                "source": {"max_generations": 60, "max_action_attempts": 60,
                           "max_API_invocations": 60, "max_prompt_tokens": 921600,
                           "max_completion_tokens": 61440},
                "suffix": {"max_generations": 6, "max_action_attempts_including_replay": 60,
                           "max_API_invocations_including_replay": 60, "max_prompt_tokens": 92160,
                           "max_completion_tokens": 6144},
                "actor_generation_seconds": 600, "setup_seconds": 120,
                "prefix_replay_charged_to_setup": True,
                "cleanup_seconds": 60, "worker_wall_seconds": 780,
                "parent_unit_seconds": 900, "scorer_seconds": 120,
                "prefix_check_seconds": 120, "maximum_prefix_checks": 12,
                "total_GPU_seconds": 21600, "serial_GPU_jobs": 3,
                "per_GPU_job_seconds": BATCH_GPU_SECONDS,
                "GPU_wall_includes_load_start_stop": True,
                "admission_wait_seconds_per_job": WAIT_SECONDS,
                "admission_wait_separate_from_GPU_time": True,
                "GPU_cap_exhausted": "stop_preserve_unstarted", "B3_training_budget_reuse": False,
            },
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                         "presence_penalty": 1.5, "repetition_penalty": 1.0,
                         "thinking": False, "context_tokens": 16384},
            "completion": {"accepted_API_self_report": "separate_from_official_boolean",
                           "score": "separate_CPU_process_after_actor_exit",
                           "help": "both_confirmed_complete_and_control_false_and_review_true",
                           "score_or_verifier_details_visible_to_actor": False},
            "data_feasibility_gate": {"minimum_complete_help_scenarios": 4,
                                      "minimum_distinct_student_targets": 20,
                                      "complete_all_12_without_unknown_or_unstarted": True,
                                      "target_source": "review_new_nonreplay_actions_only",
                                      "valid_reply_required": True, "raw_generation_equals_action": True,
                                      "complete_untruncated_student_history": True,
                                      "target_digest": "canonical_JSON_action_UTF8_SHA256",
                                      "deduplication": "at_least_as_strict_as_raw_action_text",
                                      "repeat_targets": False, "CPU_fixture_data": False,
                                      "unknown_or_unstarted": "not_evaluable"},
            "stopping": {"infrastructure_unknown": "stop_study", "replay_mismatch": "stop_study",
                         "save_log_score_or_cleanup_failure": "stop_study",
                         "ordinary_known_model_failure": "continue_fixed_schedule",
                         "add_tasks_seeds_or_triggers": False},
            "resource_policy": {"host": "amax", "ssh": "4090", "allowed_physical_GPUs": [8, 9],
                                "maximum_simultaneous_project_GPUs": 1,
                                "continuous_idle_seconds": 180, "allocator_cap_GiB": 20,
                                "port": PORT, "tmp_dir": TMPDIR,
                                "private_run_root": PRIVATE_RUN_ROOT,
                                "minimum_free_bytes": MINIMUM_FREE_BYTES,
                                "use_existing_per_device_admission_and_project_lock": True},
            "outputs": {"source_baseline_separate_from_recovery": True, "denominator": 12,
                        "writer": False, "utility": False, "dev_or_test_evaluation": False,
                        "training": False, "old_001_through_005_modified_or_imported": False,
                        "recovery_help_is_not_learning_or_transfer": True,
                        "required_costs": ["generations", "unreturned_requests", "prompt_tokens",
                                           "completion_tokens", "action_attempts", "API_invocations",
                                           "upstream_requests", "CPU_counting", "source_replay",
                                           "branch_replay", "setup", "save", "score", "logging",
                                           "cleanup", "GPU_loading_and_wall", "admission_wait"]},
        }
    except Exception:
        raise ValueError(_ERROR) from None


def _binding_map(value, *, runtime=False):
    _require(type(value) is dict and 0 < len(value) <= 4096)
    for name, digest in value.items():
        _require(_sha(digest))
        if runtime:
            _require(type(name) is str and Path(name).is_absolute()
                     and os.path.normpath(name) == name and ".." not in Path(name).parts
                     and all(ord(char) > 32 and ord(char) != 127 for char in name))
        else:
            _relative(name)


def _context(root, binding, sources, inputs):
    _require(type(binding) is dict and set(binding) == {"path", "sha256"}
             and _sha(binding["sha256"]) and binding["path"].endswith(".json")
             and inputs.get(binding["path"]) == binding["sha256"])
    proof = _read(root, binding["path"])
    fields = {"id", "status", "task_ids", "rows", "source_sha256", "preparation", "model_path",
              "model_config_sha256", "tokenizer_files_sha256", "CUDA_VISIBLE_DEVICES",
              "cuda_initialized", "model_weights_loaded", "GPU_jobs", "model_calls", "task_worlds_started",
              "task_metadata_loads", "ground_truth_load_attempts"}
    _require(type(proof) is dict and set(proof) == fields
             and proof["id"] == CONTEXT_PROOF_ID and proof["status"] == "measured_CPU_initial_contexts"
             and _same(proof["task_ids"], list(TASK_IDS)) and proof["model_path"] == MODEL_PATH
             and proof["model_config_sha256"] == MODEL_CONFIG_SHA256
             and _same(proof["tokenizer_files_sha256"], TOKENIZER_SHA256)
             and proof["CUDA_VISIBLE_DEVICES"] == ""
             and proof["cuda_initialized"] is proof["model_weights_loaded"] is False
             and all(type(proof[key]) is int and proof[key] == 0
                     for key in ("GPU_jobs", "model_calls", "task_worlds_started", "ground_truth_load_attempts"))
             and type(proof["task_metadata_loads"]) is int and proof["task_metadata_loads"] == 12)
    proof_sources = proof["source_sha256"]
    _binding_map(proof_sources)
    _require(NATIVE_SOURCES <= proof_sources.keys()
             and all(sources.get(name) == digest for name, digest in proof_sources.items()))
    preparation = proof["preparation"]
    _require(type(preparation) is dict and set(preparation) == {"path", "sha256"}
             and preparation["path"] == "scripts/prepare_native_source_contexts.py"
             and _sha(preparation["sha256"])
             and proof_sources.get(preparation["path"]) == preparation["sha256"])
    rows = proof["rows"]
    _require(type(rows) is list and len(rows) == len(TASK_IDS))
    for task_id, row in zip(TASK_IDS, rows, strict=True):
        _require(type(row) is dict and set(row) == {"task_id", "input_tokens", "initial_fits",
                                                   "metadata_sha256", "messages_sha256"}
                 and row["task_id"] == task_id and type(row["input_tokens"]) is int
                 and 1 <= row["input_tokens"] <= 15360 and row["initial_fits"] is True
                 and _sha(row["metadata_sha256"]) and _sha(row["messages_sha256"]))
    return proof


def _service(root, plan, sources, inputs, runtime):
    service, parent, gate, measured, native = (
        _read(root, name) for name in (SERVICE_PLAN, SERVICE_PARENT, GATE, GATE_B3, NATIVE_PROOF))
    _require(service["id"] == "qwen38-b3-service-001" and service["status"] == "registered"
             and service["parent_service_plan"] == SERVICE_PARENT
             and service["parent_service_plan_sha256"] == FIXED_INPUT_SHA256[SERVICE_PARENT]
             and service["model_ids"][MODEL_ARM] == MODEL_ID)
    for previous in (service, parent):
        for field, target in (("source_sha256", sources), ("input_sha256", inputs),
                              ("runtime_sha256", runtime)):
            _require(type(previous[field]) is dict
                     and all(target.get(name) == digest for name, digest in previous[field].items()))
    _require(gate["phase"] == "complete" and gate["all_resources_released"] is True
             and gate["plan_sha256"] == SERVICE_SHA
             and measured["final_health"]["model"] == MODEL_ID
             and measured["final_health"]["ready"] is True
             and measured["final_health"]["busy"] is False
             and _same(plan["expected_backend_static"],
                       {MODEL_ARM: static_backend(measured["final_health"]["backend"])}))
    policy = _read(root, "config/gpu-policy.json")
    _require(_same(plan["gpu_policy"], policy)
             and plan["gpu_policy_sha256"] == FIXED_INPUT_SHA256["config/gpu-policy.json"]
             and policy["hostname"] == "amax" and type(policy["idle_seconds"]) is int
             and policy["idle_seconds"] == 180 and policy["poll_seconds"] == 15
             and policy["max_poll_gap_seconds"] == 35 and policy["max_used_mib"] == 128
             and policy["max_utilization_percent"] == 1 and policy["minimum_free_mib"] == 22528
             and _same(policy["allowed_gpus"], {
                 "8": "GPU-ecc8cd45-fdaa-b5ee-e54a-bf2ef7df8f82",
                 "9": "GPU-4fc78635-e0b9-1acf-774f-9debcfc7fa47"}))
    _require(native["status"] == "two_fresh_mutating_prefix_replays_context_state_and_generic_scoring_verified_CPU_only"
             and native["new_scientific_study_registered"] is False
             and native["new_training_authorized"] is False
             and type(native["new_qualified_experiences"]) is int
             and native["new_qualified_experiences"] == 0)
    # The prior CPU proof remains immutable historical engineering evidence.
    # Current sources may evolve for this independently registered native pilot.
    launch = _read(root, LAUNCH_PROOF)
    _require(type(launch) is dict)
    _binding_map(launch.get("source_sha256"))
    _require(type(launch) is dict and launch.get("status") == "CPU_native_source_lifecycle_passed"
             and type(launch.get("source_sha256")) is dict
             and (NEW_SOURCES | NATIVE_SOURCES | PREPARATION_SOURCES) <= launch["source_sha256"].keys()
             and all(sources.get(name) == digest for name, digest in launch["source_sha256"].items())
             and all(type(launch.get(key)) is int and launch[key] == 0
                     for key in ("GPU_jobs", "model_calls", "new_qualified_experiences"))
             and type(launch.get("task_worlds_started")) is int and launch["task_worlds_started"] >= 0
             and launch.get("is_policy_episode") is False
             and launch.get("new_scientific_study_registered") is False
             and launch.get("training_authorized") is False)


def _private_directory(path):
    descriptor = _directory(path)
    try:
        info = os.fstat(descriptor)
        _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700)
    finally:
        os.close(descriptor)


def _private_assets():
    _private_directory(PRIVATE_STATE_ROOT)
    _private_directory(TMPDIR)
    _private_directory(PRIVATE_RUN_ROOT)
    _private_directory(ASSET_ROOT)
    _private_directory(ASSET_ROOT + "/manifests")
    _require(shutil.disk_usage(TMPDIR).free >= MINIMUM_FREE_BYTES
             and shutil.disk_usage(PRIVATE_RUN_ROOT).free >= MINIMUM_FREE_BYTES)
    info = os.lstat(ASSET_MANIFEST)
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
             and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1)
    _require(_absolute_hash(ASSET_MANIFEST) == ASSET_MANIFEST_SHA256)
    manifest = _json(_read_file("/", Path(ASSET_MANIFEST).parts[1:], maximum=64_000_000))
    _require(manifest["status"] == "complete" and set(manifest["trees"]) == {"data", "deps"})
    total_files = total_directories = total_bytes = 0
    for tree_name, tree in manifest["trees"].items():
        _require(type(tree) is dict and set(tree) == {"directories", "files"})
        expected_files, expected_directories = {}, {}
        for field, expected in (("files", expected_files), ("directories", expected_directories)):
            _require(type(tree[field]) is list)
            for row in tree[field]:
                _require(type(row) is dict and type(row.get("path")) is str)
                name = row["path"]
                if name != ".":
                    _relative(name, public=False)
                _require(name not in expected)
                expected[name] = row
        actual_files, actual_directories = set(), set()
        root = Path(ASSET_ROOT) / tree_name
        for directory, children, files in os.walk(root, followlinks=False):
            relative = Path(directory).relative_to(root).as_posix()
            actual_directories.add(relative)
            info = os.lstat(directory)
            row = expected_directories.get(relative)
            _require(type(row) is dict and stat.S_ISDIR(info.st_mode)
                     and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == row["mode"])
            for child in children:
                _require(stat.S_ISDIR(os.lstat(Path(directory) / child).st_mode))
            for file in files:
                path = Path(directory) / file
                name = path.relative_to(root).as_posix()
                actual_files.add(name)
                row = expected_files.get(name)
                info = os.lstat(path)
                _require(type(row) is dict and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                         and info.st_uid == os.getuid() and type(row["bytes"]) is int
                         and info.st_size == row["bytes"] and stat.S_IMODE(info.st_mode) == row["mode"]
                         and _sha(row["sha256"]) and _absolute_hash(str(path)) == row["sha256"])
                total_files += 1
                total_bytes += info.st_size
        _require(actual_files == set(expected_files) and actual_directories == set(expected_directories))
        total_directories += len(actual_directories)
    _require(total_files == ASSET_FILE_COUNT and total_directories == ASSET_DIRECTORY_COUNT
             and total_bytes == ASSET_BYTES)


def validate_registration(root, plan, *, private=False):
    """Authenticate all public pins; optionally perform actual remote preflight."""
    try:
        _require(type(private) is bool)
        _values(plan)
        fields = {"id", "status", "design", "task_ids", "batch_task_ids", "model_arm", "model_id",
                  "port", "service_plan", "service_plan_sha256", "expected_backend_static",
                  "gpu_policy", "gpu_policy_sha256", "source_sha256", "input_sha256",
                  "runtime_sha256", "context_proof", "asset_manifest", "asset_manifest_sha256",
                  "tmp_dir", "private_run_root", "python", "gpu_seconds", "total_gpu_seconds",
                  "admission_wait_seconds", "automatic_retry", "training_authorized", "writer_authorized",
                  "utility_authorized", "dev_or_test_authorized"}
        _require(type(plan) is dict and set(plan) == fields
                 and plan["id"] == ID and plan["status"] == "registered"
                 and _same(plan["design"], expected_design(root))
                 and _same(plan["task_ids"], list(TASK_IDS))
                 and _same(plan["batch_task_ids"], [list(batch) for batch in BATCH_TASK_IDS])
                 and plan["model_arm"] == MODEL_ARM and plan["model_id"] == MODEL_ID
                 and type(plan["port"]) is int and plan["port"] == PORT
                 and plan["service_plan"] == SERVICE_PLAN and plan["service_plan_sha256"] == SERVICE_SHA
                 and plan["asset_manifest"] == ASSET_MANIFEST
                 and plan["asset_manifest_sha256"] == ASSET_MANIFEST_SHA256
                 and plan["tmp_dir"] == TMPDIR and plan["private_run_root"] == PRIVATE_RUN_ROOT
                 and plan["python"] == PYTHON
                 and type(plan["gpu_seconds"]) is int and plan["gpu_seconds"] == BATCH_GPU_SECONDS
                 and type(plan["total_gpu_seconds"]) is int and plan["total_gpu_seconds"] == 21600
                 and type(plan["admission_wait_seconds"]) is int and plan["admission_wait_seconds"] == WAIT_SECONDS
                 and all(plan[field] is False for field in (
                     "automatic_retry", "training_authorized", "writer_authorized",
                     "utility_authorized", "dev_or_test_authorized")))
        sources, inputs, runtime = (plan[key] for key in ("source_sha256", "input_sha256", "runtime_sha256"))
        _binding_map(sources)
        _binding_map(inputs)
        _binding_map(runtime, runtime=True)
        _require(REQUIRED_SOURCES <= sources.keys() and REQUIRED_RUNTIME_PATHS <= runtime.keys()
                 and LAUNCH_PROOF in inputs
                 and all(inputs.get(name) == digest for name, digest in FIXED_INPUT_SHA256.items())
                 and all(sources.get(name) == digest for name, digest in FROZEN_SOURCE_SHA256.items()))
        for name, digest in {**sources, **inputs}.items():
            _require(name not in sources or name not in inputs or sources[name] == inputs[name])
            _require(_public_hash(root, name) == digest)
        _require(all(runtime.get(MODEL_PATH + "/" + name) == digest
                     for name, digest in TOKENIZER_SHA256.items())
                 and runtime.get(MODEL_PATH + "/config.json") == MODEL_CONFIG_SHA256)
        _service(root, plan, sources, inputs, runtime)
        _context(root, plan["context_proof"], sources, inputs)
        if private:
            _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
            _require(all(_absolute_hash(name) == digest for name, digest in runtime.items()))
            _private_assets()
        return copy.deepcopy(plan)
    except Exception:
        raise ValueError(_ERROR) from None


def load_registration(root, sha, *, private=False):
    """Read only the new plan and authenticate its exact bytes and fixed design."""
    try:
        _require(_sha(sha))
        raw = public_read(root, PLAN)
        _require(hashlib.sha256(raw).hexdigest() == sha)
        return validate_registration(root, _json(raw), private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def register(root, *, source_sha256, input_sha256, runtime_sha256, context_proof, private=False):
    """Build and validate an independent plan; the caller owns exclusive writing."""
    try:
        plan = {
            "id": ID, "status": "registered", "design": expected_design(root),
            "task_ids": list(TASK_IDS), "batch_task_ids": [list(batch) for batch in BATCH_TASK_IDS],
            "model_arm": MODEL_ARM, "model_id": MODEL_ID, "port": PORT,
            "service_plan": SERVICE_PLAN, "service_plan_sha256": SERVICE_SHA,
            "expected_backend_static": {MODEL_ARM: static_backend(_read(root, GATE_B3)["final_health"]["backend"])},
            "gpu_policy": _read(root, "config/gpu-policy.json"),
            "gpu_policy_sha256": FIXED_INPUT_SHA256["config/gpu-policy.json"],
            "source_sha256": copy.deepcopy(source_sha256), "input_sha256": copy.deepcopy(input_sha256),
            "runtime_sha256": copy.deepcopy(runtime_sha256), "context_proof": copy.deepcopy(context_proof),
            "asset_manifest": ASSET_MANIFEST, "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "tmp_dir": TMPDIR, "private_run_root": PRIVATE_RUN_ROOT, "python": PYTHON,
            "gpu_seconds": BATCH_GPU_SECONDS, "total_gpu_seconds": 21600,
            "admission_wait_seconds": WAIT_SECONDS, "automatic_retry": False,
            "training_authorized": False, "writer_authorized": False, "utility_authorized": False,
            "dev_or_test_authorized": False,
        }
        return validate_registration(root, plan, private=private)
    except Exception:
        raise ValueError(_ERROR) from None
