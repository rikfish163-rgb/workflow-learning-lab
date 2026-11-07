"""Independent source006 after an unknown predecessor, with a real CPU proof.

Builders return values only. Historical unknown costs remain unknown. Public
bindings preserve source004 assets and source005 code, without a source005 plan.
Private preflight reads terminal/probe metadata only, never Task/trace/GT bodies.
No model, world, scorer, queue, registration write or automatic retry is started.
"""

import copy
import hashlib
import math
from pathlib import Path

from scripts import native_source_gpu_queue_004 as queue4
from scripts import native_source_registration as public
from scripts import native_source_registration_004 as previous
from scripts import native_source_registration_005 as budgets5

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-source-pilot-006"
PLAN = "artifacts/qwen38-native-source-pilot-plan-006.json"
PRIVATE_RUN_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-006"
PREDECESSOR_PLAN = previous.PLAN
PREDECESSOR_PLAN_SHA256 = "1f26e164c9d315acbe5e98f9c2340a5abd4e6d878ed9f2de5ec6e7909521b4bb"
PREDECESSOR_TERMINAL = "artifacts/native-source-pilot-terminal-root-004.json"
PREDECESSOR_TERMINAL_SHA256 = "d8d40ce32b05558340db31739919c1f68e640b3faa58c37a2fd52183f9689a4f"
PREDECESSOR_RESULT_SHA256 = "6315335b68f9e706b2dd9341769e0cc3e6286b222cbafd07ff017f257a0cf0b1"
CPU_RESULT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-metadata-startup-CPU-006-001/probe-result.json"
CPU_SCOPE = "actual_CPU_metadata_save_close_nonpolicy_no_model_no_score"
TASK_IDS = previous.TASK_IDS
BATCH_TASK_IDS = tuple((task,) for task in TASK_IDS)
BATCH_GPU_SECONDS, TOTAL_GPU_SECONDS, MODEL_READY_SECONDS = 7200, 86400, 600
RELEASE_SECONDS, SHUTDOWN_RESERVE_SECONDS = 90, 150
WAIT_SECONDS, PORT, PYTHON = previous.WAIT_SECONDS, previous.PORT, previous.PYTHON
SERVICE_PLAN, SERVICE_SHA = previous.SERVICE_PLAN, previous.SERVICE_SHA
MODEL_ARM, MODEL_ID = previous.MODEL_ARM, previous.MODEL_ID
LAUNCH_PROOF = "artifacts/native-source-launch-root-CPU-006.json"
V5_PINS = {
    "scripts/native_source_registration_005.py": "babf12edd784a5857621c33570f85828efee2d82ebee71052a6ee743c353a19c",
    "scripts/native_source_gpu_queue_005.py": "384cb576621de84e3cb2484a813f3f0e7d21f8a2fa13c51f834a96cfa4a7eb01",
    "scripts/native_source_episode_005.py": "0daba5434fea3202241de0030fed60c5229f50a0be2e07768df7ae6dec20c612",
    "scripts/run_native_source_pilot_005.py": "353182da5bf05b2aa5aaeb6def303388d61d82c9086ae8e621baa88edeb195aa",
    "tests/test_native_source_registration_005.py": "4405f96e112f007797d30bd290bbce5ef345d671a7c24a6f9f5972260b4c25f5",
    "tests/test_native_source_gpu_queue_005.py": "e61c0e20575baecca7fcf3feb2a99873adbd1819e4c707bb1debee9cf1d89356",
    "tests/test_native_source_episode_005.py": "835890f01534b60b77298af1633210d0cf73a984b19c6346a788c5c03d7b685f",
    "tests/test_run_native_source_pilot_005.py": "9c3aa7b4d9aa8654d1eeb78e6c4d611dfe969261e18f428c3bc5ba34b1655eaa",
}
REQUIRED_NEW_SOURCES = frozenset(f"{directory}/{prefix}{name}_006.py"
    for name in ("native_appworld_controller", "native_appworld_worker_client", "native_source_registration",
                 "native_source_gpu_queue", "native_source_episode", "run_native_source_pilot", "native_metadata_probe")
    for directory, prefix in (("scripts", ""), ("tests", "test_")))
CPU_SOURCES = frozenset(name for name in REQUIRED_NEW_SOURCES if any(part in name for part in (
    "native_metadata_probe", "native_appworld_controller", "native_appworld_worker_client")))
REQUIRED_SOURCES = previous.REQUIRED_SOURCES | V5_PINS.keys() | REQUIRED_NEW_SOURCES
CPU_CHECKS = frozenset({"metadata_returned", "metadata_uses_full_remaining_setup", "zero_model_generations",
    "zero_policy_actions", "zero_GPU_jobs", "no_scorer", "worker_saved_closed_exit0", "parent_and_worker_released",
    "resource_profile_verified"})
PARENT_PROFILE = {"tasks_max": 256, "memory_bytes": 6442450944, "cpu_quota_percent": 400, "runtime_seconds": 300}
WORKER_PROFILE = {"tasks_max": 128, "memory_bytes": 3221225472, "cpu_quota_percent": 200, "runtime_seconds": 780}
_ERROR = "native_source_registration_006_invalid"


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def secondary_design():
    return {"name": "ordinary_control_recovery_gate", "registered_before_source006_outcomes": True,
        "writer_bank": "ordinary_control_recovery_only", "bank_fallback": False, "primary_gate_unchanged": True,
        "full_source_denominator": 12, "all_source_statuses_costs_and_releases_known": True,
        "eligibility": "first_official_false_and_matched_control_model_complete_and_official_true",
        "review_outcome_does_not_select_bank": True, "minimum_distinct_recovery_scenarios": 4,
        "minimum_distinct_untruncated_control_targets": 20, "targets": "control_new_generations_only_no_hint_removal",
        "target_filter": {"source_order": "all_qualifying_control_scenarios_ascending_source_index",
            "format_error_is_none": True, "response_text_equals_dispatched_action": True,
            "raw_choices": "exactly_one_finish_reason_stop_message_exact_role_assistant_content_action",
            "dispatched_reply_ok_is_true": True,
            "parsed_action": "exact_name_arguments_nonempty_string_name_dict_arguments",
            "target_sha256": "sha256_native_worker_encode_parsed_action",
            "deduplication": "global_first_occurrence_in_fixed_source_and_generation_order",
            "student_messages": "full_unchanged_control_generation_intent_messages",
            "hint_deletion_or_invention": False, "completion": "included_when_same_filter_passes_no_extra_exclusions"},
        "full_history_no_truncation": True, "additional_source_model_or_world_calls": 0,
        "writer_requires_separate_registration": True, "V27_same_budget_CI_and_harm_gate_unchanged": True,
        "hint_causality_TUP_or_RSI_claim_authorized": False}


def _terminal(proof, old):
    _require(type(proof) is dict and proof.get("id") == "native-source-pilot-terminal-root-004"
        and proof.get("status") == "unknown" and proof.get("plan_sha256") == PREDECESSOR_PLAN_SHA256
        and proof.get("private_result_sha256") == PREDECESSOR_RESULT_SHA256
        and public._same(proof.get("first_attempt_status_counts"), {"known": 10, "unknown": 1, "unstarted": 1})
        and public._same(proof.get("pair_class_counts"), {"ineligible_incomplete_pair": 5,
            "ineligible_no_boundary": 5, "unknown": 1, "unstarted": 1})
        and public._same(proof.get("data_feasibility_gate"), {"status": "not_evaluable",
            "complete_help_scenarios": 0, "distinct_student_targets": 0, "source_denominator": 12})
        and proof.get("source005_conditional_eligible") is False and proof.get("source005_plan_registered") is False
        and proof.get("automatic_retry") is False and proof.get("frozen_binding_mismatches") == []
        and all(type(proof.get(key)) is int and proof[key] == 0 for key in (
            "new_writer_calls", "new_training_jobs", "new_utility_calls")))
    slots = proof.get("slots")
    _require(type(slots) is list and len(slots) == 12)
    pairs = []
    for index, (slot, task) in enumerate(zip(slots, old["task_ids"], strict=True)):
        _require(type(slot) is dict and set(slot) == {"source_index", "task_id", "pair_class", "first_attempt", "control", "review"}
                 and type(slot["source_index"]) is int and slot["source_index"] == index and slot["task_id"] == task)
        if index >= 10:
            state = "unknown" if index == 10 else "unstarted"
            _require(slot["pair_class"] == state and slot["first_attempt"] == {"status": state}
                     and slot["control"] == slot["review"] == {"status": "unstarted"})
            continue
        first = slot["first_attempt"]
        _require(set(first) == budgets5._KNOWN and first["status"] == "known"
                 and type(first["model_reported_complete"]) is bool and type(first["official_success"]) is bool
                 and public._sha(first["receipt_sha256"]))
        if slot["pair_class"] == "ineligible_no_boundary":
            _require(slot["control"] == slot["review"] == {
                "status": "known_ineligible", "reason": "no_eligible_first_4xx_prefix"})
        else:
            _require(slot["pair_class"] == "ineligible_incomplete_pair")
            for phase in ("control", "review"):
                row = slot[phase]
                _require(type(row) is dict and set(row) == budgets5._KNOWN and row["status"] == "known"
                         and type(row["model_reported_complete"]) is bool and type(row["official_success"]) is bool
                         and public._sha(row["receipt_sha256"]))
            _require(not (slot["control"]["model_reported_complete"] and slot["review"]["model_reported_complete"]))
        pairs.append(slot["pair_class"])
    _require(pairs.count("ineligible_incomplete_pair") == pairs.count("ineligible_no_boundary") == 5)
    costs = proof.get("costs")
    _require(type(costs) is dict and all(type(costs.get(key)) is int and costs[key] == value
             for key, value in {"known_episodes": 20, "unknown_episodes": 1, "known_scoring_attempts": 20}.items())
             and costs.get("study_counter_totals_remain_unknown") is True
             and costs.get("cuda_seconds") is None)
    jobs = proof.get("GPU_jobs")
    _require(type(jobs) is list and len(jobs) == 3)
    for index, job in enumerate(jobs):
        _require(type(job) is dict and type(job.get("batch_index")) is int and job["batch_index"] == index
                 and job.get("job_id") == f"{previous.ID}-comparison-{index}"
                 and job.get("release_verified") is True and job.get("runtime_overrun") is False
                 and _finite(job.get("GPU_lease_seconds")) and job["GPU_lease_seconds"] <= 7200
                 and _finite(job.get("service_wall_seconds")) and job["service_wall_seconds"] == job["GPU_lease_seconds"])
    _require(sum(job["GPU_lease_seconds"] for job in jobs) <= 21600)


def _private_terminal(old, proof):
    raw = budgets5._read_predecessor_result(old)
    _require(type(raw) is bytes and _hash(raw) == proof["private_result_sha256"])
    report = public._json(raw)
    _require(report.get("id") == previous.ID and report.get("plan_sha256") == PREDECESSOR_PLAN_SHA256
             and report.get("status") == "unknown" and public._same(report.get("slots"), proof["slots"])
             and public._same(report.get("data_feasibility_gate"), proof["data_feasibility_gate"])
             and report.get("candidate_student_target_sha256") == [] and report.get("automatic_retry") is False
             and all(type(report.get(key)) is int and report[key] == 0
                     for key in ("writer_calls", "training_jobs", "utility_or_dev_or_test_calls")))
    costs = report["costs"]
    known = [row for row in costs["episodes"] if row.get("unknown_cost") is False]
    unknown = [row for row in costs["episodes"] if row.get("unknown_cost") is True]
    _require(len(costs["episodes"]) == 21 and len(known) == 20 and len(unknown) == 1)
    gap = unknown[0]
    _require(type(gap.get("source_index")) is int and gap["source_index"] == 10 and gap.get("phase") == "first_attempt"
             and gap.get("counts_complete") is False and _finite(gap.get("parent_wall_seconds"))
             and _finite(gap.get("controller_elapsed_seconds"))
             and all(key in gap and gap[key] is None for key in ("generations", "prompt_tokens", "completion_tokens",
                 "action_attempts", "api_invocations", "upstream_requests")))
    for key in ("generation_intents", "generation_results", "confirmed_not_started", "dispatch_intents", "dispatch_results",
                "journal_returned_prompt_tokens_lower_bound", "journal_returned_completion_tokens_lower_bound"):
        _require(type(gap.get(key)) is int and gap[key] == 0)
    # Audit a known-only subtotal. The source report and its unknown row are never repaired or zero-filled.
    subtotal = copy.deepcopy(report)
    subtotal["slots"] = subtotal["slots"][:10]
    subtotal["costs"]["episodes"] = known
    budgets5._known_cost_metadata(subtotal, old)
    declared = proof["costs"]["known_episode_subtotal_only"]
    for key in ("generations", "prompt_tokens", "completion_tokens", "action_attempts", "api_invocations", "upstream_requests"):
        _require(type(declared.get(key)) is int and declared[key] == sum(row[key] for row in known))
    for label, key in (("controller_wall_seconds", "controller_elapsed_seconds"), ("parent_wall_seconds", "parent_wall_seconds")):
        _require(_finite(declared.get(label)) and math.isclose(declared[label], sum(row[key] for row in known), abs_tol=1e-6))
    jobs = costs["GPU_jobs"]
    _require(type(jobs) is list and len(jobs) == 3)
    for index, item in enumerate(jobs):
        identity, release = item.get("job"), item.get("release")
        _require(type(identity) is dict and item.get("unknown_cost") is False and public._same(identity,
            queue4.make_job(index, identity.get("owner"), PREDECESSOR_PLAN_SHA256)))
        _require(type(release) is dict and release.get("verified") is True and release.get("runtime_overrun") is False
                 and public._same(release.get("GPU_seconds"), proof["GPU_jobs"][index]["GPU_lease_seconds"])
                 and public._same(release.get("service_wall_seconds"), proof["GPU_jobs"][index]["service_wall_seconds"]))
    _require(budgets5._read_predecessor_result(old) == raw)


def _cpu_proof(root, binding, *, private=False):
    _require(type(binding) is dict and set(binding) == {"path", "sha256"}
             and type(binding["path"]) is str and binding["path"].startswith("artifacts/")
             and public._sha(binding["sha256"]))
    raw = public.public_read(root, binding["path"])
    _require(_hash(raw) == binding["sha256"])
    proof = public._json(raw)
    _require(type(proof) is dict and set(proof) == {"id", "status", "scope", "source_sha256", "private_result",
        "checks", "parent_profile", "worker_profile"} and proof["id"] == "native-metadata-startup-root-CPU-006"
        and proof["status"] == "passed" and proof["scope"] == CPU_SCOPE
        and type(proof["source_sha256"]) is dict and set(proof["source_sha256"]) == CPU_SOURCES
        and type(proof["checks"]) is dict and set(proof["checks"]) == CPU_CHECKS
        and all(value is True for value in proof["checks"].values())
        and public._same(proof["parent_profile"], PARENT_PROFILE) and public._same(proof["worker_profile"], WORKER_PROFILE)
        and type(proof["private_result"]) is dict and set(proof["private_result"]) == {"path", "sha256"}
        and proof["private_result"]["path"] == CPU_RESULT and public._sha(proof["private_result"]["sha256"]))
    for name, sha in proof["source_sha256"].items():
        _require(public._sha(sha) and _hash(public.public_read(root, name)) == sha)
    if private:
        first = _read_cpu_result()
        _require(_hash(first) == proof["private_result"]["sha256"])
        result = public._json(first)
        _require(type(result) is dict and set(result) == {"id", "status", "private", "scope", "source_sha256", "checks",
            "parent_profile", "worker_profile", "metadata_timeout_seconds", "metadata_elapsed_seconds", "receipt_sha256",
            "snapshot_file_count", "nonpolicy", "returned_metadata_sha256", "model_generations", "actions", "API", "scorer", "GPU"}
            and result["id"] == "native-metadata-startup-CPU-006-001-final" and result["status"] == "CPU_metadata_saved_closed"
            and result["private"] is True and result["nonpolicy"] is True
            and all(public._same(result[key], proof[key]) for key in (
                "scope", "source_sha256", "checks", "parent_profile", "worker_profile"))
            and _finite(result["metadata_timeout_seconds"]) and 60 < result["metadata_timeout_seconds"] <= 120
            and _finite(result["metadata_elapsed_seconds"]) and public._sha(result["receipt_sha256"])
            and public._sha(result["returned_metadata_sha256"]) and type(result["snapshot_file_count"]) is int
            and result["snapshot_file_count"] == 13 and all(type(result[key]) is int and result[key] == 0
                for key in ("model_generations", "actions", "API", "scorer", "GPU")))
        _require(_read_cpu_result() == first)
    _require(public.public_read(root, binding["path"]) == raw)
    return proof


def _read_cpu_result():
    from scripts import native_source_episode_004 as episode

    reader = episode._isolated()
    reader.PRIVATE_RUN_ROOT = Path(CPU_RESULT).parent
    return reader._read_private(CPU_RESULT, maximum=4_000_000)


def _predecessor(root, binding, *, private=False):
    _require(type(binding) is dict and public._same(binding, {"plan": PREDECESSOR_PLAN,
        "plan_sha256": PREDECESSOR_PLAN_SHA256, "terminal": PREDECESSOR_TERMINAL,
        "terminal_sha256": PREDECESSOR_TERMINAL_SHA256}) and type(private) is bool)
    old = previous.load_registration(root, binding["plan_sha256"], private=private)
    _require(old["source_sha256"].get("scripts/native_source_registration_004.py") == budgets5.PREDECESSOR_SOURCE_SHA256
             and public._same(old["task_ids"], list(TASK_IDS)))
    raw = public.public_read(root, binding["terminal"])
    _require(_hash(raw) == binding["terminal_sha256"])
    proof = public._json(raw)
    _terminal(proof, old)
    if private:
        _private_terminal(old, proof)
        core = previous._get_core()
        core._private_directory(PRIVATE_RUN_ROOT)
        _require(core.shutil.disk_usage(PRIVATE_RUN_ROOT).free >= core.MINIMUM_FREE_BYTES)
    return old, proof


def _design(old, binding, cpu_binding):
    _require(_hash(public.public_read(ROOT, "scripts/native_source_registration_005.py")) == V5_PINS[
        "scripts/native_source_registration_005.py"])
    design = budgets5._design(old, binding, 20)
    design["resource_policy"]["private_run_root"] = PRIVATE_RUN_ROOT
    revision = design["execution_revision"]
    revision.pop("complete_known_failed_predecessor_gate_required", None)
    revision.update(revision=6, predecessor_status="unknown", predecessor_unknown_first_attempts=1,
        predecessor_unstarted_first_attempts=1, predecessor_policy_episodes=21, predecessor_known_policy_episodes=20,
        predecessor_attempted_policy_episodes=21,
        predecessor_unknown_costs_preserved=True, source005_unregistered_and_unrun=True,
        metadata_request_uses_full_remaining_setup=True, metadata_request_max_seconds=120,
        rpc_timeout_default_seconds=60, explicit_dispatch_and_replay_timeout="existing_phase_and_worker_deadlines",
        metadata_rpc_timeout="full_remaining_setup_120", save_close_timeout="existing_min_default60_remaining",
        worker_CLI_and_scorer_unchanged=True,
        CPU_probe=copy.deepcopy(cpu_binding))
    design["secondary_analysis"] = secondary_design()
    return design


def _bindings(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256, cpu_proof_path, cpu_proof_sha256):
    return ({"plan": PREDECESSOR_PLAN, "plan_sha256": predecessor_plan_sha256,
        "terminal": predecessor_terminal_path, "terminal_sha256": predecessor_terminal_sha256},
        {"path": cpu_proof_path, "sha256": cpu_proof_sha256})


def expected_design(root, *, predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256,
                    cpu_proof_path, cpu_proof_sha256):
    try:
        binding, cpu = _bindings(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256,
                                 cpu_proof_path, cpu_proof_sha256)
        old, _ = _predecessor(root, binding)
        _cpu_proof(root, cpu)
        return _design(old, binding, cpu)
    except Exception:
        raise ValueError(_ERROR) from None


def _expected(root, binding, cpu_binding, source_sha256, input_sha256, runtime_sha256, context_proof, *, private):
    old, _ = _predecessor(root, binding, private=private)
    cpu = _cpu_proof(root, cpu_binding, private=private)
    sources, inputs, runtime = copy.deepcopy((source_sha256, input_sha256, runtime_sha256))
    for values in (sources, inputs):
        public._binding_map(values)
    public._binding_map(runtime, runtime=True)
    previous._preserve_maps(old, {"source_sha256": sources, "input_sha256": inputs, "runtime_sha256": runtime})
    _require(REQUIRED_NEW_SOURCES <= sources.keys() and all(sources.get(key) == value for key, value in V5_PINS.items())
        and all(sources.get(key) == value for key, value in cpu["source_sha256"].items())
        and public._same(runtime, old["runtime_sha256"]) and public._same(context_proof, old["context_proof"])
        and inputs.get(PREDECESSOR_PLAN) == binding["plan_sha256"]
        and inputs.get(binding["terminal"]) == binding["terminal_sha256"]
        and inputs.get(cpu_binding["path"]) == cpu_binding["sha256"] and budgets5.PLAN not in inputs)
    for name in sources.keys() & inputs.keys():
        _require(sources[name] == inputs[name])
    for name, sha in {**sources, **inputs}.items():
        _require(_hash(public.public_read(root, name)) == sha)
    plan = copy.deepcopy(old)
    plan.update(id=ID, status="registered", design=_design(old, binding, cpu_binding),
        batch_task_ids=[list(batch) for batch in BATCH_TASK_IDS], private_run_root=PRIVATE_RUN_ROOT,
        gpu_seconds=BATCH_GPU_SECONDS, total_gpu_seconds=TOTAL_GPU_SECONDS,
        source_sha256=sources, input_sha256=inputs, runtime_sha256=runtime, context_proof=copy.deepcopy(context_proof),
        predecessor=copy.deepcopy(binding), CPU_probe=copy.deepcopy(cpu_binding))
    _require(public._same(previous.load_registration(root, binding["plan_sha256"], private=False), old))
    return plan


def register(root, *, predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256,
             cpu_proof_path, cpu_proof_sha256, source_sha256, input_sha256, runtime_sha256, context_proof, private=False):
    try:
        binding, cpu = _bindings(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256,
                                 cpu_proof_path, cpu_proof_sha256)
        return _expected(root, binding, cpu, source_sha256, input_sha256, runtime_sha256, context_proof, private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def validate_registration(root, plan, *, private=False):
    try:
        public._values(plan)
        _require(type(plan) is dict)
        expected = _expected(root, plan["predecessor"], plan["CPU_probe"], plan["source_sha256"], plan["input_sha256"],
                             plan["runtime_sha256"], plan["context_proof"], private=private)
        _require(public._same(plan, expected))
        return copy.deepcopy(plan)
    except Exception:
        raise ValueError(_ERROR) from None


def load_registration(root, sha, *, private=False):
    try:
        _require(public._sha(sha))
        raw = public.public_read(root, PLAN)
        _require(_hash(raw) == sha)
        return validate_registration(root, public._json(raw), private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def check_predecessor(root, validated_plan):
    """Enqueue-only terminal/probe metadata check; no repeated asset rehash."""
    try:
        _require(type(validated_plan) is dict and validated_plan.get("id") == ID)
        old, proof = _predecessor(root, validated_plan["predecessor"], private=False)
        _private_terminal(old, proof)
        _cpu_proof(root, validated_plan["CPU_probe"], private=True)
    except Exception:
        raise ValueError(_ERROR) from None


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(previous, name)
