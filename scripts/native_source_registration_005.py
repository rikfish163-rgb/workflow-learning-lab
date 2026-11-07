"""Conditional, independent 24-action native source registration.

Derive only declared budget/namespace changes from an authenticated source004
plan. Public admission requires an exact hash-bound complete-known failed-gate
projection. Private preflight additionally hashes and compares source004's real
terminal metadata and all three verified, non-overrun GPU release records.
No actor trace, scorer body, task constructor, GT loader, model or queue is used.
Builders return values only; callers own exclusive plan/proof writes.
"""

import copy
import hashlib
import math
from pathlib import Path

from scripts import native_source_registration as public
from scripts import native_source_registration_004 as previous

ROOT = Path(__file__).resolve().parents[1]
ID = "qwen38-native-source-pilot-005"
PLAN = "artifacts/qwen38-native-source-pilot-plan-005.json"
PRIVATE_RUN_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-005"
PREDECESSOR_PLAN = previous.PLAN
PREDECESSOR_SOURCE_SHA256 = "64053cff2c4c1d4ee5f9500b9a7848914930ab4590769d822b0a65cbf316e4eb"
TASK_IDS = previous.TASK_IDS
BATCH_TASK_IDS = tuple((task,) for task in TASK_IDS)
BATCH_GPU_SECONDS = 7200
TOTAL_GPU_SECONDS = 86400
MODEL_READY_SECONDS = 600
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
WAIT_SECONDS = previous.WAIT_SECONDS
PORT, PYTHON = previous.PORT, previous.PYTHON
SERVICE_PLAN, SERVICE_SHA = previous.SERVICE_PLAN, previous.SERVICE_SHA
MODEL_ARM, MODEL_ID = previous.MODEL_ARM, previous.MODEL_ID
LAUNCH_PROOF = "artifacts/native-source-launch-root-CPU-005.json"
REQUIRED_NEW_SOURCES = frozenset(
    f"{directory}/{prefix}{name}_005.py"
    for name in ("native_source_registration", "native_source_gpu_queue",
                 "native_source_episode", "run_native_source_pilot")
    for directory, prefix in (("scripts", ""), ("tests", "test_"))
)
REQUIRED_SOURCES = previous.REQUIRED_SOURCES | REQUIRED_NEW_SOURCES
_SUFFIX = {"max_generations": 24, "max_action_attempts_including_replay": 84,
           "max_API_invocations_including_replay": 84, "max_prompt_tokens": 368640,
           "max_completion_tokens": 24576}
_KNOWN = {"status", "model_reported_complete", "official_success", "receipt_sha256"}
_PROOF_FIELDS = {"source_id", "source_plan_sha256", "source_result_sha256", "status",
                 "slots", "candidate_student_target_sha256", "data_feasibility_gate", "all_resources_released"}
_ERROR = "native_source_registration_005_invalid"


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _terminal(proof, old, plan_sha):
    _require(type(proof) is dict and set(proof) == _PROOF_FIELDS
             and proof["source_id"] == previous.ID and proof["source_plan_sha256"] == plan_sha
             and public._sha(proof["source_result_sha256"]) and proof["status"] == "complete"
             and proof["all_resources_released"] is True)
    slots = proof["slots"]
    _require(type(slots) is list and len(slots) == 12)
    helped, episodes = set(), 0
    for index, (slot, task) in enumerate(zip(slots, old["task_ids"], strict=True)):
        _require(type(slot) is dict and set(slot) == {
            "source_index", "task_id", "pair_class", "first_attempt", "control", "review"}
            and type(slot["source_index"]) is int and slot["source_index"] == index and slot["task_id"] == task)
        for phase in ("first_attempt", "control", "review"):
            value = slot[phase]
            _require(type(value) is dict and type(value.get("status")) is str)
            if value["status"] == "known":
                _require(set(value) == _KNOWN and type(value["model_reported_complete"]) is bool
                         and type(value["official_success"]) is bool and public._sha(value["receipt_sha256"]))
                episodes += 1
            else:
                _require(phase != "first_attempt" and set(value) == {"status", "reason"}
                         and value["status"] == "known_ineligible"
                         and value["reason"] in {"no_eligible_first_4xx_prefix", "paired_initial_context_does_not_fit"})
        control, review = slot["control"], slot["review"]
        if control["status"] == "known_ineligible" or review["status"] == "known_ineligible":
            _require(public._same(control, review))
            expected = {"no_eligible_first_4xx_prefix": "ineligible_no_boundary",
                        "paired_initial_context_does_not_fit": "ineligible_initial_context"}[control["reason"]]
        elif not (control["model_reported_complete"] and review["model_reported_complete"]):
            expected = "ineligible_incomplete_pair"
        else:
            expected = {(False, True): "help", (True, False): "harm",
                        (True, True): "tie_success", (False, False): "tie_failure"}[
                            (control["official_success"], review["official_success"])]
        _require(slot["pair_class"] == expected)
        if expected == "help":
            helped.add(task.rsplit("_", 1)[0])
    targets = proof["candidate_student_target_sha256"]
    _require(type(targets) is list and all(public._sha(item) for item in targets)
             and targets == sorted(set(targets)))
    gate = {"status": "insufficient", "complete_help_scenarios": len(helped),
            "distinct_student_targets": len(targets), "source_denominator": 12}
    _require(public._same(proof["data_feasibility_gate"], gate)
             and len(targets) <= len(helped) * old["design"]["budgets"]["suffix"]["max_generations"]
             and not (len(helped) >= 4 and len(targets) >= 20))
    return episodes


def _read_predecessor_result(old):
    from scripts import native_source_episode_004 as episode

    return episode._read_private(Path(old["private_run_root"]) / "study-result.json", maximum=64_000_000)


def _private_terminal(old, proof):
    raw = _read_predecessor_result(old)
    _require(type(raw) is bytes and _hash(raw) == proof["source_result_sha256"])
    report = public._json(raw)
    _require(type(report) is dict and report.get("id") == previous.ID
             and report.get("plan_sha256") == proof["source_plan_sha256"]
             and report.get("status") == "complete")
    for key in ("slots", "candidate_student_target_sha256", "data_feasibility_gate"):
        _require(public._same(report.get(key), proof[key]))
    _known_cost_metadata(report, old)
    jobs = report.get("costs", {}).get("GPU_jobs")
    _require(type(jobs) is list and len(jobs) == 3)
    total = 0
    from scripts import native_source_gpu_queue_004 as queue4

    for index, job in enumerate(jobs):
        _require(type(job) is dict and job.get("unknown_cost") is False)
        identity = job.get("job")
        _require(type(identity) is dict and public._same(identity,
            queue4.make_job(index, identity.get("owner"), proof["source_plan_sha256"])))
        release = job.get("release")
        _require(type(release) is dict and release.get("verified") is True
                 and release.get("runtime_overrun") is False)
        duration = release.get("GPU_seconds")
        _require(type(duration) in (int, float) and math.isfinite(duration)
                 and 0 <= duration <= BATCH_GPU_SECONDS
                 and type(release.get("service_wall_seconds")) in (int, float)
                 and release["service_wall_seconds"] == duration)
        total += duration
    _require(total <= 21600 and _read_predecessor_result(old) == raw)


def _known_cost_metadata(report, old):
    """Reconcile only existing terminal counters; never open execution payloads."""
    expected = {(index, phase) for index, slot in enumerate(report["slots"])
                for phase in ("first_attempt", "control", "review") if slot[phase]["status"] == "known"}

    def phase_rows(name):
        values = report["costs"][name]
        _require(type(values) is list)
        found = {}
        for row in values:
            _require(type(row) is dict and type(row.get("source_index")) is int
                     and type(row.get("phase")) is str)
            key = row["source_index"], row["phase"]
            _require(key in expected and key not in found and row.get("unknown_cost") is False)
            found[key] = row
        _require(found.keys() == expected)
        return found

    episodes, scorers = phase_rows("episodes"), phase_rows("scorers")
    integer_fields = ("generations", "prompt_tokens", "completion_tokens", "action_attempts",
        "api_invocations", "upstream_requests", "generation_intents", "generation_results",
        "confirmed_not_started", "generation_attempts_after_cancellations", "dispatch_intents",
        "dispatch_results", "unreturned_generations", "unreturned_generation_intents_upper_bound",
        "journal_returned_prompt_tokens_lower_bound", "journal_returned_completion_tokens_lower_bound")
    help_generation_bound = 0
    for (index, phase), row in episodes.items():
        _require(row.get("counts_complete") is True and row.get("generation_accounting_valid") is True
                 and all(type(row.get(key)) is int and row[key] >= 0 for key in integer_fields)
                 and row.get("CPU_counting_setup_save_cleanup_included_in_parent_wall") is True
                 and all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                         for key in ("parent_wall_seconds", "controller_elapsed_seconds")))
        intents, returned, cancelled = row["generation_intents"], row["generation_results"], row["confirmed_not_started"]
        _require(intents - cancelled == row["generation_attempts_after_cancellations"] == row["generations"] == returned
                 and row["unreturned_generation_intents_upper_bound"] == intents - returned - cancelled == 0
                 and row["unreturned_generations"] == 0
                 and row["prompt_tokens"] == row["journal_returned_prompt_tokens_lower_bound"]
                 and row["completion_tokens"] == row["journal_returned_completion_tokens_lower_bound"]
                 and row["dispatch_results"] <= row["dispatch_intents"] <= returned
                 and row["dispatch_intents"] - row["dispatch_results"] <= 1
                 and row["dispatch_results"] <= row["action_attempts"] <= 60
                 and row["api_invocations"] <= row["action_attempts"] and row["upstream_requests"] <= 60)
        budget = old["design"]["budgets"]["source" if phase == "first_attempt" else "suffix"]
        _require(returned <= budget["max_generations"] and cancelled <= 1
                 and returned <= row["prompt_tokens"] <= returned * 15360
                 and returned <= row["completion_tokens"] <= returned * 1024
                 and row["prompt_tokens"] <= budget["max_prompt_tokens"]
                 and row["completion_tokens"] <= budget["max_completion_tokens"])
        if report["slots"][index][phase]["model_reported_complete"]:
            _require(returned > 0 and row["dispatch_results"] > 0 and cancelled == 0)
        if phase == "review" and report["slots"][index]["pair_class"] == "help":
            help_generation_bound += returned
        score = scorers[index, phase]
        _require(all(score.get(key) is True for key in ("attempted", "result_validated", "cleanup_verified")))
        _require(type(score.get("wall_seconds")) in (int, float)
                 and math.isfinite(score["wall_seconds"]) and score["wall_seconds"] >= 0)
    _require(len(report["candidate_student_target_sha256"]) <= help_generation_bound)
    expected_counts = {index for index, slot in enumerate(report["slots"])
                       if slot["pair_class"] != "ineligible_no_boundary"}
    counts = report["costs"]["paired_CPU_counts"]
    _require(type(counts) is list and len(counts) == len(expected_counts))
    found = set()
    for row in counts:
        _require(type(row) is dict and type(row.get("source_index")) is int
                 and row["source_index"] in expected_counts and row["source_index"] not in found
                 and row.get("counts_complete") is True and row.get("unknown_cost") is False
                 and type(row.get("CPU_calls")) is int and row["CPU_calls"] == 2
                 and type(row.get("wall_seconds")) in (int, float)
                 and math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0)
        found.add(row["source_index"])
        contexts = row.get("counts")
        _require(type(contexts) is dict and set(contexts) == {"control", "review"})
        for context in contexts.values():
            _require(type(context) is dict and set(context) == {"metadata_sha256", "messages_sha256", "input_tokens"}
                     and public._sha(context["metadata_sha256"]) and public._sha(context["messages_sha256"])
                     and type(context["input_tokens"]) is int and context["input_tokens"] > 0)
        _require(contexts["control"]["metadata_sha256"] == contexts["review"]["metadata_sha256"])
        overflow = any(value["input_tokens"] > 15360 for value in contexts.values())
        _require(overflow == (report["slots"][row["source_index"]]["pair_class"] == "ineligible_initial_context"))


def _predecessor(root, binding, *, private=False):
    _require(type(private) is bool and type(binding) is dict and set(binding) == {
        "plan", "plan_sha256", "terminal", "terminal_sha256"}
        and binding["plan"] == PREDECESSOR_PLAN and public._sha(binding["plan_sha256"])
        and public._sha(binding["terminal_sha256"]) and type(binding["terminal"]) is str
        and binding["terminal"].startswith("artifacts/") and binding["terminal"] not in {PLAN, PREDECESSOR_PLAN})
    old = previous.load_registration(root, binding["plan_sha256"], private=False)
    _require(old["source_sha256"].get("scripts/native_source_registration_004.py") == PREDECESSOR_SOURCE_SHA256
             and public._same(old["task_ids"], list(TASK_IDS)))
    raw = public.public_read(root, binding["terminal"])
    _require(_hash(raw) == binding["terminal_sha256"])
    proof = public._json(raw)
    episodes = _terminal(proof, old, binding["plan_sha256"])
    if private:
        _require(public._same(previous.load_registration(root, binding["plan_sha256"], private=True), old))
        _private_terminal(old, proof)
        core = previous._get_core()
        core._private_directory(PRIVATE_RUN_ROOT)
        _require(core.shutil.disk_usage(PRIVATE_RUN_ROOT).free >= core.MINIMUM_FREE_BYTES)
    return old, proof, episodes


def _design(old, binding, episodes):
    design = copy.deepcopy(old["design"])
    design["schedule"]["batch_task_ids"] = [list(batch) for batch in BATCH_TASK_IDS]
    budgets = design["budgets"]
    budgets["suffix"] = copy.deepcopy(_SUFFIX)
    budgets.update(suffix_actor_generation_seconds=1200, suffix_worker_wall_seconds=1380,
                   suffix_parent_unit_seconds=1500, model_ready_seconds=MODEL_READY_SECONDS,
                   total_GPU_seconds=TOTAL_GPU_SECONDS, serial_GPU_jobs=12)
    design["resource_policy"]["private_run_root"] = PRIVATE_RUN_ROOT
    revision = design["execution_revision"]
    revision.update(revision=5, independent_study=True, automatic_resume=False, outcomes_imported=False,
        predecessor_id=previous.ID, predecessor_policy_episodes=episodes,
        predecessor_unknown_first_attempts=0, predecessor_unstarted_first_attempts=0,
        predecessor_scorer_calls=episodes, predecessor_qualified_experiences=0,
        predecessor_plan=binding["plan"], predecessor_plan_sha256=binding["plan_sha256"],
        predecessor_terminal=binding["terminal"], predecessor_terminal_sha256=binding["terminal_sha256"],
        complete_known_failed_predecessor_gate_required=True)
    return design


def _binding(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256):
    return {"plan": PREDECESSOR_PLAN, "plan_sha256": predecessor_plan_sha256,
            "terminal": predecessor_terminal_path, "terminal_sha256": predecessor_terminal_sha256}


def expected_design(root, *, predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256):
    binding = _binding(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256)
    try:
        old, _, episodes = _predecessor(root, binding)
        return _design(old, binding, episodes)
    except Exception:
        raise ValueError(_ERROR) from None


def _expected(root, binding, source_sha256, input_sha256, runtime_sha256, context_proof, *, private):
    old, _, episodes = _predecessor(root, binding, private=private)
    sources, inputs, runtime = copy.deepcopy((source_sha256, input_sha256, runtime_sha256))
    for values in (sources, inputs):
        public._binding_map(values)
    public._binding_map(runtime, runtime=True)
    current = {"source_sha256": sources, "input_sha256": inputs, "runtime_sha256": runtime}
    previous._preserve_maps(old, current)
    _require(REQUIRED_NEW_SOURCES <= sources.keys()
             and public._same(runtime, old["runtime_sha256"])
             and public._same(context_proof, old["context_proof"])
             and inputs.get(PREDECESSOR_PLAN) == binding["plan_sha256"]
             and inputs.get(binding["terminal"]) == binding["terminal_sha256"])
    for name in sources.keys() & inputs.keys():
        _require(sources[name] == inputs[name])
    for name, digest in {**sources, **inputs}.items():
        _require(_hash(public.public_read(root, name)) == digest)
    plan = copy.deepcopy(old)
    plan.update(id=ID, status="registered", design=_design(old, binding, episodes),
                batch_task_ids=[list(batch) for batch in BATCH_TASK_IDS], private_run_root=PRIVATE_RUN_ROOT,
                gpu_seconds=BATCH_GPU_SECONDS, total_gpu_seconds=TOTAL_GPU_SECONDS,
                source_sha256=sources, input_sha256=inputs, runtime_sha256=runtime,
                context_proof=copy.deepcopy(context_proof), predecessor=copy.deepcopy(binding))
    _require(_hash(public.public_read(root, binding["terminal"])) == binding["terminal_sha256"]
             and public._same(previous.load_registration(root, binding["plan_sha256"], private=False), old))
    return plan


def register(root, *, predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256,
             source_sha256, input_sha256, runtime_sha256, context_proof, private=False):
    """Return a validated value only; never write a plan or create a run directory."""
    try:
        return _expected(root, _binding(predecessor_plan_sha256, predecessor_terminal_path, predecessor_terminal_sha256),
                         source_sha256, input_sha256, runtime_sha256, context_proof, private=private)
    except Exception:
        raise ValueError(_ERROR) from None


def validate_registration(root, plan, *, private=False):
    try:
        _require(type(plan) is dict)
        public._values(plan)
        expected = _expected(root, plan["predecessor"], plan["source_sha256"], plan["input_sha256"],
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
    """Cheap enqueue gate over real 004 metadata, without repeating asset hashes.

    The queue first authenticates the new plan with public load_registration.
    Full private assets/new-directory preflight still belongs to the coordinator.
    """
    try:
        _require(type(validated_plan) is dict and validated_plan.get("id") == ID)
        old, proof, _ = _predecessor(root, validated_plan["predecessor"], private=False)
        _private_terminal(old, proof)
    except Exception:
        raise ValueError(_ERROR) from None


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(previous, name)
