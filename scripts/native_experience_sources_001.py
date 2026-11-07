"""Read-only qualification of one terminal native-004 source study.

All outputs are PRIVATE. The caller must freeze this manifest and this loader's
source in a separate writer registration before any writer call. Existing source
authorization remains unchanged. Per-file hashes here bind a consistent post-run
snapshot; the source report did not precommit the controller/episode journals.
No task constructor, ground truth, raw scorer log, model, queue or rerun is used.
"""

import hashlib
import re
from dataclasses import asdict
from pathlib import Path

from scripts import native_source_episode_004 as episode4
from scripts import native_source_registration_004 as registration4
from scripts import run_native_source_pilot_004 as pilot4
from scripts.native_appworld_worker_client import _decode, _encode
from scripts.native_experience_writer_001 import build_writer_messages
from scripts.verification_token_counter import _message_payload

_ERROR = "native_experience_sources_invalid"
_PHASES = ("first_attempt", "control", "review")
_KNOWN = {"status", "model_reported_complete", "official_success", "receipt_sha256"}
_CONTEXT = {"metadata_sha256", "messages_sha256", "input_tokens"}


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _integer(value, minimum):
    return type(value) is int and value >= minimum


def _terminal_claim(report, plan, plan_sha):
    tasks = plan["task_ids"]
    _require(type(tasks) is list and len(tasks) == len(set(tasks)) == 12
             and report.get("id") == pilot4.ID and report.get("plan_sha256") == plan_sha
             and report.get("status") == "complete" and report.get("automatic_retry") is False
             and all(type(report.get(key)) is int and report[key] == 0
                     for key in ("writer_calls", "training_jobs", "utility_or_dev_or_test_calls")))
    slots, gate = report.get("slots"), report.get("data_feasibility_gate")
    _require(type(slots) is list and len(slots) == 12 and type(gate) is dict
             and gate.get("status") == "passed" and type(gate.get("source_denominator")) is int
             and gate["source_denominator"] == 12
             and _integer(gate.get("complete_help_scenarios"), 4)
             and _integer(gate.get("distinct_student_targets"), 20))
    for index, (slot, task) in enumerate(zip(slots, tasks, strict=True)):
        _require(type(slot) is dict and type(slot.get("source_index")) is int
                 and slot["source_index"] == index and slot.get("task_id") == task)
        for phase in _PHASES:
            value = slot.get(phase)
            _require(type(value) is dict and value.get("status") in
                     ({"known"} if phase == "first_attempt" else {"known", "known_ineligible"}))
            if value["status"] == "known":
                _require(set(value) == _KNOWN and type(value["model_reported_complete"]) is bool
                         and type(value["official_success"]) is bool and _sha(value["receipt_sha256"]))
            else:
                _require(set(value) == {"status", "reason"} and type(value["reason"]) is str)
    hashes = report.get("candidate_student_target_sha256")
    _require(type(hashes) is list and len(hashes) >= 20 and all(_sha(item) for item in hashes)
             and hashes == sorted(set(hashes)))
    return slots


def load_qualified_sources(root, plan_sha256, result_sha256):
    """Recompute source qualification and return private, fully bound writer inputs.

    Only the registered source private root is read. A claimed insufficient or
    incomplete result rejects before any actor histories are mined. Passed flags
    are then independently checked against configs, receipts, score metadata,
    native actor evidence and all canonical eligible student targets.
    """
    try:
        _require(_sha(plan_sha256) and _sha(result_sha256))
        root = Path(root)
        plan = registration4.load_registration(root, plan_sha256, private=False)
        _require(plan["id"] == pilot4.ID and plan["status"] == "registered"
                 and all(plan.get(key) is False for key in (
                     "writer_authorized", "training_authorized", "utility_authorized", "dev_or_test_authorized")))
        core, episode = pilot4._get_core(), episode4._get_core()
        private = Path(plan["private_run_root"])
        _require(private == Path(episode.PRIVATE_RUN_ROOT) and private.is_absolute())
        runtime = core._Runtime(root)
        original_read, files = runtime.read, {}

        def read(path):
            path = Path(path)
            _require(path.is_absolute() and private in path.parents and ".." not in path.parts)
            raw = original_read(path)
            _require(type(raw) is bytes)
            sha = _hash(raw)
            _require(str(path) not in files or files[str(path)] == sha)
            files[str(path)] = sha
            return raw

        runtime.read = read
        raw = read(private / "study-result.json")
        _require(_hash(raw) == result_sha256)
        report = _decode(raw)
        _require(type(report) is dict)
        slots = _terminal_claim(report, plan, plan_sha256)
        intent = _decode(read(private / "study-intent.json"))
        _require(type(intent) is dict and intent.get("plan_sha256") == plan_sha256
                 and intent.get("automatic_retry") is False
                 and core.registration._same(intent.get("task_ids"), plan["task_ids"])
                 and core.registration._same(intent.get("slots"), core.initial_slots()))
        study = core._journal(read(private / "study-journal.jsonl"))
        _require(core.registration._same(core._one(study, "study_stop")["report"], report))
        jobs = report["costs"]["GPU_jobs"]
        _require(type(jobs) is list and len(jobs) == 3
                 and all(type(item) is dict and item.get("unknown_cost") is False
                         and item.get("release", {}).get("verified") is True for item in jobs))
        context_binding = plan["context_proof"]
        context_raw = registration4.public_read(root, context_binding["path"])
        _require(_hash(context_raw) == context_binding["sha256"])
        hint = plan["design"]["review"]["hint"]

        def phase_evidence(index, phase):
            config_path = private / f"config-{index:02d}-{phase}.json"
            config_raw = read(config_path)
            config = _decode(config_raw)
            _require(config["source_index"] == index and config["phase"] == phase
                     and config["plan_sha256"] == plan_sha256
                     and core.registration._same(config["job"], jobs[index // 4]["job"]))
            checked, limits, run_dir, metadata, prefix, fixed_hint = episode._configuration(
                config, root, registration4.load_registration)
            _require(core.registration._same(checked, plan))
            ready = [event for event in study if event["event"] == "GPU_ready"
                     and core.registration._same(event.get("job"), config["job"])]
            _require(len(ready) == 1 and core.registration._same(
                ready[0]["service"]["backend"], config["expected_backend"]))
            if config["prefix_binding"] is not None:
                binding = config["prefix_binding"]
                _require(binding["path"] == str(private / f"prefix-{index:02d}.json"))
                prefix_raw = read(binding["path"])
                _require(_hash(prefix_raw) == binding["sha256"]
                         and core.registration._same(_decode(prefix_raw),
                             {"metadata": metadata, "prefix_records": prefix}))
            evidence = runtime.artifacts(run_dir)
            receipt, journal = evidence["receipt"], evidence["journal"]
            slot = slots[index][phase]
            _require(evidence["receipt_binding"]["path"] == str(run_dir / "execution-receipt.json")
                     and evidence["receipt_binding"]["sha256"] == slot["receipt_sha256"]
                     and receipt.get("task_id") == plan["task_ids"][index]
                     and receipt.get("status") == "saved_and_closed"
                     and receipt.get("is_policy_episode") is True
                     and type(receipt.get("worker_exit_code")) is int and receipt["worker_exit_code"] == 0)
            owner = config["job"]["owner"]
            experiment = f"native-source-001-{index:02d}-{phase}-{owner}"
            _require(receipt.get("experiment_name") == experiment)
            events = core._journal(read(run_dir / "episode.jsonl"))
            registered = core._one(events, "registered_episode_intent")
            _require(registered["config_sha256"] == _hash(config_raw)
                     and core.registration._same(registered["config"], config)
                     and core.registration._same(core._one(events, "registered_episode_stop")["receipt"], receipt))
            controller = core._one(journal, "controller_intent")
            expected = {"task_id": plan["task_ids"][index], "experiment_name": experiment,
                        "limits": asdict(limits), "seed": 17 + len(prefix or []),
                        "is_policy_episode": True, "review_hint_present": phase == "review"}
            _require(all(core.registration._same(controller.get(key), value) for key, value in expected.items())
                     and core.registration._same(core._one(journal, "controller_stop")["result"], receipt)
                     and core.registration._same(core._one(journal, "actor_result")["outcome"], receipt["actor_outcome"]))
            verification = [event for event in events if event["event"] == "initial_context_verified"]
            if any(event["event"] == "generation_intent" for event in journal) or verification:
                _require(len(verification) == 1 and core.registration._same(verification[0], {
                    "event": "initial_context_verified",
                    "messages_sha256": config["initial_context"]["messages_sha256"],
                    "input_tokens": config["initial_context"]["input_tokens"]}))
            snapshot = receipt.get("snapshot")
            _require(type(snapshot) is dict and set(snapshot) == {
                "unit_name", "db_path", "files", "cost", "manifest_sha256"})
            bare = {key: value for key, value in snapshot.items() if key != "manifest_sha256"}
            _require(_hash(_encode(bare)) == snapshot["manifest_sha256"]
                     and core.registration._same(core._one(journal, "execution_snapshot")["snapshot"], snapshot))
            score = _decode(read(private / f"score-{index:02d}-{phase}-{owner}" / "scoring-result.json"))
            _require(type(score) is dict and set(score) == {
                "status", "success", "receipt_sha256", "manifest_sha256", "is_policy_episode"}
                and score["status"] == "scored" and type(score["success"]) is bool
                and score["is_policy_episode"] is True and score["receipt_sha256"] == slot["receipt_sha256"]
                and score["manifest_sha256"] == snapshot["manifest_sha256"]
                and score["success"] is slot["official_success"]
                and (receipt["actor_outcome"]["status"] == "model_reported_complete")
                    is slot["model_reported_complete"])
            if phase != "first_attempt":
                _require(receipt.get("replay_outcome", {}).get("status") == "matched")
            metadata, records, generations = core.actor_evidence(
                receipt, journal, expected_context=config["initial_context"],
                prefix_records=prefix, review_hint=fixed_hint)
            return metadata, records, generations, core._one(journal, "actor_start")["messages"]

        targets, helped = {}, []
        for index, slot in enumerate(slots):
            metadata, records, _, _ = phase_evidence(index, "first_attempt")
            prefix = core.select_prefix(metadata, records)
            if prefix is None:
                _require(slot["pair_class"] == "ineligible_no_boundary" and all(slot[name] == {
                    "status": "known_ineligible", "reason": "no_eligible_first_4xx_prefix"}
                    for name in ("control", "review")))
                continue
            prefix_raw = read(private / f"prefix-{index:02d}.json")
            _require(core.registration._same(_decode(prefix_raw), prefix))
            if slot["control"]["status"] == "known_ineligible":
                _require(slot["pair_class"] == "ineligible_initial_context" and all(slot[name] == {
                    "status": "known_ineligible", "reason": "paired_initial_context_does_not_fit"}
                    for name in ("control", "review")))
                counts = [row for row in report["costs"]["paired_CPU_counts"] if row["source_index"] == index]
                _require(len(counts) == 1 and counts[0].get("counts_complete") is True
                         and counts[0].get("unknown_cost") is False
                         and type(counts[0].get("CPU_calls")) is int and counts[0]["CPU_calls"] == 2)
                contexts = counts[0]["counts"]
                _require(set(contexts) == {"control", "review"})
                for name, context in contexts.items():
                    messages = core.actor.build_actor_messages(metadata, prefix_records=prefix["prefix_records"],
                        review_hint=hint if name == "review" else None)
                    _require(type(context) is dict and set(context) == _CONTEXT
                             and context["metadata_sha256"] == _hash(_encode(metadata))
                             and context["messages_sha256"] == _message_payload(messages)[1]
                             and _integer(context["input_tokens"], 1))
                _require(any(context["input_tokens"] > 15360 for context in contexts.values()))
                continue
            _require(slot["review"]["status"] == "known")
            phase_evidence(index, "control")
            _, review_records, generations, start = phase_evidence(index, "review")
            control, review = slot["control"], slot["review"]
            pair = ("ineligible_incomplete_pair" if not (
                control["model_reported_complete"] and review["model_reported_complete"]) else {
                (False, True): "help", (True, False): "harm",
                (True, True): "tie_success", (False, False): "tie_failure"
            }[(control["official_success"], review["official_success"])])
            _require(slot["pair_class"] == pair)
            if pair == "help":
                for target in core.student_targets(generations,
                        prefix_length=len(prefix["prefix_records"]), hint=hint):
                    targets.setdefault(target["target_sha256"], {"task_id": slot["task_id"], **target})
                helped.append((index, slot["task_id"].rsplit("_", 1)[0], start, review_records,
                               len(prefix["prefix_records"])))
        gate = {"status": "passed", "complete_help_scenarios": len({row[1] for row in helped}),
                "distinct_student_targets": len(targets), "source_denominator": 12}
        _require(gate["complete_help_scenarios"] >= 4 and gate["distinct_student_targets"] >= 20
                 and core.registration._same(report["data_feasibility_gate"], gate)
                 and report["candidate_student_target_sha256"] == sorted(targets)
                 and core.registration._same(_decode(read(private / "candidate-student-targets.json")), list(targets.values())))
        writer_inputs = []
        for index, scenario, start, records, prefix_length in helped:
            messages = build_writer_messages(start, records, prefix_length=prefix_length, fixed_hint=hint)
            writer_inputs.append({"source_index": index, "source_scenario": scenario,
                                  "messages": messages, "messages_sha256": _message_payload(messages)[1]})
        _require(len(writer_inputs) <= 12)
        for path, sha in files.items():
            _require(_hash(original_read(Path(path))) == sha)
        _require(registration4.public_read(root, context_binding["path"]) == context_raw
                 and core.registration._same(registration4.load_registration(root, plan_sha256, private=False), plan))
        manifest = {
            "private": True, "source_plan_sha256": plan_sha256, "source_result_sha256": result_sha256,
            "private_files_sha256": dict(sorted(files.items())),
            "public_context_binding": context_binding,
            "writer_messages_sha256": [item["messages_sha256"] for item in writer_inputs],
            "scope": "Post-run consistency bindings; source result did not precommit these journal hashes.",
        }
        return {"private": True, "gate": gate, "writer_inputs": writer_inputs,
                "input_manifest": manifest, "input_manifest_sha256": _hash(_encode(manifest))}
    except Exception:
        raise ValueError(_ERROR) from None
