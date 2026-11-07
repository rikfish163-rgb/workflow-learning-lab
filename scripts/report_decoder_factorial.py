"""Audit the registered 2x2 initialization/decoding diagnostic, without trace text."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from wll.cli import read_episodes
from wll.diagnostics import diagnose
from wll.protocol import paired_estimate
from wll.storage import atomic_json

ARMS = {"base_sampled": ("base", .7), "sft_sampled": ("sft", .7),
        "base_greedy": ("base", 0), "sft_greedy": ("sft", 0)}
MODELS = {"base": "qwen3-4b-2507-local", "sft": "qwen3-4b-teacher-init-001"}


def add_cost(cost, step):
    for key in ("prompt_tokens", "completion_tokens"):
        value = step[key]
        if type(value) is not int or value < 0:
            raise ValueError("token cost must be a nonnegative integer")
        cost[key] += value
    cost["steps"] += 1


def verify_journal(plan, episodes, journal):
    grouped = {}
    cost = {"prompt_tokens": 0, "completion_tokens": 0, "steps": 0}
    for row in journal:
        if row["task_id"] not in plan["task_ids"]:
            raise ValueError("unknown journal task")
        rows = grouped.setdefault(row["task_id"], [])
        if (type(row["index"]) is not int or row["index"] != len(rows)
                or row["index"] >= plan["registered_steps_per_episode"]
                or row["replayed"] is not False):
            raise ValueError("invalid journal index or replay flag")
        rows.append(row["step"])
        add_cost(cost, row["step"])
    for episode in episodes:
        if grouped.get(episode.task.task_id, []) != [asdict(s) for s in episode.steps]:
            raise ValueError("completed episode and journal differ")
    return cost


def compare_arms(plan, arms, *, allowed_gpu_policies=()):
    task_ids = plan["task_ids"]
    if set(arms) != set(ARMS) or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("unexpected arms or duplicate planned tasks")
    base = {e.task.task_id: e for e in arms["base_sampled"]}
    if len(base) != len(arms["base_sampled"]) or set(base) != set(task_ids):
        raise ValueError("complete unique source baseline is required")
    base_backend = base[task_ids[0]].metadata["backend"]
    if "adapter_files" in base_backend:
        raise ValueError("source baseline already has an adapter")
    indexed, metrics, adapter_files = {}, {}, None
    for name, (weight, temperature) in ARMS.items():
        entries = {}
        cost = {"prompt_tokens": 0, "completion_tokens": 0, "steps": 0}
        for episode in arms[name]:
            task_id = episode.task.task_id
            expected = {"max_steps": plan["registered_steps_per_episode"],
                        "max_tokens": plan["generation"]["max_tokens"],
                        "temperature": temperature, "prompt_version": plan["prompt_version"],
                        "environment_id": plan["image"]}
            if (task_id not in base or task_id in entries or episode.task != base[task_id].task
                    or episode.task.split != "train" or type(episode.success) is not bool
                    or episode.seed != plan["seed"] or episode.model_id != MODELS[weight]
                    or episode.branch != "first" or episode.replayed_steps or episode.experience_id
                    or len(episode.steps) > plan["registered_steps_per_episode"]
                    or any(episode.metadata.get(k) != v for k, v in expected.items())):
                raise ValueError("episode violates the registered comparison")
            backend = episode.metadata.get("backend", {})
            comparable_backend = {k: v for k, v in backend.items() if k != "adapter_files"}
            if backend.get("gpu_policy") in allowed_gpu_policies:
                comparable_backend["gpu_policy"] = base_backend["gpu_policy"]
            if comparable_backend != base_backend:
                raise ValueError("base backend differs")
            if weight == "base":
                if "adapter_files" in backend:
                    raise ValueError("base arm must not load an adapter")
            else:
                files = backend.get("adapter_files", {})
                if files.get("adapter_model.safetensors") != plan["adapter_weights_sha256"]:
                    raise ValueError("adapter weights differ")
                if adapter_files is not None and adapter_files != files:
                    raise ValueError("adapter metadata differs between trained arms")
                adapter_files = files
            for step in episode.steps:
                add_cost(cost, asdict(step))
            entries[task_id] = episode
        indexed[name] = entries
        successes = sum(e.success for e in entries.values())
        metrics[name] = {"scheduled": len(task_ids), "completed": len(entries),
                         "successes": successes,
                         "success_rate": successes / len(task_ids) if len(entries) == len(task_ids) else None,
                         "completed_trajectory_cost": cost, "diagnostics": diagnose(arms[name])}
    ready = all(len(entries) == len(task_ids) for entries in indexed.values())
    outcomes = [{"task_id": task_id, **{name: entries[task_id].success if task_id in entries else None
                                       for name, entries in indexed.items()}} for task_id in task_ids]
    paired = decision = None
    if ready:
        comparisons = {"sft_effect_at_sampled": ("base_sampled", "sft_sampled"),
                       "sft_effect_at_greedy": ("base_greedy", "sft_greedy"),
                       "decoding_effect_on_base": ("base_sampled", "base_greedy"),
                       "decoding_effect_on_sft": ("sft_sampled", "sft_greedy")}
        paired = {label: paired_estimate(arms[a], arms[b]) for label, (a, b) in comparisons.items()}
        qualifying = [name for name in ("base_greedy", "sft_greedy")
                      if len({e.task.scenario_id for e in arms[name] if e.success}) >= 2]
        decision = {"qualifying_configurations_for_separate_origin_study": qualifying,
                    "stop_decoding_and_optimization_variations_on_current_demonstrations": not qualifying}
    return {"comparison_ready": ready, "outcomes": outcomes, "by_arm": metrics,
            "paired": paired, "decision": decision,
            "interpretation": "Exploratory same-family train calibration. Missing outcomes are unknown. "
                              "No held-out scenario, method efficacy, equivalence or RSI claim. "
                              "A decoding gain shared by base and trained models is not an SFT gain."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-plan", help="Frozen infrastructure-only continuation amendment")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    evidence = {}

    def read_json(path):
        raw = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def verify(directory, entries):
        for name, digest in entries.items():
            if Path(name).name != name:
                raise ValueError("invalid evidence filename")
            path = directory / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError("evidence changed: " + str(path))
            evidence[str(path)] = actual

    def rows(path):
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    plan_path = root / "decoder-factorial-plan-v1.json"
    plan = read_json(plan_path)
    calibration = root / "teacher-warmstart-calibration-completion-001"
    original = root / "teacher-warmstart-calibration-001"
    activation = read_json(calibration / "report.json")
    verify(original, activation["original_files"])
    verify(calibration, activation["files"])
    baseline = root / "qwen3-gpu9-origin-v1-shard-2"
    verify(baseline, {"episodes.jsonl": activation["baseline_sha256"]})
    def select(episodes):
        return [e for e in episodes if e.task.task_id in plan["task_ids"]]

    arms = {"base_sampled": select(read_episodes(baseline / "episodes.jsonl")),
            "sft_sampled": read_episodes(original / "episodes.jsonl") +
                           read_episodes(calibration / "episodes.jsonl")}
    baseline_journal = baseline / "steps.jsonl"
    evidence[str(baseline_journal)] = hashlib.sha256(baseline_journal.read_bytes()).hexdigest()
    recorded = {"base_sampled": verify_journal(plan, arms["base_sampled"],
                [row for row in rows(baseline_journal) if row["task_id"] in plan["task_ids"]])}
    allowed_gpu_policies = []
    for weight in ("base", "sft"):
        directory = root / f"decoder-{weight}-greedy-001"
        if weight == "sft" and args.resume_plan:
            resume = read_json(Path(args.resume_plan))
            if resume["decoder_plan_sha256"] != evidence[str(plan_path)]:
                raise ValueError("continuation plan refers to another diagnostic")
            verify(directory, resume["original_files"])
            allowed_gpu_policies = resume["allowed_gpu_policies"]
            episodes = read_episodes(directory / "episodes.jsonl")
            if [e.task.task_id for e in episodes] != resume["preserved_completed_ids"]:
                raise ValueError("original completed tasks changed")
            cost = verify_journal(plan, episodes, rows(directory / "steps.jsonl"))
            completion = Path(resume["completion_output"])
            if completion.exists():
                added = read_episodes(completion / "episodes.jsonl")
                if {e.task.task_id for e in added} - set(resume["pending_ids"]):
                    raise ValueError("continuation re-ran a completed task")
                added_cost = verify_journal(plan, added, rows(completion / "steps.jsonl"))
                for key in cost:
                    cost[key] += added_cost[key]
                episodes += added
                for name in ("episodes.jsonl", "steps.jsonl", "manifest.json", "summary.json"):
                    path = completion / name
                    if path.exists():
                        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                if len(added) == len(resume["pending_ids"]):
                    summary = read_json(completion / "summary.json")
                    if not summary["valid_for_comparison"]:
                        raise ValueError("continuation summary is invalid")
                    observed = summary["observed_cost"]
                    if added_cost != {"steps": observed["observed_steps"],
                                      "prompt_tokens": observed["prompt_tokens"],
                                      "completion_tokens": observed["completion_tokens"]}:
                        raise ValueError("continuation cost differs from journal")
            arms["sft_greedy"] = episodes
            recorded["sft_greedy"] = cost
            continue
        report = read_json(directory / "report.json")
        if (report["complete"] is not True or report["arm"] != weight or report["temperature"] != 0
                or report["plan_sha256"] != evidence[str(plan_path)]):
            raise ValueError("arm report does not match the frozen diagnostic")
        verify(directory, report["files"])
        episodes = read_episodes(directory / "episodes.jsonl")
        cost = verify_journal(plan, episodes, rows(directory / "steps.jsonl"))
        if cost != {"steps": report["cost"]["observed_steps"],
                    "prompt_tokens": report["cost"]["prompt_tokens"],
                    "completion_tokens": report["cost"]["completion_tokens"]}:
            raise ValueError("arm cost differs from journal")
        arms[weight + "_greedy"] = episodes
        recorded[weight + "_greedy"] = cost
    interrupted_cost = {"prompt_tokens": 0, "completion_tokens": 0, "steps": 0}
    for directory in (original, calibration):
        cost = verify_journal(plan, read_episodes(directory / "episodes.jsonl"), rows(directory / "steps.jsonl"))
        for key in cost:
            interrupted_cost[key] += cost[key]
    if interrupted_cost != {"steps": activation["cost_including_interrupted_attempts"]["observed_steps"],
                            **{key: activation["cost_including_interrupted_attempts"][key]
                               for key in ("prompt_tokens", "completion_tokens")}}:
        raise ValueError("calibration interruption costs differ")
    recorded["sft_sampled"] = interrupted_cost
    result = compare_arms(plan, arms, allowed_gpu_policies=allowed_gpu_policies)
    if not result["comparison_ready"] and not args.allow_incomplete:
        raise ValueError("final factorial report requires all registered outcomes")
    result.update(observed_run_cost=recorded, teacher_generation_tokens=None,
                  allowed_resource_changes=allowed_gpu_policies,
                  training_metrics=read_json(root / "teacher-warmstart-sft-001/metrics.json"),
                  input_sha256=evidence,
                  cost_note="Run costs include original interrupted sampled and greedy SFT attempts; completed trajectory "
                            "costs exclude that duplicate work. In-flight missing responses have unknown "
                            "additional cost. Base sampled trajectories reuse a subset of an earlier run. "
                            "Teacher generation is unmetered, not free. No matched wall-time claim.")
    atomic_json(output, result)


if __name__ == "__main__":
    main()
