import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from .appworld import APPWORLD_REV, AppWorldEnvironment, load_tasks
from .model import LocalModel
from .prompts import DEFAULT_VERSION, SYSTEMS
from .protocol import Episode, fingerprint, paired_estimate
from .runner import run_episode
from .storage import atomic_json, require_space
from .training import export_first_attempt_examples, export_validated_transfer_examples


def read_episodes(path):
    return [Episode.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines()
            if line.strip()]


def write_json(path, value):
    atomic_json(path, value)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    candidate, evidence = None, None
    if getattr(args, "experience_report", None):
        from .experience import load_validated_experience

        candidate, evidence = load_validated_experience(args.experience_report)
        if args.model != evidence["model_id"]:
            raise ValueError("experience was validated for a different model")
    id_file = getattr(args, "task_ids_file", None)
    tasks = load_tasks(args.data_dir, args.split, args.limit, args.image, args.deps_dir,
                       one_per_scenario=args.one_per_scenario,
                       include_task_ids=Path(id_file).read_text().splitlines() if id_file else None)
    if candidate and args.split == "train":
        excluded = set(candidate.source_scenarios + evidence["validation_scenarios"])
        if any(t.scenario_id in excluded for t in tasks):
            raise ValueError("validated transfer collection requires independent training scenarios")
    model = LocalModel(args.endpoint, args.model, max_tokens=args.max_tokens,
                       temperature=args.temperature)
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image,
                                        "--format", "{{.Id}}"], text=True).strip()
    manifest = {
        "config": vars(args), "tasks": [asdict(t) for t in tasks],
        "task_manifest_hash": fingerprint([asdict(t) for t in tasks]),
        "appworld_revision": APPWORLD_REV, "docker_image_id": image_id,
        "python": platform.python_version(), "status": "running",
        "purpose": "subset" if args.limit or id_file or args.one_per_scenario else "full_split",
        "code_fingerprint": fingerprint({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}),
        "experience_id": candidate.experience_id if candidate else None,
        "validation_report_hash": evidence["report_hash"] if evidence else None,
    }
    write_json(output / "manifest.json", manifest)
    completed, errors = [], []
    cost = {"prompt_tokens": 0, "completion_tokens": 0, "observed_steps": 0}
    try:
        with (output / "episodes.jsonl").open("w") as stream, \
                (output / "steps.jsonl").open("w") as journal:
            for task in tasks:
                def observe(index, step, replayed, task_id=task.task_id):
                    row = {"task_id": task_id, "index": index, "replayed": replayed,
                           "step": asdict(step)}
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    cost["observed_steps"] += 1
                    if not replayed:
                        cost["prompt_tokens"] += step.prompt_tokens
                        cost["completion_tokens"] += step.completion_tokens

                try:
                    episode = run_episode(
                        task, lambda task=task: AppWorldEnvironment(task.task_id, args.data_dir,
                            args.steps, image_id, args.deps_dir),
                        model, max_steps=args.steps, seed=args.seed, on_step=observe,
                        experience=candidate.render() if candidate else None,
                        experience_id=candidate.experience_id if candidate else None,
                        prompt_version=getattr(args, "prompt_version", DEFAULT_VERSION))
                    episode.metadata["environment_id"] = image_id
                    if evidence:
                        if episode.metadata.get("backend") != evidence["backend"]:
                            raise ValueError("episode inference backend differs from validation")
                        episode.metadata.update(validation_report_hash=evidence["report_hash"],
                                                experience_status="validated_on_independent_train_scenarios")
                    completed.append(episode)
                    stream.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps({"task_id": task.task_id, "success": episode.success,
                                      "steps": len(episode.steps), "tokens": episode.generated_tokens}),
                          flush=True)
                except Exception as exc:
                    errors.append({"task_id": task.task_id, "type": type(exc).__name__})
                    write_json(output / "errors.json", errors)
                    print(json.dumps(errors[-1]), flush=True)
        manifest["status"] = "incomplete" if errors else "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        write_json(output / "summary.json", {
            "scheduled": len(tasks), "completed": len(completed), "infrastructure_errors": len(errors),
            "successes": sum(e.success for e in completed),
            "success_rate_completed_only": sum(e.success for e in completed)/len(completed) if completed else None,
            "success_rate_all_scheduled": sum(e.success for e in completed)/len(tasks) if tasks else None,
            "observed_cost": cost,
            "valid_for_comparison": not errors and len(completed) == len(tasks),
            "cost_note": "A generation interrupted before its response may consume additional unreported tokens.",
        })
        write_json(output / "manifest.json", manifest)
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("run")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--deps-dir")
    p.add_argument("--image", default="wll-appworld:local")
    p.add_argument("--split", choices=["train", "dev", "test_normal", "test_challenge"], default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--one-per-scenario", action="store_true")
    p.add_argument("--task-ids-file")
    p.add_argument("--experience-report", help="Accepted report with trace-verified independent train utility")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--endpoint", default="http://127.0.0.1:18876/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--prompt-version", choices=list(SYSTEMS), default=DEFAULT_VERSION)
    p.add_argument("--output", required=True)
    p = subs.add_parser("compare")
    p.add_argument("--control", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("export-sft")
    p.add_argument("--episodes", required=True)
    p.add_argument("--validation-reports", nargs="+", help="Require validated transfer-distillation provenance")
    p.add_argument("--output", required=True)
    p = subs.add_parser("recover")
    p.add_argument("--episodes", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--deps-dir")
    p.add_argument("--image", default="wll-appworld:local")
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:18876/v1")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--start-attempt", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--output", required=True)
    p = subs.add_parser("validate-experience")
    p.add_argument("--experience", required=True, help="JSON candidate or single candidate JSONL row")
    p.add_argument("--task-ids-file", required=True, help="Frozen independent train utility tasks")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--deps-dir")
    p.add_argument("--image", default="wll-appworld:local")
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:18876/v1")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=.3)
    p.add_argument("--prompt-version", choices=list(SYSTEMS), default=DEFAULT_VERSION)
    p.add_argument("--min-scenarios", type=int, default=8)
    p.add_argument("--max-harm-rate", type=float, default=.05)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in {"run", "recover", "validate-experience"}:
        require_space(args.output)
    if args.command == "run":
        run(args)
    elif args.command == "compare":
        write_json(args.output, paired_estimate(read_episodes(args.control), read_episodes(args.treatment)))
    elif args.command == "recover":
        from .recovery import recover_failures

        model = LocalModel(args.endpoint, args.model, max_tokens=args.max_tokens,
                           temperature=args.temperature)
        recover_failures(read_episodes(args.episodes), model, data_dir=args.data_dir,
                         deps_dir=args.deps_dir, image=args.image, output=args.output,
                         max_steps=args.steps, attempts=args.attempts, start_attempt=args.start_attempt)
    elif args.command == "validate-experience":
        from .experience import Experience
        from .utility import validate_candidate

        candidate = json.loads(Path(args.experience).read_text())
        candidate = Experience(**candidate.get("experience", candidate))
        tasks = load_tasks(args.data_dir, "train", image=args.image, deps_dir=args.deps_dir,
                           include_task_ids=Path(args.task_ids_file).read_text().splitlines())
        model = LocalModel(args.endpoint, args.model, max_tokens=args.max_tokens,
                           temperature=args.temperature)
        image_id = subprocess.check_output(["docker", "image", "inspect", args.image,
                                            "--format", "{{.Id}}"], text=True).strip()
        validate_candidate(candidate, tasks,
            lambda task: AppWorldEnvironment(task.task_id, args.data_dir,
                                             args.steps, image_id, args.deps_dir),
            model, output=args.output, max_steps=args.steps, seed=args.seed,
            environment_id=image_id, prompt_version=args.prompt_version,
            min_scenarios=args.min_scenarios, max_harm_rate=args.max_harm_rate)
    else:
        episodes = read_episodes(args.episodes)
        rows = (export_validated_transfer_examples(episodes, args.validation_reports)
                if args.validation_reports else export_first_attempt_examples(episodes))
        with Path(args.output).open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
