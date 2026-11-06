"""Bounded recovery and candidate construction on training failures only."""
import json
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

from .appworld import AppWorldEnvironment
from .experience import ExperienceDraftError, draft_experience
from .protocol import fingerprint
from .runner import run_episode
from .storage import atomic_json


def pivot_index(episode):
    """Use visible feedback, never hidden evaluator assertions, to choose a prefix."""
    for i, step in enumerate(episode.steps):
        if step.observation.startswith("Execution failed"):
            return i + 1
    # For an incorrect completion, restart before the last operation. This is
    # explicitly an outcome-informed training branch, not test-time pass@1.
    return max(0, len(episode.steps) - 1)


@contextmanager
def record_status(path, stats):
    stats["status"] = "running"
    atomic_json(path, stats)
    try:
        yield
        stats["status"] = "complete"
    except BaseException as exc:
        stats.update(status="interrupted", error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(path, stats)


def recover_failures(episodes, model, *, data_dir, deps_dir, image, output,
                     max_steps, attempts=2, start_attempt=0):
    if any(e.task.split != "train" for e in episodes):
        raise ValueError("cannot mine recovery experience from dev/test")
    if attempts < 1 or start_attempt < 0:
        raise ValueError("attempts must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stats = {"failed_source_tasks": 0, "recovery_attempts": 0, "recovered_tasks": 0,
             "candidates": 0, "fresh_resolves": 0, "fresh_successes": 0,
             "lesson_generation_tokens": 0, "lesson_prompt_tokens": 0,
             "generation_errors": 0, "rollout_completion_tokens": 0,
             "rollout_prompt_tokens": 0, "replayed_steps": 0,
             "source_generation_tokens": sum(e.generated_tokens for e in episodes),
             "cost_note": "Source generation is reported separately; interrupted in-flight generations may be unreported."}
    atomic_json(output / "manifest.json", {
        "source_episode_hashes": [fingerprint(e.to_dict()) for e in episodes],
        "model_id": model.model_id, "max_steps": max_steps, "attempts": attempts,
        "start_attempt": start_attempt,
        "image": image, "prompt_versions": sorted({e.metadata.get("prompt_version", "appworld-v1")
                                                    for e in episodes}),
        "max_tokens": getattr(model, "max_tokens", None),
        "temperature": getattr(model, "temperature", None),
        "code_hash": fingerprint({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}),
    })
    with record_status(output / "summary.json", stats), \
            (output / "steps.jsonl").open("w") as journal, \
            (output / "recoveries.jsonl").open("w") as recovery_file, \
            (output / "experiences.jsonl").open("w") as lessons_file, \
            (output / "resolves.jsonl").open("w") as resolves_file:
        for source in episodes:
            if source.success:
                continue
            stats["failed_source_tasks"] += 1
            factory = lambda source=source: AppWorldEnvironment(  # noqa: E731
                source.task.task_id, data_dir, max_steps, image, deps_dir)
            prefix = source.steps[:pivot_index(source)]
            for attempt in range(start_attempt, start_attempt+attempts):
                def observe(index, step, replayed, task_id=source.task.task_id,
                            attempt=attempt, phase="recovery"):
                    journal.write(json.dumps({"task_id": task_id, "attempt": attempt,
                        "phase": phase, "index": index, "replayed": replayed,
                        "step": asdict(step)}, ensure_ascii=False) + "\n")
                    journal.flush()
                    if replayed:
                        stats["replayed_steps"] += 1
                    else:
                        stats["rollout_completion_tokens"] += step.completion_tokens
                        stats["rollout_prompt_tokens"] += step.prompt_tokens

                if len(prefix) >= max_steps:
                    raise ValueError("recovery budget leaves no generation steps")
                stats["recovery_attempts"] += 1
                recovery = run_episode(source.task, factory, model, max_steps=max_steps,
                                       seed=source.seed+1000+attempt, prefix=prefix, branch="recovery",
                                       on_step=observe,
                                       prompt_version=source.metadata.get("prompt_version", "appworld-v1"))
                recovery.metadata.update(source_hash=fingerprint(source.to_dict()),
                                         pivot_rule="first_visible_error_else_before_last_v1")
                recovery_file.write(json.dumps(recovery.to_dict(), ensure_ascii=False) + "\n")
                recovery_file.flush()
                print(json.dumps({"task_id": source.task.task_id, "phase": "recovery",
                                  "attempt": attempt, "success": recovery.success}), flush=True)
                if not recovery.success:
                    continue
                stats["recovered_tasks"] += 1
                try:
                    lesson, cost = draft_experience(source, recovery, model, source.seed+2000)
                except ExperienceDraftError as exc:
                    stats["lesson_generation_tokens"] += exc.cost["completion_tokens"]
                    stats["lesson_prompt_tokens"] += exc.cost["prompt_tokens"]
                    stats["generation_errors"] += 1
                    break
                stats["candidates"] += 1
                stats["lesson_generation_tokens"] += cost["completion_tokens"]
                stats["lesson_prompt_tokens"] += cost["prompt_tokens"]
                lessons_file.write(json.dumps({"experience": asdict(lesson), "cost": cost,
                                                "status": "candidate_unvalidated"}) + "\n")
                lessons_file.flush()
                resolved = run_episode(source.task, factory, model, max_steps=max_steps,
                                       seed=source.seed+3000, experience=lesson.render(),
                                       experience_id=lesson.experience_id,
                                       on_step=partial(observe, phase="resolve"),
                                       prompt_version=source.metadata.get("prompt_version", "appworld-v1"))
                resolved.metadata.update(experience_status="candidate_unvalidated",
                                         source_hash=fingerprint(source.to_dict()))
                resolves_file.write(json.dumps(resolved.to_dict(), ensure_ascii=False) + "\n")
                resolves_file.flush()
                stats["fresh_resolves"] += 1
                stats["fresh_successes"] += int(resolved.success)
                break
            atomic_json(output / "summary.json", stats)
    return stats
