"""Re-verify visible error states from successful train rollouts.

This is retrospective data mining, not a natural full-task failure or pass@1
measurement. Never relabel the original successful suffix as a new recovery.
"""
import json
from pathlib import Path

from .protocol import fingerprint
from .runner import run_episode
from .storage import atomic_json


def probe_intraepisode_errors(episodes, world_factory, *, output):
    if any(e.task.split != "train" for e in episodes):
        raise ValueError("intraepisode probes are train-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": "retrospective_error_states_from_successful_train_rollouts",
                "source_episode_hashes": [fingerprint(e.to_dict()) for e in episodes],
                "source_generation_tokens": sum(e.generated_tokens for e in episodes),
                "status": "running", "replayed_steps": 0, "failed_prefixes": 0}
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    probes = []
    try:
        with (output / "episodes.jsonl").open("w") as stream:
            for source in episodes:
                if not source.success:
                    continue
                error = next((i for i, s in enumerate(source.steps)
                              if s.observation.startswith("Execution failed")), None)
                if error is None:
                    continue

                class NoGeneration:
                    model_id = source.model_id

                    def generate(self, *args, **kwargs):
                        raise AssertionError("a prefix probe cannot generate new actions")

                def observe(index, step, replayed):
                    manifest["replayed_steps"] += 1

                probe = run_episode(source.task, lambda source=source: world_factory(source.task),
                    NoGeneration(), max_steps=error+1, seed=source.seed,
                    prefix=source.steps[:error+1], branch="prefix_probe", on_step=observe,
                    prompt_version=source.metadata.get("prompt_version", "appworld-v1"))
                probe.metadata.update(parent_episode_hash=fingerprint(source.to_dict()),
                                      source_rollout_success=True,
                                      environment_id=source.metadata.get("environment_id"),
                                      selection="first_visible_error_in_successful_train_rollout")
                stream.write(json.dumps(probe.to_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                probes.append(probe)
                manifest["failed_prefixes"] += int(not probe.success)
        manifest["status"] = "complete"
        return probes
    except BaseException as exc:
        manifest.update(status="interrupted", error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(manifest_path, manifest)
