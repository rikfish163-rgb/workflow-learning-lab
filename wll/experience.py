"""Conditional lessons and out-of-origin, paired empirical utility checks.

Acceptance is an empirical filter, not proof of causality or general validity.
Thresholds must be fixed before running comparisons; test split is never used.
"""
from dataclasses import asdict, dataclass

from .protocol import Episode, fingerprint, paired_estimate


class ExperienceDraftError(ValueError):
    """Schema failure after generation; consumed tokens must remain accounted."""
    def __init__(self, cost):
        super().__init__("invalid experience schema")
        self.cost = cost


@dataclass(frozen=True)
class Experience:
    experience_id: str
    failure_mode: str
    applies_when: str
    avoid_when: str
    rule: str
    source_scenarios: list[str]
    source_episode_hashes: list[str]

    def render(self):
        return (f"Apply when: {self.applies_when}\nAvoid when: {self.avoid_when}\n"
                f"Procedure: {self.rule}")


def draft_experience(failure, recovery, model, seed=0):
    if failure.task.split != "train" or recovery.task != failure.task:
        raise ValueError("experience requires the same train task")
    if failure.success or not recovery.success:
        raise ValueError("requires a failed attempt and a verified successful recovery")
    observed = recovery.branch == "first"
    if observed:
        if not 0 < len(failure.steps) < len(recovery.steps) or (
                failure.replayed_steps != len(failure.steps)) or not (
                failure.steps[-1].observation.startswith("Execution failed")) or (
                failure.branch != "prefix_probe") or failure.metadata.get("parent_episode_hash") != (
                fingerprint(recovery.to_dict())) or failure.steps != recovery.steps[:len(failure.steps)]:
            raise ValueError("observed self-correction requires exact prefix and parent provenance")
    elif recovery.branch != "recovery":
        raise ValueError("requires a new recovery or a traceable observed self-correction")
    import json

    # Bounded, declared projection: duplicate replay history otherwise consumes
    # most of the context. Full trace hashes below preserve auditability.
    trigger = next((i for i, s in enumerate(failure.steps)
                    if s.observation.startswith("Execution failed")), 0)
    failure_indices = sorted({*range(max(0, trigger-1), min(len(failure.steps), trigger+2)),
                              *range(max(0, len(failure.steps)-2), len(failure.steps))})
    suffix_start = len(failure.steps) if observed else recovery.replayed_steps
    recovery_indices = sorted({*range(suffix_start, min(len(recovery.steps), suffix_start+4)),
                               *range(max(suffix_start, len(recovery.steps)-3), len(recovery.steps))})

    def project(episode, indices):
        return [{"step_index": i, "action": episode.steps[i].action[:1200],
                 "observation": episode.steps[i].observation[:800],
                 "truncated": len(episode.steps[i].action) > 1200 or
                              len(episode.steps[i].observation) > 800} for i in indices]

    prompt = {
        "instruction": failure.task.instruction,
        "projection_note": "Only error-neighbor/tail steps and recovery head/tail are shown; fields may be truncated.",
        "failure": project(failure, failure_indices),
        "recovery": project(recovery, recovery_indices),
    }
    response = model.generate([
        {"role": "system", "content":
         "Extract one reusable workflow lesson. Return JSON with exactly failure_mode, "
         "applies_when, avoid_when, rule (all strings). State observable preconditions and "
         "when the lesson would be harmful. No task-specific answers, names, IDs, account "
         "credentials or copied solution code. This is a candidate, not verified knowledge."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ], seed=seed)
    cost = {"prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"], "model_id": model.model_id,
            "trace_projection": "trigger_tail_recovery_head_tail_v1",
            "evidence_kind": "observed_self_correction" if observed else "new_recovery"}
    text = response["text"].strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ExperienceDraftError(cost) from exc
    fields = {"failure_mode", "applies_when", "avoid_when", "rule"}
    if not isinstance(data, dict) or set(data) != fields or any(not isinstance(v, str) or not 1 <= len(v) <= 1500
                                  for v in data.values()):
        raise ExperienceDraftError(cost)
    hashes = [fingerprint(failure.to_dict()), fingerprint(recovery.to_dict())]
    result = Experience(fingerprint([data, hashes])[:16], **data,
                        source_scenarios=[failure.task.scenario_id], source_episode_hashes=hashes)
    return result, cost


def validate_experience(experience, control, treatment, *, min_scenarios=8, max_harm_rate=.05):
    for e in control + treatment:
        if e.task.split != "train":
            raise ValueError("experience validation is restricted to train scenarios")
        if e.task.scenario_id in experience.source_scenarios:
            raise ValueError("origin scenario is not independent validation")
        if e.branch != "first" or e.replayed_steps:
            raise ValueError("utility validation requires fresh first attempts")
    if any(e.experience_id is not None for e in control):
        raise ValueError("control must not contain an experience")
    if any(e.experience_id != experience.experience_id for e in treatment):
        raise ValueError("treatment experience does not match candidate")
    report = paired_estimate(control, treatment)
    controls = {(e.task.task_id, e.seed): e for e in control}
    for e in treatment:
        a = controls[(e.task.task_id, e.seed)]
        if e.model_id != a.model_id:
            raise ValueError("utility comparison must use the same model per pair")
        for key in ["max_steps", "max_tokens", "temperature", "prompt_version", "environment_id", "backend"]:
            if e.metadata.get(key) != a.metadata.get(key):
                raise ValueError(f"utility comparison requires equal budget/scaffold: {key}")
    report.update({
        "candidate": asdict(experience),
        "accepted": report["scenario_count"] >= min_scenarios
                    and report["cluster_bootstrap_ci95"][0] > 0
                    and report["harmed"] / report["n_pairs"] <= max_harm_rate,
        "selection_config": {"min_scenarios": min_scenarios, "max_harm_rate": max_harm_rate},
        "validation_episode_hashes": [fingerprint(e.to_dict()) for e in control + treatment],
    })
    return report


def load_validated_experience(report_path):
    """Recompute acceptance from trace hashes, rather than trusting a status flag."""
    import json
    from pathlib import Path

    path = Path(report_path)
    report = json.loads(path.read_text())
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("status") != "complete" or not report.get("accepted"):
        raise ValueError("experience validation is incomplete or rejected")
    if manifest.get("thresholds") != report["selection_config"] or (
            manifest.get("candidate") != report["candidate"]):
        raise ValueError("validation report differs from its preregistered configuration")
    episodes = [Episode.from_dict(json.loads(line)) for line in
                (path.parent / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if sorted(fingerprint(e.to_dict()) for e in episodes) != sorted(report["validation_episode_hashes"]):
        raise ValueError("validation episode hash mismatch")
    candidate = Experience(**report["candidate"])
    control = [e for e in episodes if e.experience_id is None]
    treatment = [e for e in episodes if e.experience_id is not None]
    recomputed = validate_experience(candidate, control, treatment, **report["selection_config"])
    if recomputed != report:
        raise ValueError("validation report cannot be reproduced by the current protocol")
    models = {e.model_id for e in episodes}
    if len(models) != 1:
        raise ValueError("transfer validation must identify one target model")
    backends = {fingerprint(e.metadata.get("backend")) for e in episodes}
    if len(backends) != 1:
        raise ValueError("transfer validation must use one inference backend")
    return candidate, {"report_hash": fingerprint(report), "model_id": next(iter(models)),
                       "backend": episodes[0].metadata.get("backend"),
                       "validation_scenarios": sorted({e.task.scenario_id for e in episodes})}
