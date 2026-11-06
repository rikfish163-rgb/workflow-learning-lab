"""Experiment records and paired, scenario-clustered statistics."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from statistics import mean


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    scenario_id: str
    split: str
    instruction: str
    supervisor: str = ""


@dataclass
class Step:
    action: str
    observation: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    raw_response: str = ""


@dataclass
class Episode:
    task: Task
    model_id: str
    seed: int
    steps: list[Step]
    success: bool
    termination: str
    experience_id: str | None = None
    branch: str = "first"
    replayed_steps: int = 0
    elapsed_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def generated_tokens(self):
        return sum(s.completion_tokens for s in self.steps[self.replayed_steps:])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["task"] = Task(**data["task"])
        data["steps"] = [Step(**s) for s in data["steps"]]
        return cls(**data)


def assert_disjoint(*partitions: list[Task]) -> None:
    seen = set()
    for tasks in partitions:
        scenarios = {t.scenario_id for t in tasks}
        overlap = seen & scenarios
        if overlap:
            raise ValueError(f"scenario leakage: {sorted(overlap)}")
        seen |= scenarios


def paired_estimate(control: list[Episode], treatment: list[Episode], *,
                    bootstrap_samples=2000, seed=0) -> dict:
    def index(episodes):
        out = {}
        for e in episodes:
            key = (e.task.task_id, e.seed)
            if key in out:
                raise ValueError(f"duplicate pair: {key}")
            out[key] = e
        return out

    a, b = index(control), index(treatment)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical nonempty task/seed pairs")
    clusters = {}
    differences = []
    for key in sorted(a):
        if a[key].task != b[key].task:
            raise ValueError(f"pair task metadata differs: {key}")
        delta = int(b[key].success) - int(a[key].success)
        differences.append(delta)
        clusters.setdefault(a[key].task.scenario_id, []).append(delta)
    rng = random.Random(seed)
    values = list(clusters.values())
    samples = sorted(mean(d for group in rng.choices(values, k=len(values)) for d in group)
                     for _ in range(bootstrap_samples))
    if not samples:
        raise ValueError("bootstrap_samples must be positive")
    return {
        "n_pairs": len(a), "scenario_count": len(values),
        "control_success": mean(e.success for e in a.values()),
        "treatment_success": mean(e.success for e in b.values()),
        "delta": mean(differences),
        "helped": differences.count(1), "harmed": differences.count(-1),
        "cluster_bootstrap_ci95": [samples[int(.025 * len(samples))],
                                   samples[min(len(samples)-1, int(.975 * len(samples)))]],
        "bootstrap_seed": seed,
        "bootstrap_degenerate": len(set(differences)) == 1,
        "warning": "Few scenarios give unreliable intervals; paired seeds do not guarantee identical random draws. "
                   "When all observed differences are identical, the bootstrap interval is degenerate "
                   "and cannot establish equivalence or population certainty.",
    }
