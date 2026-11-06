"""Freeze train roles by scenario before mining or candidate selection."""
from .protocol import fingerprint


def partition_train_ids(task_ids, *, seed=17, origin_scenarios=12, utility_scenarios=9,
                        reserved_origin=()):
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")
    scenarios = {task_id.rsplit("_", 1)[0] for task_id in task_ids}
    reserved = set(reserved_origin)
    if not reserved <= scenarios or len(reserved) > origin_scenarios:
        raise ValueError("invalid reserved origin scenarios")
    if min(origin_scenarios, utility_scenarios) <= 0 or (
            origin_scenarios + utility_scenarios >= len(scenarios)):
        raise ValueError("all three partitions must be nonempty")
    ordered = sorted(scenarios - reserved, key=lambda s: fingerprint([seed, s]))
    origin = reserved | set(ordered[:origin_scenarios-len(reserved)])
    remaining = [s for s in ordered if s not in origin]
    utility = set(remaining[:utility_scenarios])
    distill = set(remaining[utility_scenarios:])
    return {name: sorted(t for t in task_ids if t.rsplit("_", 1)[0] in selected)
            for name, selected in [("origin", origin), ("utility", utility), ("distill", distill)]}
