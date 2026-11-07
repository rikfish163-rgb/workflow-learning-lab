"""A counterfactual upper bound for complete fixed controls; never candidate selection."""
from collections import defaultdict

from scripts.exact_cluster_resampling import exact_cluster_resampling
from scripts.run_foundation_pilot import require


def perfect_candidate_ceiling(tasks, controls):
    tasks, controls = list(tasks), list(controls)
    by_id = {task.task_id: task for task in tasks}
    require(tasks and len(by_id) == len(tasks)
        and all(t.split == 'train' and t.scenario_id == t.task_id.rsplit('_', 1)[0] for t in tasks),
        'unique declared training tasks required')
    require(len(controls) == len(tasks) and len({r['task_id'] for r in controls}) == len(tasks)
        and {r['task_id'] for r in controls} == set(by_id), 'all fixed controls required exactly once')
    groups = defaultdict(list)
    for row in controls:
        task = by_id[row['task_id']]
        require(row['scenario_id'] == task.scenario_id and type(row['success']) is bool,
            'known boolean outcome and matching scenario required')
        groups[task.scenario_id].append(int(not row['success']))
    interval = exact_cluster_resampling(list(groups.values()))['ci95']
    failures = sum(sum(group) for group in groups.values())
    return {'kind': 'counterfactual_fixed_control_upper_bound', 'task_count': len(tasks),
        'scenario_count': len(groups), 'actual_control_successes': len(tasks) - failures,
        'failed_control_scenarios': sum(any(group) for group in groups.values()),
        'hypothetical_perfect_candidate_successes': len(tasks), 'maximum_helped': failures,
        'minimum_harmed': 0, 'exact_conditional_ci95': interval,
        'positive_exact_lower_bound_possible': interval[0] > 0,
        'accepted_experiences': 0, 'new_model_calls': 0,
        'scope': 'A perfect hypothetical candidate pointwise maximizes every paired difference, '
            'resampled mean and quantile for these fixed controls. A nonpositive upper-bound '
            'lower endpoint rules out passing the required positive endpoint for any candidate '
            'on this design. A positive endpoint does not prove any actual candidate passes, '
            'satisfies other gates or improves held-out performance. This is not a model outcome.'}
