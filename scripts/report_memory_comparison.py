"""Matched three-arm statistics; the shared control is never counted as two trials."""
import math
from collections import defaultdict

from scripts.exact_cluster_resampling import exact_cluster_resampling
from scripts.run_foundation_pilot import require
from scripts.run_memory_branch import ARMS
from scripts.run_memory_comparison import validate_inputs
from scripts.run_recovery_trial import inference_signature
from wll.experience import validate_experience
from wll.protocol import fingerprint, paired_estimate


def summarize_episodes(tasks, candidates, episodes, *, thresholds):
    validate_inputs(tasks, candidates, 0)
    require(set(thresholds) == {'min_scenarios', 'max_harm_rate'}
        and type(thresholds['min_scenarios']) is int and thresholds['min_scenarios'] >= 8
        and math.isfinite(thresholds['max_harm_rate']) and 0 <= thresholds['max_harm_rate'] <= .05,
        'original utility gate cannot be weakened')
    require(len(episodes) == 3 * len(tasks), 'all three arms are required')
    by_id, indexed = {t.task_id: t for t in tasks}, {}
    for episode in episodes:
        task_id, arm = episode.task.task_id, episode.metadata.get('arm')
        key = (task_id, arm)
        require(task_id in by_id and arm in ARMS and key not in indexed and episode.task == by_id[task_id]
            and type(episode.success) is bool and episode.branch == 'first' and episode.replayed_steps == 0
            and episode.experience_id == (candidates[arm].experience_id if arm != 'no_memory' else None),
            'unknown, duplicated or non-fresh branch')
        require(episode.task.split == 'train' and all(episode.task.scenario_id not in c.source_scenarios
            for c in candidates.values()), 'source scenario cannot validate its own candidates')
        indexed[key] = episode
    signature = set()
    for task in tasks:
        control = indexed[(task.task_id, 'no_memory')]
        for arm in ARMS:
            episode = indexed[(task.task_id, arm)]
            require(episode.seed == control.seed and episode.model_id == control.model_id, 'paired sampling differs')
            for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id', 'backend', 'rollout_seconds'):
                require(key in episode.metadata and episode.metadata[key] == control.metadata.get(key), 'paired setting differs: ' + key)
            signature.add(inference_signature(episode.metadata['backend']))
    require(len(signature) == 1, 'inference configuration differs across batches')
    groups = {arm: [indexed[(task.task_id, arm)] for task in tasks] for arm in ARMS}

    def contrast(control, treatment):
        clusters = defaultdict(list)
        for a, b in zip(control, treatment, strict=True):
            clusters[a.task.scenario_id].append(int(b.success) - int(a.success))
        return {'paired': paired_estimate(control, treatment), 'exact': exact_cluster_resampling(list(clusters.values()))}

    contrasts, passes = {}, {}
    for arm in ARMS[1:]:
        value = contrast(groups['no_memory'], groups[arm])
        legacy = validate_experience(candidates[arm], groups['no_memory'], groups[arm], **thresholds)
        value['legacy'] = legacy
        contrasts[arm + '_vs_no_memory'] = value
        passes[arm] = legacy['accepted'] and value['exact']['ci95'][0] > 0
    direct = contrast(groups['no_hint_memory'], groups['proposal_hint_memory'])
    contrasts['proposal_hint_vs_no_hint'] = direct
    steps = [step for e in episodes for step in e.steps]
    return {'audit': 'passed', 'n_tasks': len(tasks), 'scenario_count': len({t.scenario_id for t in tasks}),
        'observed_branches': len(episodes), 'shared_control_branches': len(groups['no_memory']),
        'successes_by_arm': {a: sum(e.success for e in es) for a, es in groups.items()},
        'contrasts': contrasts, 'candidate_gate_passes': passes,
        'source_contrast_positive': direct['paired']['delta'] > 0 and direct['exact']['ci95'][0] > 0,
        'observed_step_cost': {'steps': len(steps), 'prompt_tokens': sum(s.prompt_tokens for s in steps),
            'completion_tokens': sum(s.completion_tokens for s in steps)},
        'thresholds': thresholds, 'selection_rule_activated': False, 'accepted_experiences': 0,
        'episode_sha256': [fingerprint(e.to_dict()) for e in episodes],
        'scope': 'Reused train utility with correlated shared-control contrasts, not independent replications '
            'or familywise confidence. Episode statistics alone do not certify raw generation cost, resource '
            'release, candidate content, held-out efficacy or parameter benefit.'}
