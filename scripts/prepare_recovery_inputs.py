"""Prepare audited natural-failure references after the complete capability gate.

CPU-only preparation, with no queue, model generation, recovery or training.
The next study must separately freeze these inputs and its execution protocol.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scripts.report_observation_comparison import compare, lines
from scripts.run_foundation_pilot import digest, environment_id, read, require, verify_frozen_plan
from wll.cli import read_episodes
from wll.protocol import fingerprint
from wll.storage import atomic_json


def select_sources(episodes, proofs, *, max_steps):
    episodes, proofs = list(episodes), list(proofs)
    require(type(max_steps) is int and max_steps > 0, 'positive step budget required')
    ids = [e.task.task_id for e in episodes]
    require(ids and len(ids) == len(set(ids)), 'unique nonempty source episodes required')
    require(all(e.task.split == 'train' and e.branch == 'first' and e.replayed_steps == 0
                and e.experience_id is None and type(e.success) is bool for e in episodes),
            'only completed natural first train attempts are sources')
    expected = {e.task.task_id for e in episodes
                if any(s.observation.startswith('Execution failed') for s in e.steps)}
    require(len(proofs) == len(expected) and {p['task_id'] for p in proofs} == expected,
            'prefix proofs are missing, duplicated or unexpected')
    by_id = {p['task_id']: p for p in proofs}
    selected, excluded = [], []
    for episode in episodes:
        task_id = episode.task.task_id
        length = next((i+1 for i, s in enumerate(episode.steps)
                       if s.observation.startswith('Execution failed')), None)
        if length is not None:
            proof = by_id[task_id]
            source_class = ('successful_trace_with_visible_error' if episode.success
                            else 'failed_trace_with_visible_error')
            require(proof.get('scenario_id') == episode.task.scenario_id
                    and proof.get('source_hash') == fingerprint(episode.to_dict())
                    and proof.get('source_class') == source_class
                    and proof.get('source_success') is episode.success
                    and proof.get('prefix_length') == length
                    and proof.get('prefix_sha256') == fingerprint(
                        [asdict(s) for s in episode.steps[:length]])
                    and proof.get('status') == 'verified' and proof.get('prefix_success') is False
                    and proof.get('replay_observations') == length
                    and proof.get('cleanup_verified') is True,
                    'changed or unverified failed-prefix evidence')
        reason = ('source_already_successful' if episode.success else
                  'no_visible_execution_error' if length is None else
                  'no_remaining_step_budget' if length >= max_steps else None)
        if reason:
            excluded.append({'task_id': task_id, 'reason': reason})
        else:
            selected.append({'task_id': task_id, 'scenario_id': episode.task.scenario_id,
                'source_hash': fingerprint(episode.to_dict()), 'prefix_proof': by_id[task_id],
                'remaining_steps': max_steps-length})
    return {'selected': selected, 'excluded': excluded}


def prepare_inputs(root, plan_path, plan_sha, prefix_root):
    root, prefix_root = Path(root), Path(prefix_root)
    plan = verify_frozen_plan(root, plan_path, plan_sha)
    comparison = compare(root, plan_path, plan_sha)
    require(comparison['decision'] == 'eligible_for_separately_registered_recovery_study',
            'complete capability gate must pass before recovery preparation')
    identity = environment_id(plan)
    selected, excluded, hashes = [], [], {}
    for index in range(3):
        stage = root/'artifacts'/plan['id']/f'batch-{index}'
        prefix = prefix_root/f'batch-{index}'
        episodes = read_episodes(stage/'run/episodes.jsonl')
        manifest, summary = read(prefix/'manifest.json'), read(prefix/'summary.json')
        evidence = manifest['source_evidence']
        require(manifest['task_ids'] == [e.task.task_id for e in episodes]
                and manifest['source_hashes'] == [fingerprint(e.to_dict()) for e in episodes]
                and manifest['environment_id'] == evidence['environment_id'] == identity
                and evidence['plan_sha256'] == plan_sha and evidence['batch'] == index
                and manifest['script_sha256'] == evidence['tool_sha256'] ==
                    digest(root/'scripts/audit_recovery_prefixes.py'), 'prefix provenance differs')
        for name, expected_sha in evidence['source_files_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                    'prefix evidence must reference project files')
            require(digest(root/name) == expected_sha, 'prefix source evidence changed: '+name)
            hashes[name] = expected_sha
        proofs = lines(prefix/'prefixes.jsonl')
        result = select_sources(episodes, proofs, max_steps=plan['max_steps'])
        require(summary['status'] == 'complete' and summary['unknown_prefixes'] == 0
                and summary['model_generation_calls'] == 0
                and summary['source_tasks'] == len(episodes)
                and all(summary[k] == len(proofs) for k in ('selected_prefixes',
                    'attempted_prefixes', 'verified_prefixes', 'verified_failed_prefixes'))
                and summary['replay_observations'] == sum(p['prefix_length'] for p in proofs),
                'prefix summary is incomplete or inconsistent')
        cleanup = read(prefix/'cleanup.json')
        require(cleanup['all_recorded_containers_absent'] is True
                and all(v is True for v in cleanup['recorded_container_absence'].values()),
                'prefix cleanup was not recorded')
        for record in result['selected']:
            record.update(source_episodes=str((stage/'run/episodes.jsonl').relative_to(root)),
                          prefix_proofs=str((prefix/'prefixes.jsonl').relative_to(root)), batch=index)
        selected.extend(result['selected'])
        excluded.extend(result['excluded'])
        for folder in (stage, prefix):
            for path in folder.rglob('*'):
                if path.is_file():
                    hashes[str(path.relative_to(root))] = digest(path)
    return {'status': 'prepared_not_registered', 'source_study': plan['id'],
        'source_plan_sha256': plan_sha, 'environment_id': identity,
        'capability_result': comparison['feedback']['result'],
        'source_count': len(selected)+len(excluded), 'selected_count': len(selected),
        'selected': selected, 'excluded': excluded, 'input_sha256': hashes,
        'selection': 'All completed natural failures, in frozen source order, with the first '
                     'visible error verified unsuccessful and positive remaining step budget.',
        'source_sha256': {name: digest(root/name) for name in
            ('scripts/prepare_recovery_inputs.py', 'scripts/audit_recovery_prefixes.py',
             'scripts/report_observation_comparison.py', 'scripts/run_recovery_trial.py')},
        'new_LLM_calls': 0, 'new_GPU_jobs': 0, 'automatic_recovery_or_training': False,
        'limits': 'Historical source and CPU-prefix costs are references, not new generation. '
                  'A separately registered recovery plan and fresh GPU admission are still required.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--prefix-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), 'prepared inputs already exist')
    result = prepare_inputs(Path(__file__).resolve().parent.parent, Path(args.plan),
                            args.plan_sha256, Path(args.prefix_root).resolve())
    atomic_json(output, result)
    print(json.dumps({'status': result['status'], 'source_count': result['source_count'],
                      'selected_count': result['selected_count']}))


if __name__ == '__main__':
    main()
