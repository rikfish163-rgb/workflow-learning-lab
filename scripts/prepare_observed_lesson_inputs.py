"""Prepare two historical self-correction sources without replay, generation or GPU work.

These are derived views of previously verified prefixes, not newly executed
recoveries. The unchanged writer supports observed self-correction explicitly.
"""
import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from scripts.prepare_recovery_inputs import select_sources
from scripts.run_foundation_pilot import digest, read, require
from wll.cli import read_episodes
from wll.experience import draft_experience
from wll.protocol import fingerprint
from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA = '07e1059d459d6da2b74943d246975ab6da543b1fb16b35f3915f1669428a2e55'
COMPLETE = 'artifacts/qwen38-source-complete-integration-001.json'
COMPLETE_SHA = 'eb3e1306abc7a0ab5fa90c4af6ba8266948e115e9fca04353035fc48c51f1caf'
OUTPUT = 'artifacts/observed-lesson-inputs-001'
PARTITION = 'artifacts/partitions-v1/manifest.json'


def observed_messages(source, proof, *, origin_scenarios):
    require(source.task.split == 'train' and source.task.scenario_id in origin_scenarios
        and source.success is True and source.branch == 'first' and source.replayed_steps == 0
        and source.experience_id is None, 'successful first origin attempt required')
    checked = select_sources([source], [proof], max_steps=40)
    require(checked['selected'] == [] and checked['excluded'] == [
        {'task_id': source.task.task_id, 'reason': 'source_already_successful'}], 'observed source differs')
    length = proof['prefix_length']
    require(type(length) is int and 0 < length < len(source.steps), 'verified unfinished prefix required')
    prefix = replace(source, steps=copy.deepcopy(source.steps[:length]), success=False,
        termination='previously_verified_prefix', branch='prefix_probe', replayed_steps=length,
        elapsed_seconds=0, metadata={**copy.deepcopy(source.metadata),
            'parent_episode_hash': fingerprint(source.to_dict()),
            'record_kind': 'derived_view_of_previously_verified_prefix',
            'prefix_proof_hash': fingerprint(proof), 'new_execution': False})

    class Captured(Exception):
        pass

    class Capture:
        def generate(self, messages, seed):
            self.messages = copy.deepcopy(messages)
            raise Captured()

    capture = Capture()
    try:
        draft_experience(prefix, source, capture, seed=5017)
    except Captured:
        return prefix, capture.messages
    raise ValueError('writer request was not captured')


def main():
    from scripts.run_qwen38_capability import ID, PLAN, load_registration

    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation required')
    output = ROOT / OUTPUT
    require(not output.exists(), 'observed inputs already prepared')
    plan, _ = load_registration(ROOT, ROOT / PLAN, SOURCE_SHA)
    require(digest(ROOT / COMPLETE) == COMPLETE_SHA, 'completed source evidence differs')
    complete = read(ROOT / COMPLETE)
    for name, expected in complete['input_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
            and digest(ROOT / name) == expected, 'source evidence changed: ' + name)
    require(complete['result']['completed'] == 12 and complete['result']['successes'] == 9,
        'complete known source cohort required')
    partition = read(ROOT / PARTITION)
    roles = partition['task_ids']
    scenarios = {role: {task.rsplit('_', 1)[0] for task in ids} for role, ids in roles.items()}
    require(set(roles) == {'origin', 'utility', 'distill'}
        and all(not scenarios[a] & scenarios[b] for a, b in [('origin', 'utility'),
            ('origin', 'distill'), ('utility', 'distill')]), 'training roles overlap')
    official_path = Path(plan['data_dir']) / 'datasets/train.txt'
    official = [line.strip() for line in official_path.read_text().splitlines() if line.strip()]
    ids = [task for values in roles.values() for task in values]
    require(len(ids) == len(set(ids)) == len(official) == 90 and set(ids) == set(official),
        'registered roles do not exactly cover official train')
    ordered, census, seen = [], [], []
    for index in range(3):
        source_name = f'artifacts/{ID}/batch-{index}/run/episodes.jsonl'
        prefix_name = f'artifacts/qwen38-prefix-readiness-001/batch-{index}/prefixes.jsonl'
        episodes = read_episodes(ROOT / source_name)
        proofs = [json.loads(line) for line in (ROOT / prefix_name).read_text().splitlines()]
        # Reuse validation over all sources, including the failed/error case;
        # choose every successful source with an already verified visible error.
        select_sources(episodes, proofs, max_steps=plan['max_steps'])
        by_id = {proof['task_id']: proof for proof in proofs}
        for source in episodes:
            seen.append(source.task.task_id)
            proof = by_id.get(source.task.task_id)
            selected = source.success is True and proof is not None
            census.append({'task_id': source.task.task_id, 'scenario_id': source.task.scenario_id,
                'success': source.success, 'visible_execution_error': proof is not None,
                'selected_for_observed_writer_preparation': selected,
                'source_episode_hash': fingerprint(source.to_dict())})
            if selected:
                prefix, messages = observed_messages(source, proof, origin_scenarios=scenarios['origin'])
                ordered.append((source, prefix, messages, source_name, prefix_name, proof))
    require(seen == plan['task_ids'] and len(ordered) == 2
        and len({s.task.scenario_id for s, *_ in ordered}) == 2, 'source census differs')
    output.mkdir()
    records = []
    for source, prefix, messages, source_name, prefix_name, proof in ordered:
        name = source.task.task_id
        prefix_path, message_path = output / (name + '-prefix-view.json'), output / (name + '-messages.json')
        atomic_json(prefix_path, prefix.to_dict())
        atomic_json(message_path, messages)
        projection = json.loads(messages[1]['content'])
        records.append({'task_id': name, 'scenario_id': source.task.scenario_id,
            'evidence_kind': 'observed_self_correction', 'original_success_file': source_name,
            'original_success_hash': fingerprint(source.to_dict()), 'prefix_proofs': prefix_name,
            'prefix_proof_hash': fingerprint(proof), 'prefix_length': len(prefix.steps),
            'derived_prefix_file': str(prefix_path.relative_to(ROOT)), 'derived_prefix_sha256': digest(prefix_path),
            'messages': str(message_path.relative_to(ROOT)), 'messages_sha256': digest(message_path),
            'message_fingerprint': fingerprint(messages), 'draft_seed': 5017,
            'projection': {kind: {'step_indices': [row['step_index'] for row in projection[kind]],
                'truncated_steps': [row['step_index'] for row in projection[kind] if row['truncated']]}
                for kind in ('failure', 'recovery')}})
    manifest = {'state': 'prepared_not_registered', 'created_epoch': time.time(), 'uses_gpu': False,
        'new_model_calls': 0, 'new_environment_executions': 0, 'new_recovery_successes': 0,
        'new_candidates': 0, 'accepted_experiences': 0, 'automatic_training': False,
        'source_plan': PLAN, 'source_plan_sha256': SOURCE_SHA, 'source_complete': COMPLETE,
        'source_complete_sha256': COMPLETE_SHA, 'writer': 'unchanged_wll.experience.draft_experience',
        'source_census': census, 'ordered_sources': records,
        'registered_train_roles': {role: {'tasks': len(roles[role]), 'scenarios': len(values)}
            for role, values in scenarios.items()},
        'official_train_file_sha256': digest(official_path), 'registered_roles_cover_all_official_train': True,
        'remaining_unselected_origin_variants': sorted(set(roles['origin']) - set(seen)),
        'selection': 'All original successful first attempts with an already replay-verified failed first '
            'visible-error prefix. Original frozen task order. No candidate or utility outcome filtering.',
        'planning_context': 'Prepared after inspecting some outcomes of ongoing memory comparison 002; '
            'this is retrospective preparation, not an outcome-blind preregistered new hypothesis.',
        'max_tokens': 1024, 'temperature': .7, 'max_context': 16384, 'draft_seed': 5017,
        'source_sha256': {n: digest(ROOT / n) for n in ['scripts/prepare_observed_lesson_inputs.py',
            'tests/test_observed_lesson_inputs.py', 'scripts/prepare_recovery_inputs.py', 'wll/experience.py']},
        'input_sha256': {**complete['input_sha256'], COMPLETE: COMPLETE_SHA, PLAN: SOURCE_SHA,
            PARTITION: digest(ROOT / PARTITION)},
        'scope': 'Two previously observed self-corrections from two origin scenarios. Derived prefix views '
            'reuse existing replay/scoring evidence and add no replay or generation costs. '
            'Not new recoveries, learned knowledge, causal repair attribution or independent utility. '
            'All prompts/derived traces remain private; future writer execution needs separate registration '
            'after the active GPU study, actual tokenizer preflight and content review before any promotion.'}
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({'state': manifest['state'], 'source_scenarios': len(ordered),
        'prefix_lengths': [row['prefix_length'] for row in records], 'new_model_calls': 0,
        'new_environment_executions': 0, 'GPU_job_queued': False,
        'remaining_origin_variants': len(manifest['remaining_unselected_origin_variants'])}))


if __name__ == '__main__':
    main()
