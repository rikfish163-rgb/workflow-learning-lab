"""Prepare the frozen writer's inputs from both audited exploratory recoveries.

No model, network, task execution, candidate acceptance, or training occurs here.
The provenance checks in lesson_messages are not a substitute for the complete
raw pair and service-ledger audit performed by main.
"""
import copy
import json
import os
import time
from pathlib import Path

from scripts.run_foundation_pilot import digest, read, require
from wll.experience import draft_experience
from wll.protocol import fingerprint
from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
PARENT_SHA = '0520a92e25ea994e38ae1fbe58f80ad3c93487e2cd6f7ef07c0daa05084c033a'
COMPLETE = 'artifacts/proposed-pair-complete-integration-001.json'
COMPLETE_SHA = '273329d26f5abd5bda7591ff34ce0e2fd7f4a40f2d63fb667759d24245a58127'
OUTPUT = 'artifacts/recovery-lesson-inputs-001'


def lesson_messages(failure, recovery, *, condition, origin_scenarios):
    require(failure.task.split == 'train' and failure.task.scenario_id in origin_scenarios
        and failure.task == recovery.task and failure.success is False and recovery.success is True
        and failure.branch in {'first', 'recovery'} and recovery.branch == 'recovery'
        and failure.experience_id is None and recovery.experience_id is None,
        'known failed origin and successful recovery required')
    length = recovery.replayed_steps
    require(type(length) is int and 0 < length <= len(failure.steps) and length < len(recovery.steps)
        and failure.steps[:length] == recovery.steps[:length], 'recovery prefix differs')
    require(condition in ('no_hint', 'proposal_hint') and recovery.metadata.get('condition') == condition
        and recovery.metadata.get('source_hash') == fingerprint(failure.to_dict())
        and recovery.metadata.get('posthoc_exploratory_only') is True, 'recovery provenance differs')

    class Captured(Exception):
        pass

    class Capture:
        def generate(self, messages, seed):
            self.messages = copy.deepcopy(messages)
            raise Captured()

    model = Capture()
    try:
        draft_experience(failure, recovery, model, seed=5017)
    except Captured:
        return model.messages
    raise ValueError('frozen experience writer did not expose its request')


def main():
    from scripts.audit_proposed_pair import audit_stage
    from scripts.run_qwen38_proposed_pair import (
        CONDITIONS,
        ID,
        PARTITION,
        PLAN,
        load_registration,
        load_sources,
    )
    from wll.cli import read_episodes

    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation required')
    target = ROOT / OUTPUT
    design_path = ROOT / 'artifacts/recovery-lesson-design-001.json'
    require(not target.exists() and not design_path.exists(), 'lesson inputs already prepared')
    plan, _, inputs = load_registration(ROOT, ROOT / PLAN, PARENT_SHA)
    require(digest(ROOT / COMPLETE) == COMPLETE_SHA, 'completed pair evidence differs')
    complete = read(ROOT / COMPLETE)
    for name, sha in complete['input_sha256'].items():
        require(digest(ROOT / name) == sha, 'completed pair input changed: ' + name)
    stage = ROOT / 'artifacts' / ID / 'batch-0'
    generation = audit_stage(ROOT, plan, inputs, stage)
    result = read(stage / 'report.json')
    require(complete['phase'] == 'complete' and generation == complete['generation']
        and complete['result'] == result and result['outcomes'] == dict.fromkeys(CONDITIONS, True)
        and not result['hint_helped'] and not result['hint_harmed']
        and result['accepted_experiences'] == 0 and result['candidate_drafting'] is False,
        'complete pair or neutral success contrast differs')
    source, = load_sources(ROOT, inputs['ordered_sources'])
    origin = {s.rsplit('_', 1)[0] for s in read(ROOT / PARTITION)['task_ids']['origin']}
    records = []
    target.mkdir()
    for condition in CONDITIONS:
        recovery_path = stage / 'trials' / condition / 'episodes.jsonl'
        recovery, = read_episodes(recovery_path)
        messages = lesson_messages(source, recovery, condition=condition, origin_scenarios=origin)
        message_path = target / (condition + '-messages.json')
        atomic_json(message_path, messages)
        projection = json.loads(messages[1]['content'])
        records.append({'condition': condition, 'task_id': source.task.task_id,
            'source_file': inputs['ordered_sources'][0]['source_file'], 'source_hash': fingerprint(source.to_dict()),
            'recovery_file': str(recovery_path.relative_to(ROOT)), 'recovery_file_sha256': digest(recovery_path),
            'recovery_hash': fingerprint(recovery.to_dict()), 'replayed_steps': recovery.replayed_steps,
            'messages': str(message_path.relative_to(ROOT)), 'messages_sha256': digest(message_path),
            'message_fingerprint': fingerprint(messages), 'draft_seed': 5017,
            'projection': {kind: {'step_indices': [r['step_index'] for r in projection[kind]],
                'truncated_steps': [r['step_index'] for r in projection[kind] if r['truncated']]}
                for kind in ('failure', 'recovery')}})
    source_files = ['scripts/prepare_recovery_lesson_inputs.py', 'wll/experience.py', 'wll/protocol.py',
                    'tests/test_recovery_lesson_inputs.py']
    manifest = {'state': 'prepared_not_registered', 'created_epoch': time.time(), 'uses_gpu': False,
        'new_LLM_calls': 0, 'new_candidates': 0, 'accepted_experiences': 0,
        'parent_plan': PLAN, 'parent_plan_sha256': PARENT_SHA, 'complete_integration': COMPLETE,
        'complete_integration_sha256': COMPLETE_SHA, 'partition_sha256': digest(ROOT / PARTITION),
        'writer': 'unchanged_wll.experience.draft_experience',
        'trace_projection': 'trigger_tail_recovery_head_tail_v1',
        'max_tokens': 1024, 'temperature': .7, 'max_context': 16384, 'draft_seconds': 120,
        'ordered_sources': records, 'source_sha256': {n: digest(ROOT / n) for n in source_files},
        'selection': 'Both successful conditions in fixed no_hint, proposal_hint order. Same failed source, '
            'same frozen writer and seed; do not select a condition based on the lesson output.',
        'scope': 'One post-hoc origin, two correlated recoveries. No first-attempt gain, hint success-rate '
            'benefit, independent utility or learning. Protected trace inputs remain private.'}
    atomic_json(target / 'manifest.json', manifest)
    design = {'id': 'qwen38-recovery-lessons-001', 'state': 'draft_not_registered',
        'parent_plan': PLAN, 'parent_plan_sha256': PARENT_SHA, 'complete_integration': COMPLETE,
        'complete_integration_sha256': COMPLETE_SHA, 'inputs': OUTPUT + '/manifest.json',
        'inputs_sha256': digest(target / 'manifest.json'), 'conditions': CONDITIONS,
        'model': 'same frozen Qwen3.8-27B service as parent', 'draft_seed': 5017,
        'max_tokens': 1024, 'temperature': .7, 'max_context': 16384, 'draft_seconds': 120,
        'max_calls_per_condition': 1, 'no_automatic_retry': True,
        'stop_rule': 'Retain known invalid outputs and proceed; stop remaining inputs on unknown '
            'response, backend mismatch, interruption or unconfirmed service release.',
        'gpu_policy': plan['gpu_policy'], 'gpu_policy_sha256': plan['gpu_policy_sha256'],
        'accepted_experiences': 0, 'automatic_utility': False, 'automatic_training': False,
        'origin_contrast': {'no_hint': True, 'proposal_hint': True, 'hint_helped': False, 'hint_harmed': False},
        'historical_cost_records': [COMPLETE, 'artifacts/qwen38-semantic-critic-complete-integration-001.json',
            'artifacts/qwen38-semantic-critic-complete-integration-002.json'],
        'scope': manifest['scope'], 'missing_before_registration': [
            'Actual frozen tokenizer CPU preflight of both writer inputs',
            'Bounded standalone writer with raw response/meter audit and malformed/timeout/interrupt checks',
            'Single-GPU controller and service-event accounting with frozen source/input registration']}
    atomic_json(design_path, design)
    print(json.dumps({'state': manifest['state'], 'conditions': CONDITIONS, 'LLM_calls': 0,
        'GPU_jobs_queued': False, 'writer_projection': {r['condition']: r['projection'] for r in records}}))


if __name__ == '__main__':
    main()
