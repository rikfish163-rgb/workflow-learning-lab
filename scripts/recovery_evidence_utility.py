"""Pure, prospective full-27 control-first utility protocol and statistics.

Inputs are authenticated metadata and immutable Episodes supplied by a future
private runtime adapter. These functions perform no I/O, scheduling or training.
Review/release flags are caller attestations, not independent certification.
"""

import copy
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace

from scripts import coverage_contract
from scripts.exact_cluster_resampling import exact_cluster_resampling
from scripts.recovery_evidence_lesson import EVIDENCE_KIND, SOURCE_HASH_KIND, SOURCE_SCENARIOS
from scripts.run_qwen38_b3_dev import static_backend
from scripts.utility_feasibility import perfect_candidate_ceiling
from wll.experience import Experience, validate_experience
from wll.protocol import Episode, Step, Task, fingerprint, paired_estimate

ARMS = ('no_memory', 'candidate', 'matched_unrelated_placebo')
UTILITY_SCENARIOS = ('229360a', '34d9492', '6104387', '76f2c72', '7d7fbf6',
                     'aa8502b', 'afc0fce', 'ce359b5', 'e7a10f8')
TASK_IDS = tuple(f'{scenario}_{variant}' for scenario in UTILITY_SCENARIOS for variant in (1, 2, 3))
MODEL_ID = 'qwen38-b3-matched-b3-001'
B3_STATIC_SHA256 = '5ba013087f2310751371def2f1ffefa53e8378b3c964f86d0a1cb5e388c93249'
GPU_POLICY_SHA256 = '22dec644b27956ef8eb379cdc6560fceafe89b9d5e0a45212f5420a267c0e166'
ALLOWED_GPUS = {'8': 'GPU-ecc8cd45-fdaa-b5ee-e54a-bf2ef7df8f82',
                '9': 'GPU-4fc78635-e0b9-1acf-774f-9debcfc7fa47'}
THRESHOLDS = {'min_scenarios': 8, 'max_harm_rate': .05}
FIXED = {'design': 'recovery-lesson-utility-design-002', 'model_id': MODEL_ID,
    'max_steps': 40, 'rollout_seconds': 600, 'max_tokens': 1024,
    'max_input_tokens': 15360, 'max_context_tokens': 16384, 'temperature': .7,
    'seed': 17, 'environment_seed': 100, 'environment_seed_protocol': 'wll-appworld-seed-v1',
    'prompt_version': 'appworld-v3', 'observation_feedback_kind': 'silent_documentation_print_hint_v1',
    'environment_image': 'sha256:9256a65732145731805985e32e9632d2920d7bddc38777e5cd05c24b3672e1fa',
    'automatic_retry': False, 'training_authorized': False}
EPISODE_SETTINGS = ('max_steps', 'max_tokens', 'max_input_tokens', 'max_context_tokens',
    'temperature', 'prompt_version', 'environment_id', 'rollout_seconds', 'environment_seed',
    'environment_seed_protocol', 'observation_feedback_kind', 'environment_image')
COMPARISON_VIEW = {'name': 'existing_B3_static_backend_v1',
                  'function': 'scripts.run_qwen38_b3_dev.static_backend',
                  'excluded_backend_fields': ['gpu_policy', 'loading_peak_reserved_bytes']}
COVERAGE_CONTRACT_DESIGN = 'coverage-contract-utility-001'
STRUCTURAL_PLACEBO_KIND = 'assistant_authored_unrelated_placebo'
WRITER_PLACEBO_KIND = 'human_authored_unrelated_placebo'
_SOURCE_KEYS = ('source_scenarios', 'source_trace_sha256', 'source_hash_kind',
                'evidence_kind', 'source_episode_hashes_semantics')
_STRUCTURAL_KEYS = {'creator', 'contract_version', 'construction_sha256', 'evidence',
                    'source_provenance_authenticated'}


def _require(condition):
    if not condition:
        raise ValueError('recovery_evidence_utility_invalid')


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def validate_cohort(partition, tasks):
    """Validate only frozen train-role metadata; never open a task or outcome."""
    _require(type(partition) is dict and partition.get('seed') == 17
        and type(partition.get('task_ids')) is dict
        and set(partition['task_ids']) == {'origin', 'utility', 'distill'})
    roles = partition['task_ids']
    _require(all(type(ids) is list and ids and len(ids) == len(set(ids))
                 and all(type(name) is str and re.fullmatch('[a-z0-9]+_[123]', name) for name in ids)
                 for ids in roles.values()))
    scenarios = {role: {name.rsplit('_', 1)[0] for name in ids} for role, ids in roles.items()}
    _require(roles['utility'] == list(TASK_IDS)
        and all(not scenarios[a] & scenarios[b] for a, b in
                [('origin', 'utility'), ('origin', 'distill'), ('utility', 'distill')])
        and {f'{source}_{variant}' for source in SOURCE_SCENARIOS for variant in (1, 2, 3)} <= set(roles['origin']))
    _require(type(tasks) in (list, tuple) and len(tasks) == 27
        and all(isinstance(task, Task) and task.split == 'train'
                and task.scenario_id == task.task_id.rsplit('_', 1)[0] for task in tasks)
        and [task.task_id for task in tasks] == list(TASK_IDS))
    return list(TASK_IDS)


def branch_schedule():
    return ([(name, ARMS[0]) for name in TASK_IDS]
        + [(name, arm) for index, name in enumerate(TASK_IDS)
           for arm in (ARMS[1:] if index % 2 == 0 else ARMS[:0:-1])])


def batch_schedules(phase):
    _require(phase in {'controls', 'prompted'})
    schedule = branch_schedule()[:27] if phase == 'controls' else branch_schedule()[27:]
    return [schedule[index:index + 6] for index in range(0, len(schedule), 6)]


def _backend(backend):
    _require(type(backend) is dict and type(backend.get('gpu_policy')) is dict)
    gpu, peak = backend['gpu_policy'], backend.get('loading_peak_reserved_bytes')
    index = gpu.get('physical_gpu_index')
    _require(type(index) is int and str(index) in ALLOWED_GPUS
        and gpu.get('hostname') == 'amax' and gpu.get('gpu_uuid') == ALLOWED_GPUS[str(index)]
        and type(gpu.get('visible_device_count')) is int and gpu['visible_device_count'] == 1
        and gpu.get('policy_sha256') == GPU_POLICY_SHA256
        and type(peak) is int and 0 < peak <= 20 * 1024**3
        and fingerprint(static_backend(backend)) == B3_STATIC_SHA256)


def _contexts(initial, lesson_tokens):
    _require(type(lesson_tokens) is dict and set(lesson_tokens) == set(ARMS[1:])
        and all(type(value) is int and 0 < value <= FIXED['max_input_tokens'] for value in lesson_tokens.values())
        and lesson_tokens[ARMS[1]] == lesson_tokens[ARMS[2]])
    _require(type(initial) is dict and set(initial) == set(TASK_IDS)
        and all(type(values) is dict and set(values) == set(ARMS)
            and all(type(value) is int and 0 < value <= FIXED['max_input_tokens'] for value in values.values())
            for values in initial.values()))


def _candidate_sources(provenance):
    _require(type(provenance) is dict
        and provenance.get('source_scenarios') == list(SOURCE_SCENARIOS)
        and type(provenance.get('source_trace_sha256')) is list
        and len(provenance['source_trace_sha256']) == 4
        and all(_sha(value) for value in provenance['source_trace_sha256'])
        and len(set(provenance['source_trace_sha256'])) == 4
        and provenance.get('source_hash_kind') == SOURCE_HASH_KIND
        and provenance.get('evidence_kind') == EVIDENCE_KIND
        and provenance.get('source_episode_hashes_semantics') == 'private_trace_sha256_not_episode_objects')


def _validate_candidate_provenance(candidate, provenance):
    """Keep writer and structural origins distinct; neither authenticates source bytes.

    A structural snapshot is fixed public grammar plus counts/enums. Validate the
    complete snapshot even when a control/placebo caller supplies no Experience.
    """
    p = provenance
    _require(type(p) is dict and 'writer_result_fingerprint' not in p
             and 'extraction_result_fingerprint' not in p)
    structural = bool(_STRUCTURAL_KEYS & p.keys())
    if structural:
        coverage_contract.validate_contract(p)
        if candidate is None:
            candidate = Experience(**p['experience'])
    _candidate_sources(p)
    _require(isinstance(candidate, Experience) and p.get('status') == 'candidate_unvalidated'
        and p.get('experience') == asdict(candidate)
        and candidate.source_scenarios == p['source_scenarios']
        and candidate.source_episode_hashes == p['source_trace_sha256']
        and all(p.get(key) is False for key in
                ('accepted', 'content_certified', 'utility_verified', 'training_authorized')))
    if structural:
        return {'design': COVERAGE_CONTRACT_DESIGN, 'candidate_provenance': copy.deepcopy(p),
            'construction_sha256': p['construction_sha256'],
            'extraction_result_fingerprint': fingerprint(p)}
    return {'design': FIXED['design'],
        'candidate_provenance': {key: copy.deepcopy(p[key]) for key in _SOURCE_KEYS},
        'writer_result_fingerprint': fingerprint(p)}


def _validate_placebo_provenance(provenance, design, *, placebo=None):
    """Check each design's creator, and bind structural fields when supplied."""
    _require(design in {COVERAGE_CONTRACT_DESIGN, FIXED['design']})
    expected_kind = STRUCTURAL_PLACEBO_KIND if design == COVERAGE_CONTRACT_DESIGN else WRITER_PLACEBO_KIND
    _require(type(provenance) is dict and set(provenance) == {
        'kind', 'construction_sha256', 'derived_from_recovery_sources'}
        and provenance['kind'] == expected_kind
        and _sha(provenance['construction_sha256'])
        and provenance['derived_from_recovery_sources'] is False)
    if design == COVERAGE_CONTRACT_DESIGN and placebo is not None:
        _require(isinstance(placebo, Experience))
        fields = {key: getattr(placebo, key) for key in ('failure_mode', 'applies_when', 'avoid_when', 'rule')}
        construction = fingerprint({'creator': expected_kind, 'fields': fields})
        _require(provenance['construction_sha256'] == construction
            and placebo.experience_id == construction[:16])


def _validate_protocol_provenance(protocol, candidate=None):
    """Recheck either origin for every arm, including controls without a lesson."""
    _require(type(protocol) is dict)
    p = protocol.get('candidate_provenance')
    if protocol.get('design') == COVERAGE_CONTRACT_DESIGN:
        expected = _validate_candidate_provenance(candidate, p)
        _validate_placebo_provenance(protocol.get('placebo_provenance'), COVERAGE_CONTRACT_DESIGN)
        _require(protocol.get('placebo_id') == protocol['placebo_provenance']['construction_sha256'][:16])
        _require(expected['design'] == COVERAGE_CONTRACT_DESIGN
            and 'writer_result_fingerprint' not in protocol
            and all(protocol.get(key) == value for key, value in expected.items()))
        candidate = Experience(**p['experience'])
    else:
        _require(protocol.get('design') == FIXED['design']
            and type(p) is dict and set(p) == set(_SOURCE_KEYS)
            and 'construction_sha256' not in protocol and 'extraction_result_fingerprint' not in protocol
            and _sha(protocol.get('writer_result_fingerprint')))
        _candidate_sources(p)
    _require(type(protocol.get('candidate_id')) is str and bool(protocol['candidate_id'])
        and _sha(protocol.get('candidate_sha256')))
    if candidate is not None:
        _require(isinstance(candidate, Experience)
            and candidate.source_scenarios == p['source_scenarios']
            and candidate.source_episode_hashes == p['source_trace_sha256']
            and candidate.experience_id == protocol['candidate_id']
            and fingerprint(asdict(candidate)) == protocol['candidate_sha256'])


def make_protocol(partition, tasks, candidate, placebo, *, candidate_provenance,
                  placebo_provenance, reviews, lesson_tokens, initial_input_tokens,
                  expected_backend, environment_id):
    """Freeze one reviewed candidate and a provenance-bound matched placebo.

    Token counts and reviews must come from separately authenticated preparation;
    this validates their metadata, without tokenizing or certifying private text.
    """
    validate_cohort(partition, tasks)
    _require(all(isinstance(value, Experience) and type(value.experience_id) is str and value.experience_id
        and all(type(getattr(value, key)) is str and 0 < len(getattr(value, key).strip()) <= 1500
                for key in ('failure_mode', 'applies_when', 'avoid_when', 'rule'))
        for value in (candidate, placebo)))
    _require(candidate.experience_id != placebo.experience_id and candidate.render() != placebo.render()
        and candidate.source_scenarios == list(SOURCE_SCENARIOS)
        and type(candidate.source_episode_hashes) is list and len(candidate.source_episode_hashes) == 4
        and len(set(candidate.source_episode_hashes)) == 4 and all(_sha(value) for value in candidate.source_episode_hashes)
        and placebo.source_scenarios == [] and placebo.source_episode_hashes == [])
    origin = _validate_candidate_provenance(candidate, candidate_provenance)
    _validate_placebo_provenance(placebo_provenance, origin['design'], placebo=placebo)
    _require(type(reviews) is dict and set(reviews) == {'candidate', 'placebo', 'placebo_unrelated'}
             and all(value is True for value in reviews.values()))
    _contexts(initial_input_tokens, lesson_tokens)
    _backend(expected_backend)
    _require(type(environment_id) is str and bool(environment_id.strip()))
    return {**FIXED, **origin, 'task_sha256': {task.task_id: fingerprint(asdict(task)) for task in tasks},
        'partition_fingerprint': fingerprint(partition), 'environment_id': environment_id,
        'candidate_id': candidate.experience_id, 'candidate_sha256': fingerprint(asdict(candidate)),
        'placebo_id': placebo.experience_id, 'placebo_sha256': fingerprint(asdict(placebo)),
        'placebo_provenance': copy.deepcopy(placebo_provenance),
        'reviews': dict(reviews), 'lesson_message_tokens': dict(lesson_tokens),
        'initial_input_tokens': copy.deepcopy(initial_input_tokens),
        'expected_backend_static': copy.deepcopy(static_backend(expected_backend)),
        'registration_backend_sha256': fingerprint(expected_backend),
        'comparison_view': copy.deepcopy(COMPARISON_VIEW), 'thresholds': dict(THRESHOLDS)}


def _protocol(partition, tasks, protocol):
    validate_cohort(partition, tasks)
    _require(type(protocol) is dict and all(type(protocol.get(key)) is type(value) and protocol[key] == value
        for key, value in FIXED.items() if key != 'design') and protocol.get('thresholds') == THRESHOLDS
        and protocol.get('comparison_view') == COMPARISON_VIEW
        and protocol.get('partition_fingerprint') == fingerprint(partition)
        and protocol.get('task_sha256') == {task.task_id: fingerprint(asdict(task)) for task in tasks}
        and fingerprint(protocol.get('expected_backend_static')) == B3_STATIC_SHA256
        and protocol.get('reviews') == {'candidate': True, 'placebo': True, 'placebo_unrelated': True})
    _validate_protocol_provenance(protocol)
    _contexts(protocol['initial_input_tokens'], protocol['lesson_message_tokens'])


def phase_decision(partition, tasks, protocol, records):
    """Check a schedule prefix; unknowns never become failures or authorize a ceiling."""
    _protocol(partition, tasks, protocol)
    schedule = branch_schedule()
    _require(type(records) is list and len(records) <= 81)
    known = []
    for index, row in enumerate(records):
        _require(type(row) is dict and set(row) == {'task_id', 'scenario_id', 'arm', 'outcome',
            'valid_for_continuation', 'released', 'protocol_sha256'}
            and (row['task_id'], row['arm']) == schedule[index]
            and row['scenario_id'] == row['task_id'].rsplit('_', 1)[0]
            and (type(row['outcome']) is bool or row['outcome'] is None)
            and type(row['valid_for_continuation']) is bool and type(row['released']) is bool
            and row['protocol_sha256'] == fingerprint(protocol))
        valid = type(row['outcome']) is bool and row['valid_for_continuation'] and row['released']
        _require(valid or index == len(records) - 1)
        known.append(valid)
    attempted = Counter(row['arm'] for row in records)
    result = {'status': None, 'attempted_branches': len(records), 'observed_branches': len(records),
        'known_branches': sum(known), 'unknown_branches': len(records) - sum(known),
        'unattempted_by_arm': {arm: 27 - attempted[arm] for arm in ARMS},
        'known_true_by_arm': {arm: sum(valid and row['arm'] == arm and row['outcome'] is True
                                    for row, valid in zip(records, known, strict=True)) for arm in ARMS},
        'known_false_by_arm': {arm: sum(valid and row['arm'] == arm and row['outcome'] is False
                                     for row, valid in zip(records, known, strict=True)) for arm in ARMS},
        'ceiling': None, 'accepted_experiences': 0, 'training_authorized': False}
    if len(records) >= 27 and all(known[:27]):
        ceiling = perfect_candidate_ceiling(tasks, [{'task_id': row['task_id'],
            'scenario_id': row['scenario_id'], 'success': row['outcome']} for row in records[:27]])
        result['ceiling'] = ceiling
        if not ceiling['positive_exact_lower_bound_possible']:
            _require(len(records) == 27)
            result['status'] = 'stopped_uninformative_ceiling'
            return result
    if not all(known):
        result['status'] = 'stopped_unknown'
    elif len(records) < 27:
        result['status'] = 'awaiting_controls'
    else:
        result['status'] = 'ready_for_prompted' if len(records) == 27 else (
            'complete' if len(records) == 81 else 'awaiting_prompted')
    return result


def _contrast(control, treatment):
    clusters = defaultdict(list)
    for a, b in zip(control, treatment, strict=True):
        clusters[a.task.scenario_id].append(int(b.success) - int(a.success))
    return {'paired': paired_estimate(control, treatment),
            'exact': exact_cluster_resampling(list(clusters.values()))}


def summarize(partition, tasks, candidate, placebo, protocol, records, episodes):
    """Report unchanged primary gates plus separate placebo contrasts; never accept."""
    result = phase_decision(partition, tasks, protocol, records)
    result.update(candidate_gate_passes=None, contrasts=None, selection_rule_activated=False)
    if result['status'] != 'complete':
        return result
    _require(fingerprint(asdict(candidate)) == protocol['candidate_sha256']
        and fingerprint(asdict(placebo)) == protocol['placebo_sha256']
        and type(episodes) is list and len(episodes) == 81)
    by_id, indexed, views = {task.task_id: task for task in tasks}, {}, []
    rows = {(row['task_id'], row['arm']): row for row in records}
    for episode in episodes:
        _require(isinstance(episode, Episode))
        key = (episode.task.task_id, episode.metadata.get('arm'))
        _require(key in rows and key not in indexed and episode.task == by_id[key[0]]
            and type(episode.success) is bool and episode.success is rows[key]['outcome']
            and episode.branch == 'first' and type(episode.replayed_steps) is int
            and episode.replayed_steps == 0
            and type(episode.seed) is int and episode.seed == 17 and episode.model_id == MODEL_ID
            and episode.experience_id == (None if key[1] == ARMS[0] else protocol[
                'candidate_id' if key[1] == ARMS[1] else 'placebo_id'])
            and all(type(episode.metadata.get(name)) is type(protocol[name])
                    and episode.metadata[name] == protocol[name] for name in EPISODE_SETTINGS)
            and type(episode.metadata.get('world_completed')) is bool)
        _require(type(episode.steps) is list and len(episode.steps) <= 40
            and all(isinstance(step, Step) and type(step.prompt_tokens) is int
                and 0 <= step.prompt_tokens <= 15360 and type(step.completion_tokens) is int
                and 0 <= step.completion_tokens <= 1024 for step in episode.steps))
        _backend(episode.metadata['backend'])
        view = replace(episode, metadata={**copy.deepcopy(episode.metadata),
                                        'backend': copy.deepcopy(static_backend(episode.metadata['backend']))})
        indexed[key] = view
        views.append(view)
    groups = {arm: [indexed[(task.task_id, arm)] for task in tasks] for arm in ARMS}
    primary = _contrast(groups[ARMS[0]], groups[ARMS[1]])
    legacy = validate_experience(candidate, groups[ARMS[0]], groups[ARMS[1]], **THRESHOLDS)
    primary['legacy'] = {key: legacy[key] for key in ('accepted', 'cluster_bootstrap_ci95',
                                                   'n_pairs', 'scenario_count', 'helped', 'harmed')}
    result.update(scenario_count=9, shared_control_branches=27,
        candidate_gate_passes=bool(legacy['accepted'] and primary['exact']['ci95'][0] > 0),
        contrasts={'candidate_vs_no_memory': primary,
            'placebo_vs_no_memory': _contrast(groups[ARMS[0]], groups[ARMS[2]]),
            'candidate_vs_placebo': _contrast(groups[ARMS[2]], groups[ARMS[1]])},
        raw_episode_sha256=[fingerprint(episode.to_dict()) for episode in episodes],
        comparison_view={**copy.deepcopy(COMPARISON_VIEW),
            'episode_sha256': [fingerprint(view.to_dict()) for view in views]},
        observed_step_cost={'steps': sum(len(episode.steps) for episode in episodes),
            'prompt_tokens': sum(step.prompt_tokens for episode in episodes for step in episode.steps),
            'completion_tokens': sum(step.completion_tokens for episode in episodes for step in episode.steps)},
        scope='All81 known/released is required. Primary candidate versus27 shared controls uses the '
            'unchanged legacy AND exact-positive gate; placebo comparisons are secondary. '
            'Observed step costs overlap returned-generation costs and are not full resource billing. '
            'This is exposed train utility metadata, not content certification, acceptance, '
            'training, causal attribution or unseen first-attempt transfer.')
    return result
