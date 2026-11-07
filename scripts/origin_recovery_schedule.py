"""Pure scheduling and accounting for the fixed eleven-task follow-up cohort.

Inputs are normalized, independently audited attempt summaries and external
release attestations. This module neither runs attempts nor verifies physical
resource release. A caller must persist each started batch before side effects.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from scripts import origin_recovery_cohort as cohort
from scripts import prepare_origin_recovery_pilot as preparation
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'artifacts/qwen38-origin-unattempted-design-001.json'
DESIGN_SHA256 = 'f8643987d4bc4ce7f5ef4e9e6f90762062ba99cdbdc3f961088d4ac7bd903ba0'
ENVIRONMENT_ID = '497c6d91a3764256d3b39c059849a1301c32f07717449f1ca75a9b8eb1a4b423'
GPU_JOB_SECONDS = 7200
ADMISSION_WAIT_SECONDS = 86400


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def frozen_ids():
    """Read only the frozen design and train-ID preparation, never task text."""
    _require(not DESIGN.is_symlink() and DESIGN.is_file(), 'frozen design is missing or unsafe')
    raw = DESIGN.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == DESIGN_SHA256, 'frozen design hash changed')
    design = json.loads(raw)
    _require(design['schedule']['per_GPU_job_seconds'] == GPU_JOB_SECONDS
             and design['schedule']['per_job_admission_wait_seconds'] == ADMISSION_WAIT_SECONDS,
             'frozen job budgets differ')
    prep_path = Path(preparation.__file__)
    _require(hashlib.sha256(prep_path.read_bytes()).hexdigest() == design['selection']['component_sha256'],
             'ID preparation source changed')
    original = preparation.prepare()['selected_ids']
    ids = cohort.cohort_ids()
    _require(type(ids) is list and len(ids) == len(set(ids)) == 11
             and original == design['selection']['original_task_ids'] and original[1:] == ids
             and ids == design['selection']['task_ids'], 'frozen IDs differ')
    return ids


def _identity(state, ids):
    expected = {'design_sha256', 'prepared_manifest_sha256', 'model_id', 'backend_semantic_sha256',
                'task_sha256', 'source_batches', 'prefix_checks', 'branch_batches'}
    _require(type(state) is dict and set(state) == expected, 'invalid schedule state fields')
    _require(state['design_sha256'] == DESIGN_SHA256 and _sha(state['prepared_manifest_sha256'])
             and type(state['model_id']) is str and bool(state['model_id'])
             and _sha(state['backend_semantic_sha256']), 'invalid study identity')
    tasks = state['task_sha256']
    _require(type(tasks) is dict and set(tasks) == set(ids) and all(_sha(v) for v in tasks.values())
             and len(set(tasks.values())) == len(ids), 'invalid prepared task bindings')
    _require(all(type(state[key]) is list for key in ('source_batches', 'prefix_checks', 'branch_batches')),
             'schedule histories must be lists')


def _seed(seed, known):
    _require(type(seed) is dict and set(seed) == {'requested', 'effective', 'verified'}
             and type(seed['requested']) is int and seed['requested'] == 100
             and type(seed['verified']) is bool, 'invalid environment seed provenance')
    verified = seed['verified'] is True and type(seed['effective']) is int and seed['effective'] == 100
    unverified = seed['verified'] is False and seed['effective'] is None
    _require(verified or unverified, 'environment seed differs')
    _require(not known or verified, 'known result needs verified environment seed')


def _common(row, state, task_id, kind, *, started=False):
    _require(type(row) is dict and row.get('task_id') == task_id and row.get('kind') == kind,
             'attempt ID, kind or order differs')
    _require(row.get('prepared_manifest_sha256') == state['prepared_manifest_sha256']
             and row.get('model_id') == state['model_id'], 'attempt provenance differs')
    unknown = row.get('status') == 'unknown' and not started
    expected = {'task_sha256': state['task_sha256'][task_id],
                'backend_semantic_sha256': state['backend_semantic_sha256'], 'environment_id': ENVIRONMENT_ID}
    for key, value in expected.items():
        _require((unknown and row.get(key) is None) or row.get(key) == value, 'attempt provenance differs')
    _require((unknown and row.get('backend_sha256') is None) or _sha(row.get('backend_sha256')),
             'attempt provenance differs')
    _require((unknown and row.get('seed') is None) or (type(row.get('seed')) is int and row['seed'] == 17),
             'attempt provenance differs')
    if started:
        _require(row.get('audit_valid') is None and row.get('audit_sha256') is None,
                 'started attempt cannot claim an audit')
        _seed(row.get('environment_seed'), False)
        return
    _require(type(row.get('audit_valid')) is bool and _sha(row.get('audit_sha256')),
             'audited result and audit digest required')
    _require(row['audit_valid'] is (not unknown)
             and ('known_outcome' not in row or row['known_outcome'] is (not unknown)),
             'audit validity and outcome differ')
    _require(fingerprint({key: value for key, value in row.items() if key != 'audit_sha256'})
             == row['audit_sha256'], 'audit digest does not match report contents')
    if not unknown or row.get('environment_seed') is not None:
        _seed(row.get('environment_seed'), not unknown)


def _source(row, state, task_id):
    _common(row, state, task_id, 'source')
    status = row.get('status')
    _require(status in ('complete', 'unknown'), 'invalid source outcome status')
    if status == 'unknown':
        _require(row.get('success') is None and row.get('source_classification') in (None, 'unknown'),
                 'unknown source cannot be imputed')
        return
    _require(type(row.get('success')) is bool and _sha(row.get('episode_sha256')),
             'known source needs a boolean and episode digest')
    classification, k, prefix = row.get('source_classification'), row.get('prefix_length'), row.get('prefix_sha256')
    _require(type(k) is int and ((classification == 'no_error' and k == 0 and prefix is None)
             or (classification == 'no_budget' and k == 40 and _sha(prefix))
             or (classification == 'candidate' and 1 <= k < 40 and _sha(prefix))),
             'source error classification or prefix differs')


def _prefix(row, state, task_id, source):
    started = row.get('status') == 'started'
    _common(row, state, task_id, 'prefix_check', started=started)
    _require(row.get('source_sha256') == source['episode_sha256']
             and (row.get('prefix_sha256') == source['prefix_sha256']
                  or (row.get('status') == 'unknown' and row.get('prefix_sha256') is None)),
             'prefix source binding differs')
    status = row.get('status')
    _require(status in ('started', 'eligible', 'already_terminal', 'unknown'), 'invalid prefix status')
    if status in ('started', 'unknown'):
        _require(row.get('prefix_success') is None and row.get('prefix_completed') is None
                 and row.get('proof_sha256') is None, 'unfinished prefix cannot supply a proof')
        return
    _require(_sha(row.get('proof_sha256')) and type(row.get('prefix_success')) is bool
             and type(row.get('prefix_completed')) is bool, 'invalid terminal prefix proof')
    terminal = row['prefix_success'] or row['prefix_completed']
    _require((status == 'already_terminal') == terminal, 'prefix eligibility differs from terminal state')
    _full_replay(row, source)


def _branch(row, state, planned, source, proof):
    task_id, kind = planned['task_id'], planned['kind']
    _common(row, state, task_id, kind)
    _require(row.get('source_sha256') == source['episode_sha256']
             and (row.get('prefix_sha256') == source['prefix_sha256']
                  or (row.get('status') == 'unknown' and row.get('prefix_sha256') is None))
             and row.get('prefix_proof_sha256') == proof['proof_sha256'],
             'branch source or proof binding differs')
    status = row.get('status')
    _require(status in ('complete', 'unknown'), 'invalid branch outcome status')
    if status == 'unknown':
        _require(row.get('success') is None, 'unknown branch cannot be imputed')
    else:
        _require(type(row.get('success')) is bool and _sha(row.get('episode_sha256')),
                 'known branch needs a boolean and episode digest')
        if kind == 'hint':
            _require(type(row.get('targets')) is list, 'audited hint targets are required')
        _full_replay(row, source)


def _full_replay(row, source):
    if row.get('costs') is None or _cost_summary([row])['lower_bound_only']:
        return
    replay = row['costs']['replay']
    _require(replay['attempted'] == replay['returned'] == source['prefix_length'],
             'complete replay costs do not cover the source prefix')


def _batch_series(batches, kind, plans, validate):
    """Validate fixed started batches; return their rows and the first barrier."""
    _require(len(batches) <= len(plans), 'too many batches')
    rows, barrier = [], None
    for index, batch in enumerate(batches):
        _require(barrier is None, 'later batch follows an unfinished or unknown batch')
        _require(type(batch) is dict and batch.get('kind') == kind
                 and type(batch.get('batch_index')) is int and batch['batch_index'] == index
                 and batch.get('planned') == plans[index]
                 and batch.get('status') in ('started', 'finished')
                 and type(batch.get('attempts')) is list, 'batch identity or planned order differs')
        attempts = batch['attempts']
        _require(len(attempts) <= len(plans[index]), 'too many batch attempts')
        cost_unknown = False
        for offset, row in enumerate(attempts):
            validate(row, plans[index][offset])
            _require(row.get('status') != 'unknown' or offset == len(attempts) - 1,
                     'attempt followed an unknown result')
            cost_unknown = _cost_summary([row])['lower_bound_only']
            _require(not cost_unknown or offset == len(attempts) - 1,
                     'attempt followed unknown cost')
            rows.append(row)
        release = batch.get('release')
        if release is not None:
            _require(type(release) is dict and set(release) == {'verified', 'evidence_sha256'}
                     and type(release['verified']) is bool and _sha(release['evidence_sha256'])
                     and batch['status'] == 'finished', 'invalid external release attestation')
        if attempts and attempts[-1].get('status') == 'unknown':
            barrier = ('stop', 'attempt_unknown')
        elif cost_unknown:
            barrier = ('stop', 'attempt_cost_unknown')
        elif batch['status'] == 'started':
            barrier = ('wait', 'batch_in_progress')
        elif len(attempts) != len(plans[index]):
            barrier = ('stop', 'terminal_batch_missing_results')
        elif release is None:
            barrier = ('wait_release', 'external_release_attestation_pending')
        elif release['verified'] is not True:
            barrier = ('stop', 'external_release_unknown')
        elif batch.get('costs') is None:
            barrier = ('wait_costs', 'external_gpu_job_costs_pending')
        elif _batch_costs([batch])['lower_bound_only']:
            barrier = ('stop', 'external_gpu_job_cost_unknown')
        elif (batch['costs']['gpu_job_wall_seconds'] > GPU_JOB_SECONDS
              or batch['costs']['admission_wait_seconds'] > ADMISSION_WAIT_SECONDS):
            barrier = ('stop', 'job_budget_exceeded')
    return rows, barrier


def _item(kind, task_id):
    return {'kind': kind, 'task_id': task_id}


def _decision(action, reason='', *, attempts=None, batch_index=None):
    return {'action': action, 'reason': reason, 'attempts': attempts or [], 'batch_index': batch_index}


def _inspect(state):
    ids = frozen_ids()
    _identity(state, ids)
    source_plans = [[_item('source', name) for name in ids[i:i + 6]] for i in (0, 6)]
    sources, source_barrier = _batch_series(state['source_batches'], 'source', source_plans,
        lambda row, item: _source(row, state, item['task_id']))
    source_map = {row['task_id']: row for row in sources}
    if source_barrier is not None:
        _require(not state['prefix_checks'] and not state['branch_batches'], 'work followed a source barrier')
        return ids, sources, [], [], _decision(*source_barrier)
    if len(state['source_batches']) < 2:
        _require(not state['prefix_checks'] and not state['branch_batches'], 'prefix ran before sources')
        index = len(state['source_batches'])
        return ids, sources, [], [], _decision('launch_source_batch', attempts=source_plans[index],
                                               batch_index=index)

    candidates = [task_id for task_id in ids if source_map[task_id]['source_classification'] == 'candidate']
    checks = state['prefix_checks']
    _require(len(checks) <= len(candidates), 'too many prefix checks')
    for index, row in enumerate(checks):
        name = candidates[index]
        _prefix(row, state, name, source_map[name])
        if row['status'] in ('unknown', 'started'):
            _require(index == len(checks) - 1 and not state['branch_batches'],
                     'work followed an unfinished or unknown prefix')
            action = 'stop' if row['status'] == 'unknown' else 'wait'
            return ids, sources, checks, [], _decision(action, 'prefix_' + row['status'])
        if _cost_summary([row])['lower_bound_only']:
            _require(index == len(checks) - 1 and not state['branch_batches'],
                     'work followed unknown prefix cost')
            return ids, sources, checks, [], _decision('stop', 'attempt_cost_unknown')
    if len(checks) != len(candidates):
        _require(not state['branch_batches'], 'branches ran before all prefix checks')
        return ids, sources, checks, [], _decision('launch_prefix_check',
            attempts=[_item('prefix_check', candidates[len(checks)])])

    proof_map = {row['task_id']: row for row in checks}
    eligible = [name for name in candidates if proof_map[name]['status'] == 'eligible']
    slots = [_item(kind, name) for name in eligible for kind in ('control', 'hint')]
    branch_plans = [slots[i:i + 6] for i in range(0, len(slots), 6)]
    branches, branch_barrier = _batch_series(state['branch_batches'], 'branch', branch_plans,
        lambda row, item: _branch(row, state, item, source_map[item['task_id']], proof_map[item['task_id']]))
    if branch_barrier is not None:
        return ids, sources, checks, branches, _decision(*branch_barrier)
    if len(state['branch_batches']) < len(branch_plans):
        index = len(state['branch_batches'])
        return ids, sources, checks, branches, _decision('launch_branch_batch',
            attempts=branch_plans[index], batch_index=index)
    return ids, sources, checks, branches, _decision('complete')


def next_action(state):
    """Return a fixed next batch, one prefix check, wait, stop or complete."""
    return _inspect(state)[-1]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _count(value):
    return type(value) is int and value >= 0


def _cost_summary(rows):
    result = {
        'generation': {'attempted': 0, 'returned': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                       'unknown_additional_cost': False},
        'replay': {'attempted': 0, 'returned': 0, 'elapsed_seconds': 0.0, 'unknown_additional_cost': False},
        'new_execution': {'attempted': 0, 'returned': 0, 'elapsed_seconds': 0.0,
                          'unknown_additional_cost': False},
        'startup_seconds': 0.0, 'cleanup_seconds': 0.0, 'rollout_seconds': 0.0, 'wall_seconds': 0.0,
        'missing_cost_records': 0, 'unusable_cost_records': 0, 'lower_bound_only': False,
    }
    for row in rows:
        if row.get('audit_valid') is False:
            # Invalid journals can contain orphaned return events. Preserve the
            # original audit, count coherent observations only, and never infer
            # a completed call or an exact cost from malformed evidence.
            result['lower_bound_only'] = True
            for category in ('generation', 'replay', 'new_execution'):
                result[category]['unknown_additional_cost'] = True
            try:
                partial = _cost_summary([{**row, 'audit_valid': True}])
            except (ValueError, TypeError, KeyError, OverflowError):
                result['unusable_cost_records'] += 1
                continue
            for category in ('generation', 'replay', 'new_execution'):
                for key, value in partial[category].items():
                    if key != 'unknown_additional_cost':
                        result[category][key] += value
            for key in ('startup_seconds', 'cleanup_seconds', 'rollout_seconds', 'wall_seconds',
                        'missing_cost_records', 'unusable_cost_records'):
                result[key] += partial[key]
            continue
        cost = row.get('costs')
        if cost is None:
            result['missing_cost_records'] += 1
            result['lower_bound_only'] = True
            continue
        _require(type(cost) is dict and type(cost.get('lower_bound_only')) is bool, 'invalid attempt costs')
        result['lower_bound_only'] |= cost['lower_bound_only']
        for category in ('generation', 'replay', 'new_execution'):
            part = cost.get(category)
            keys = ('attempted', 'returned', 'prompt_tokens', 'completion_tokens') if category == 'generation' \
                else ('attempted', 'returned')
            _require(type(part) is dict and all(_count(part.get(k)) for k in keys)
                     and part['returned'] <= part['attempted']
                     and type(part.get('unknown_additional_cost')) is bool
                     and (part['attempted'] == part['returned'] or part['unknown_additional_cost']),
                     'invalid or imputed attempt costs')
            for key in keys:
                result[category][key] += part[key]
            if category != 'generation':
                _require(_number(part.get('elapsed_seconds')), 'invalid execution time')
                result[category]['elapsed_seconds'] += part['elapsed_seconds']
            result[category]['unknown_additional_cost'] |= part['unknown_additional_cost']
            result['lower_bound_only'] |= part['unknown_additional_cost']
        gen = cost['generation']
        _require(gen['returned'] > 0 or (gen['prompt_tokens'] == 0 and gen['completion_tokens'] == 0),
                 'generation tokens require a returned response')
        if row.get('kind') == 'source':
            _require(cost['replay']['attempted'] == 0 and cost['replay']['elapsed_seconds'] == 0,
                     'source cannot charge replay')
        if row.get('kind') == 'prefix_check':
            _require(gen['attempted'] == 0 and cost['new_execution']['attempted'] == 0
                     and cost['new_execution']['elapsed_seconds'] == 0,
                     'CPU prefix cannot generate or perform new actions')
        if row.get('status') == 'complete' and not result['lower_bound_only']:
            _require(cost['new_execution']['attempted'] == gen['returned'],
                     'complete generated responses and new actions do not match')
        for key in ('startup_seconds', 'cleanup_seconds', 'rollout_seconds', 'wall_seconds'):
            value = cost.get(key)
            if value is None:
                result['lower_bound_only'] = True
            else:
                _require(_number(value), 'invalid attempt wall time')
                result[key] += value
    return result


def _batch_costs(batches):
    keys = ('gpu_job_wall_seconds', 'admission_wait_seconds', 'model_loading_seconds', 'backend_wait_seconds')
    result = {key: 0.0 for key in keys}
    missing = 0
    for batch in batches:
        cost = batch.get('costs')
        if cost is None:
            missing += 1
            continue
        _require(type(cost) is dict and type(cost.get('lower_bound_only')) is bool, 'invalid GPU job costs')
        incomplete = cost['lower_bound_only']
        for key in keys:
            value = cost.get(key)
            if value is None:
                incomplete = True
            else:
                _require(_number(value), 'invalid GPU job time')
                result[key] += value
        missing += int(incomplete)
    result.update(missing_or_lower_bound_jobs=missing, lower_bound_only=missing > 0)
    return result


def summarize(state, targets):
    """Count twelve fixed outcomes and conservative help-target yield.

    Target summaries must exactly match the ordered target lists in completed
    independent hint audits, whose self-digests are checked above. The target
    digest fingerprints action text alone. No task text or trajectory is read.
    Missing cost evidence is reported as a lower bound.
    """
    ids, sources, checks, branches, decision = _inspect(state)
    _require(type(targets) is list, 'target summaries must be a list')
    source_map = {row['task_id']: row for row in sources}
    proof_map = {row['task_id']: row for row in checks}
    branch_map = {(row['task_id'], row['kind']): row for row in branches}
    audited_targets = [target for row in branches if row['kind'] == 'hint' and row['status'] == 'complete'
                       for target in row['targets']]
    _require(targets == audited_targets, 'targets differ from audited hint targets')
    classes = {name: 0 for name in ('help', 'harm', 'tie_success', 'tie_failure',
                                    'unknown', 'ineligible', 'unattempted')}
    rows = []
    for name in ids:
        source = source_map.get(name)
        proof = proof_map.get(name)
        control, hint = branch_map.get((name, 'control')), branch_map.get((name, 'hint'))
        if source is None:
            result = 'unattempted'
        elif source['status'] == 'unknown':
            result = 'unknown'
        elif source['source_classification'] != 'candidate':
            result = 'ineligible'
        elif proof is None:
            result = 'unattempted'
        elif proof['status'] in ('unknown', 'started'):
            result = 'unknown'
        elif proof['status'] == 'already_terminal':
            result = 'ineligible'
        elif (control and control['status'] == 'unknown') or (hint and hint['status'] == 'unknown'):
            result = 'unknown'
        elif control is None or hint is None:
            result = 'unattempted'
        elif not control['success'] and hint['success']:
            result = 'help'
        elif control['success'] and not hint['success']:
            result = 'harm'
        else:
            result = 'tie_success' if control['success'] else 'tie_failure'
        classes[result] += 1
        rows.append({'task_id': name, 'pair_class': result})

    class_by_id = {row['task_id']: row['pair_class'] for row in rows}
    unique_steps, unique_targets = set(), set()
    accepted = 0
    flags = ('messages_reconstructed', 'full_history', 'nonempty', 'no_error', 'raw_action_match', 'eligible')
    required = {'task_id', 'episode_sha256', 'step_index', 'target_sha256',
                'student_messages_sha256', 'sample_sha256', *flags}
    for target in targets:
        _require(type(target) is dict and set(target) == required
                 and all(type(target[key]) is bool for key in flags), 'invalid target summary')
        name = target['task_id']
        hint = branch_map.get((name, 'hint'))
        _require(name in ids and hint is not None and hint['status'] == 'complete'
                 and target['episode_sha256'] == hint['episode_sha256']
                 and _sha(target['target_sha256']) and _sha(target['student_messages_sha256'])
                 and target['sample_sha256'] == fingerprint({
                     'student_messages_sha256': target['student_messages_sha256'],
                     'target_sha256': target['target_sha256']})
                 and type(target['step_index']) is int
                 and 0 <= target['step_index'] < 40
                 and _sha(target['sample_sha256']),
                 'target identity or episode binding differs')
        step_key = (target['episode_sha256'], target['step_index'])
        _require(step_key not in unique_steps, 'duplicate target step')
        unique_steps.add(step_key)
        if target['eligible']:
            _require(all(target[key] for key in flags[:-1])
                     and target['step_index'] >= source_map[name]['prefix_length'],
                     'eligible target lacks full-history, non-replay or action evidence')
            if class_by_id[name] == 'help':
                accepted += 1
                unique_targets.add(target['target_sha256'])
    scenarios = len({name.rsplit('_', 1)[0] for name in ids if class_by_id[name] == 'help'})
    complete = decision['action'] == 'complete' and not classes['unknown'] and not classes['unattempted']
    gate = ('not_evaluable' if not complete else
            'sufficient' if scenarios >= 4 and len(unique_targets) >= 20 else 'insufficient')
    jobs = _batch_costs(state['source_batches'] + state['branch_batches'])
    return {
        'fixed_source_denominator': len(ids), 'pair_classes': classes, 'pairs': rows,
        'next_action': decision,
        'gate': {'status': gate, 'distinct_help_scenarios': scenarios,
                 'distinct_eligible_targets': len(unique_targets), 'eligible_target_rows': accepted,
                 'automatic_training': False,
                 'interpretation': 'Data feasibility only; not efficacy, transfer or accepted experience.'},
        'costs': {'source': _cost_summary(sources), 'prefix_check': _cost_summary(checks),
                  'branch': _cost_summary(branches), 'gpu_jobs': jobs,
                  'gpu_job_costs_incomplete': jobs['lower_bound_only'],
                  'note': 'Caller-audited lower bounds; wall, rollout and subcomponent times overlap. '
                          'Unusable cost records are excluded and remain in their original audit. '
                          'Do not add historical replayed step tokens or double-count returned generation.'},
        'evidence_scope': 'Release and attempt attestations are caller-supplied; this function does not '
                          'verify physical GPU, process or container release.',
    }
