"""One exclusive, no-retry parent for a future 27-pair CodeAct utility study.

Closed batches are independently authenticated and re-read before analysis.
Import does not load private data, register, start a task, model, queue or GPU.
"""

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts.codeact_fault_sources_001 import validate_fault
from wll.prompts import TUTORIAL

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-within-utility-001'
OUTPUT = ROOT / 'artifacts' / ID
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-utility-001')
SOURCE = Path(__file__).with_name('run_codeact_within_source_study_001.py')
SOURCE_SHA256 = '129128bd787e4785934e977d98f3faf8256cc52f7d13840fc579b4332d6dd25c'
ERROR = 'codeact_utility_study_invalid'
_COUNTS = ('generation_attempts', 'generation_returns', 'prompt_tokens', 'completion_tokens',
           'actions', 'api_request_dispatches', 'budget_rejections')
_HELPERS = None


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _identity(who):
    _require(type(who) is dict and set(who) == {'pid', 'start_ticks'}
             and all(type(value) is int and value > 0 for value in who.values()))
    return who


def _validated_fault(value, counts):
    _require(type(counts) is dict and type(counts.get('api_request_dispatches')) is int
             and 0 <= counts['api_request_dispatches'] <= 500)
    return copy.deepcopy(validate_fault(value, counts['api_request_dispatches']))


def _optional_fault(row, outcome, bound):
    if not bound:
        return None
    try:
        return _validated_fault(row.get('fault_provenance'), outcome['counts'])
    except Exception:
        return None


def _planned(plan, index):
    _require(type(index) is int and 0 <= index < 18)
    group = plan['slots'][index * 3:index * 3 + 3]
    _require(type(group) is list and len(group) == 3
             and all(type(slot) is dict and set(slot) == {'task_id', 'arm', 'pair_index'}
                     and type(slot['task_id']) is str and slot['arm'] in {'control', 'experience'}
                     and type(slot['pair_index']) is int and 0 <= slot['pair_index'] < 27 for slot in group)
             and [slot['task_id'] for slot in group] == plan['batches'][index]
             and len({slot['task_id'] for slot in group}) == 3)
    return [{'task_id': slot['task_id'], 'arm': slot['arm'], 'phase': 'utility'} for slot in group]


def _isolated_helpers(*, raw=None):
    raw = SOURCE.read_bytes() if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256)
    source = raw.decode('utf-8')
    tree = ast.parse(source)
    names = ('_inspect_batch', '_preserve_partial', '_costs')
    found = {node.name: ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    _require(set(found) == set(names) and all(type(found[name]) is str for name in names))
    text = '\n\n'.join(found[name] for name in names)
    changes = (
        ("    _require(c.verify_batch_report(stage, plan, index, sha) is True)\n",
         "    _, verified_report_sha = c._read(stage / 'run/report.json', maximum=128_000_000)\n"
         "    _require(c.verify_batch_report(stage, plan, index, sha) is True)\n", 1),
        ("    report, report_sha = read(stage / 'run/report.json')\n",
         "    report, report_sha = read(stage / 'run/report.json', expected=verified_report_sha)\n", 1),
        ("item['arm'] == 'first_attempt' and _same(item['intent'], intent['intent'])",
         "item['arm'] == plan['slots'][index * 3 + offset]['arm'] and intent.get('arm') == item['arm'] "
         "and intent.get('phase') == 'utility' and _same(item['intent'], intent['intent'])", 1),
        ("row['arm'] == 'first_attempt' and row['phase'] == 'source'",
         "row['arm'] == plan['slots'][source_index]['arm'] and row['phase'] == 'utility'", 1),
        ("planned = [dict(task_id=t, arm='first_attempt', phase='source') for t in plan['batches'][index]]",
         'planned = _planned(plan, index)', 1),
        ("row.get('task_id') in plan['batches'][index] for n, row in enumerate(event_rows)",
         "row.get('task_id') in plan['batches'][index] and row.get('phase') == 'utility' "
         "and row.get('arm') == next(s['arm'] for s in planned if s['task_id'] == row.get('task_id')) "
         'for n, row in enumerate(event_rows)', 1),
        ("c._verify_private(values['actor_trace'], values['wire_trace'], episode, request['expected_backend'])",
         "c._verify_private(values['actor_trace'], values['wire_trace'], episode, request['expected_backend'], "
         "deps.registration.load_experience(plan) if planned[offset]['arm'] == 'experience' else None)", 1),
        ("public.append(dict(source_index=source_index, task_id=row['task_id'], batch_index=index,",
         "public.append(dict(slot_index=source_index, task_id=row['task_id'], arm=row['arm'], "
         "pair_index=plan['slots'][source_index]['pair_index'], batch_index=index,", 1),
        ("        owner = row['owner']\n", "        fault = _validated_fault(row['fault_provenance'], outcome['counts'])\n"
         "        owner = row['owner']\n", 1),
        ("phase_seconds=copy.deepcopy(owner['phase_seconds'])))",
         "phase_seconds=copy.deepcopy(owner['phase_seconds']), worker_exit_code=owner['creator_exit_code'], "
         "owner_cleanup_verified=owner['cleanup_verified'], owner_release_verified=True, "
         'task_context_sha256=task_context_sha256, fault_provenance=fault))', 1),
        ("for key, filename in (('actor_trace', f'actor-{offset:02}.jsonl'), ('wire_trace', f'wire-{offset:02}.jsonl')):",
         "initial = episode['private']['steps'][0]['messages_before'] if episode['private']['steps'] "
         "else episode['private']['messages']\n        "
         "_require(type(initial) is list and len(initial) >= _INITIAL_MESSAGES)\n        "
         "_require((row['status'] == 'scored' and type(outcome['official_success']) is bool) or "
         "(row['status'] == 'known_unscored' and outcome['official_success'] is None))\n        "
         "task_context_sha256 = hashlib.sha256(json.dumps(initial[:_INITIAL_MESSAGES], sort_keys=True, "
         "ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()\n        "
         "for key, filename in (('actor_trace', f'actor-{offset:02}.jsonl'), ('wire_trace', f'wire-{offset:02}.jsonl')):", 1),
        ("observed_unverified_costs=observed)",
         "observed_unverified_costs=observed, "
         "worker_exit_code=row['owner']['creator_exit_code'] if bound else None, "
         "reported_owner_cleanup_verified=row['owner']['cleanup_verified'] if bound else None, "
         "fault_provenance=_optional_fault(row, outcome, bound))", 1),
        ("report['sources']", "report['slots']", 3),
        ('full = len(rows) == 12 and len(batches) == 4',
         'full = len(rows) == 54 and len(batches) == 18', 1),
    )
    for old, new, count in changes:
        _require(text.count(old) == count)
        text = text.replace(old, new)
    module = ModuleType('scripts._codeact_within_utility_parent_helpers')
    module.__dict__.update(copy=copy, json=json, hashlib=hashlib, ID=ID, OUTPUT=OUTPUT, PRIVATE=PRIVATE,
        _require=_require, _same=_same, _number=_number, _identity=_identity, _planned=_planned,
        _validated_fault=_validated_fault, _optional_fault=_optional_fault,
        _INITIAL_MESSAGES=len(TUTORIAL) + 2, _COUNTS=_COUNTS)
    # This compiles only three SHA-bound human-authored parent helpers.
    exec(compile(text, str(SOURCE), 'exec', dont_inherit=True), module.__dict__)
    return module


def _helpers():
    global _HELPERS
    if _HELPERS is None:
        _HELPERS = _isolated_helpers()
    _HELPERS.OUTPUT, _HELPERS.PRIVATE, _HELPERS.ID = OUTPUT, PRIVATE, ID
    return _HELPERS


def _inspect_batch(deps, plan, sha, index, result, parent):
    return _helpers()._inspect_batch(deps, plan, sha, index, result, parent)


def _preserve_partial(deps, plan, sha, index, report):
    return _helpers()._preserve_partial(deps, plan, sha, index, report)


def _costs(report):
    return _helpers()._costs(report)


def _dependencies():
    from scripts import codeact_within_utility_registration_001 as registration
    from scripts import run_codeact_within_utility_batch_001 as coordinator
    from scripts.codeact_within_utility_001 import evaluate_utility
    return SimpleNamespace(registration=registration, coordinator=coordinator, evaluate_utility=evaluate_utility)


def _validate_plan(plan, reg):
    _require(type(plan) is dict and plan.get('id') == ID and plan.get('private_directory') == str(PRIVATE)
             and _same(plan.get('task_ids'), list(reg.TASK_IDS)) and len(reg.TASK_IDS) == 54
             and _same(plan.get('batches'), reg.BATCHES) and len(reg.BATCHES) == 18
             and _same(plan.get('slots'), reg.SLOTS) and len(reg.SLOTS) == 54
             and _same(plan.get('expected_pairs'), reg.EXPECTED_PAIRS) and len(reg.EXPECTED_PAIRS) == 27
             and type(reg.SOURCE_SCENARIOS) in (tuple, list) and len(set(reg.SOURCE_SCENARIOS)) == 12)
    identity = plan.get('identity')
    _require(type(identity) is dict and set(identity) == {'model_id', 'scaffold_id', 'backend_id', 'budget_id'}
             and all(type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}', value)
                     for value in identity.values()) and identity['model_id'] == plan.get('model_id')
             and _same(identity, reg.expected_identity(plan))
             and type(plan.get('experience_bundle_id')) is str
             and re.fullmatch(r'codeact-experience-001:[a-f0-9]{64}', plan['experience_bundle_id']))
    pairs = plan['expected_pairs']
    _require(all(type(pair) is list and len(pair) == 3 and type(pair[0]) is str and type(pair[1]) is str
                 and type(pair[2]) is int and pair[2] == 17 for pair in pairs)
             and len({pair[0] for pair in pairs}) == 27 and len({pair[1] for pair in pairs}) == 9
             and not set(reg.SOURCE_SCENARIOS) & {pair[1] for pair in pairs})
    seen = set()
    for index in range(18):
        _planned(plan, index)
        for slot in plan['slots'][index * 3:index * 3 + 3]:
            key = slot['pair_index'], slot['arm']
            _require(slot['task_id'] == pairs[key[0]][0] and key not in seen)
            seen.add(key)
    _require(seen == {(index, arm) for index in range(27) for arm in ('control', 'experience')})


def _empty_utility(reason):
    return {'status': 'not_evaluable', 'reason': reason, 'planned_pairs': 27, 'planned_summaries': 54,
            'passed': None, 'checks': None, 'statistics': None}


def _paired_summaries(plan, rows):
    summaries = []
    for row in rows:
        pair = plan['expected_pairs'][row['pair_index']]
        summaries.append({'task_id': row['task_id'], 'scenario_id': pair[1], 'seed': pair[2],
            'arm': row['arm'], 'split': 'train', 'status': 'known', 'closed': True, 'branch': 'first',
            'replayed_steps': 0, 'success': row['official_success'], 'outcome_unknown': False,
            'unknown_cost': False, 'is_policy_episode': True, 'worker_exit_code': row['worker_exit_code'],
            'score_status': 'scored', 'owner_cleanup_verified': row['owner_cleanup_verified'],
            'owner_release_verified': row['owner_release_verified'], 'identity': copy.deepcopy(plan['identity']),
            'experience_bundle_id': plan['experience_bundle_id'] if row['arm'] == 'experience' else None})
    return summaries


def _validate_utility(value, plan, rows, origins):
    fields = {'comparison', 'experience_bundle_id', 'cohort_sha256', 'source_scenarios_sha256', 'identity',
              'planned_pairs', 'planned_scenarios', 'planned_summaries', 'observed_summaries',
              'known_summaries', 'unknown_summaries', 'incomplete_summaries', 'unstarted_summaries',
              'missing_summaries', 'complete_pairs', 'thresholds', 'status', 'reason', 'passed',
              'checks', 'statistics', 'limits'}
    _require(type(value) is dict and set(value) == fields and len(rows) == 54
             and value['comparison'] == 'ordinary_experience_vs_no_experience_train_only'
             and value['experience_bundle_id'] == plan['experience_bundle_id']
             and _same(value['identity'], plan['identity']) and type(value['limits']) is str)
    def digest(item):
        return hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    _require(value['cohort_sha256'] == digest(sorted(plan['expected_pairs']))
             and value['source_scenarios_sha256'] == digest(sorted(origins))
             and all(type(value[key]) is int for key in ('planned_pairs', 'planned_scenarios',
                 'planned_summaries', 'observed_summaries', 'known_summaries', 'unknown_summaries',
                 'incomplete_summaries', 'unstarted_summaries', 'missing_summaries', 'complete_pairs'))
             and (value['planned_pairs'], value['planned_scenarios'], value['planned_summaries'],
                  value['observed_summaries']) == (27, 9, 54, 54)
             and value['unknown_summaries'] == value['unstarted_summaries'] == value['missing_summaries'] == 0)
    known = [row for row in rows if row['worker_exit_code'] in (0, -9)]
    complete_pairs = sum(sum(row['pair_index'] == index for row in known) == 2 for index in range(27))
    _require(value['known_summaries'] == len(known) and value['incomplete_summaries'] == 54 - len(known)
             and value['complete_pairs'] == complete_pairs
             and _same(value['thresholds'], {'min_scenarios': 8, 'max_observed_harm_rate': 0.05,
                 'bootstrap_samples': 2000, 'bootstrap_seed': 0, 'both_lower_bounds_strictly_positive': True}))
    if value['status'] == 'complete':
        checks, stats = value['checks'], value['statistics']
        _require(len(known) == 54 and complete_pairs == 27 and value['reason'] is None
                 and type(value['passed']) is bool and type(checks) is dict
                 and set(checks) == {'at_least_8_scenarios', 'bootstrap_lower_positive',
                                     'exact_cluster_lower_positive', 'observed_harm_at_most_5_percent'}
                 and all(type(flag) is bool for flag in checks.values())
                 and value['passed'] is all(checks.values()) and type(stats) is dict
                 and type(stats.get('n_pairs')) is int and stats['n_pairs'] == 27
                 and type(stats.get('scenario_count')) is int and stats['scenario_count'] == 9
                 and all(type(stats.get(key)) is int and stats[key] >= 0 for key in ('helped', 'harmed', 'ties'))
                 and sum(stats[key] for key in ('helped', 'harmed', 'ties')) == 27)
    else:
        _require(value['status'] == 'not_evaluable' and value['reason'] in {
            'incomplete_cohort', 'exact_resampling_unavailable'}
            and value['passed'] is value['checks'] is value['statistics'] is None)
    return value


def run_study(sha, *, _deps=None):
    _require(type(sha) is str and re.fullmatch('[a-f0-9]{64}', sha) is not None)
    deps = _deps or _dependencies()
    reg, c = deps.registration, deps.coordinator
    plan = reg.load_registration(ROOT, sha, private=True)
    _require(reg.check_predecessor(ROOT, plan) is True)
    _validate_plan(plan, reg)
    _helpers()  # Fail closed on source/helper drift before reserving any study output.
    core = c._lifecycle()
    preflight = core.remote({'operation': 'preflight', 'sha': sha})
    _require(type(preflight) is dict and preflight.get('registration') == 'passed'
             and preflight.get('monitor_live') is True)
    lock_fd = os.open(ROOT / 'artifacts/foundation-pilot-001.lock',
                      os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(lock_fd, 'a+') as lock:
        info = os.fstat(lock.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                 and stat.S_IMODE(info.st_mode) & 0o022 == 0)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core.validate_directory(PRIVATE)
        c.io.reserve(OUTPUT)
        parent = _identity(core.identity(os.getpid()))
        intent = dict(id=ID, plan_sha256=sha, parent=parent, started_epoch=time.time(), automatic_retry=False)
        intent_sha = c._write_private(PRIVATE / 'study-intent.json', intent)
        report = dict(id=ID, plan_sha256=sha, status='running', parent=parent,
            study_intent_sha256=intent_sha, started_epoch=intent['started_epoch'], fixed_denominator=54,
            planned_pairs=27, batches=[dict(batch_index=i, status='unattempted', accepted=None,
                auth_complete=False, gpu_costs=None, release=None) for i in range(18)],
            slots=[dict(slot_index=i, task_id=slot['task_id'], arm=slot['arm'], pair_index=slot['pair_index'],
                batch_index=i // 3, status='unattempted', attempted=False, auth_complete=False,
                model_reported_complete=None, official_success=None, costs_known=None, counts=None,
                actor_wall_seconds=None, world_wall_seconds=None, phase_seconds=None, worker_exit_code=None,
                owner_cleanup_verified=None, owner_release_verified=None, task_context_sha256=None,
                fault_provenance=None)
                for i, slot in enumerate(plan['slots'])], unattempted_batches=list(range(18)),
            automatic_retry=False, utility_evaluated=False, utility=None)
        journal_fd = os.open(OUTPUT / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(journal_fd, 'w', encoding='ascii') as journal:
            sequence = 0

            def save(event, **fields):
                nonlocal sequence
                value = dict(event=event, event_index=sequence, epoch=time.time(), plan_sha256=sha, **fields)
                journal.write(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                sequence += 1
                report['costs'] = _costs(report)
                c.io.checkpoint(OUTPUT / 'report.json', report)

            results, first_batches = [], []
            save('study_start', parent=parent)
            for index in range(18):
                report['unattempted_batches'].remove(index)
                report['batches'][index]['status'] = 'running'
                for row in report['slots'][index * 3:index * 3 + 3]:
                    row.update(status='unknown', attempted=None)
                save('batch_start', batch_index=index)
                try:
                    result = c.run_stage(plan, sha, index)
                    _require(type(result) is dict and type(result.get('accepted')) is bool)
                    results.append(result)
                    _require(result['accepted'] is True)
                    _, public, batch = _inspect_batch(deps, plan, sha, index, result, parent)
                    report['slots'][index * 3:index * 3 + 3], report['batches'][index] = public, batch
                    first_batches.append(copy.deepcopy(batch))
                except (Exception, KeyboardInterrupt, SystemExit) as exc:
                    report['status'] = ('interrupted_no_retry' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                        else 'incomplete_no_retry')
                    report['batches'][index].update(status='unknown', accepted=False, auth_complete=False)
                    _preserve_partial(deps, plan, sha, index, report)
                    save('batch_unknown', batch_index=index)
                    break
                save('batch_complete', batch_index=index)
            if all(batch['status'] == 'complete' for batch in report['batches']):
                try:
                    for index, result in enumerate(results):
                        report['batches'][index].update(status='unknown', accepted=False, auth_complete=False)
                        for row in report['slots'][index * 3:index * 3 + 3]:
                            row.update(status='unknown', auth_complete=False, costs_known=None,
                                       official_success=None, model_reported_complete=None, fault_provenance=None)
                        _, public, batch = _inspect_batch(deps, plan, sha, index, result, parent)
                        _require(_same(batch, first_batches[index]))
                        report['slots'][index * 3:index * 3 + 3], report['batches'][index] = public, batch
                    _require(_costs(report)['all_costs_known'] is True)
                    for pair_index in range(27):
                        paired = [row for row in report['slots'] if row['pair_index'] == pair_index]
                        _require(len(paired) == 2 and {row['arm'] for row in paired} == {'control', 'experience'}
                                 and paired[0]['task_context_sha256'] == paired[1]['task_context_sha256'])
                    if all(type(row['official_success']) is bool for row in report['slots']):
                        result = deps.evaluate_utility(plan['expected_pairs'], _paired_summaries(plan, report['slots']),
                            identity=plan['identity'], source_scenario_ids=reg.SOURCE_SCENARIOS,
                            experience_bundle_id=plan['experience_bundle_id'])
                        report['utility'] = _validate_utility(result, plan, report['slots'], reg.SOURCE_SCENARIOS)
                        report['utility_evaluated'] = True
                    else:
                        report['utility'] = _empty_utility('unscored_or_unknown')
                    report['status'] = 'complete'
                except (Exception, KeyboardInterrupt, SystemExit) as exc:
                    report['status'] = ('interrupted_no_retry' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                        else 'incomplete_no_retry')
            if report['utility'] is None:
                report['utility'] = _empty_utility('incomplete_evidence')
            report['finished_epoch'] = time.time()
            save('study_stop', status=report['status'])
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        try:
            report = run_study(args.plan_sha256)
            result = dict(id=ID, status=report['status'], fixed_denominator=54)
            code = 0 if report['status'] == 'complete' else 1
        except Exception:
            result, code = dict(id=ID, status='failed', error_code=ERROR), 1
        print(json.dumps(result, sort_keys=True))
        return code
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
