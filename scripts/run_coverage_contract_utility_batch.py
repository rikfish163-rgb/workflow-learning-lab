"""One fixed private utility batch, with injected execution and independent audit.

Prior records must already be authenticated by the caller. This component owns
only schedule admission and write-ahead batch records, not model services, GPU
admission, task loading or fresh physical release verification.
"""

import copy
from dataclasses import asdict

from scripts import recovery_evidence_utility as utility
from scripts.completion_trace import TraceWriteError, _exception_type
from scripts.run_recovery_evidence_utility_branch import _reserve, _write_private
from wll.experience import Experience
from wll.protocol import fingerprint

_INTERRUPTS = (KeyboardInterrupt, SystemExit, GeneratorExit)
_COUNTS = ('generation_attempts', 'generation_returns', 'generation_returns_with_unusable_usage',
    'context_count_attempts', 'context_count_returns', 'context_rejections', 'execution_attempts',
    'execution_returns', 'evaluation_attempts', 'evaluation_returns', 'generation_calls_without_response')
_FLAGS = ('unknown_additional_response_cost', 'unknown_additional_context_count_cost', 'unknown_additional_cost')
_USAGE = ('prompt_tokens', 'completion_tokens')


def _require(condition):
    if not condition:
        raise ValueError('coverage_contract_utility_batch_invalid')


def _cost(value):
    """Project numeric observations only; absent/malformed fields remain unknown."""
    value = value if type(value) is dict else {}
    incomplete = False
    safe = {}
    for key in _COUNTS:
        valid = type(value.get(key)) is int and value[key] >= 0
        safe[key] = value[key] if valid else 0
        incomplete |= not valid
    safe['generation_usage_observed'] = {}
    usage = value.get('generation_usage_observed', {})
    for key in _USAGE:
        observed = usage.get(key) if type(usage) is dict else None
        valid = type(observed) is int and observed >= 0
        safe['generation_usage_observed'][key] = observed if valid else 0
        incomplete |= not valid
    for key in _FLAGS:
        valid = type(value.get(key)) is bool
        safe[key] = value[key] if valid else True
        incomplete |= not valid
    api = value.get('api_attempts_observed')
    safe['api_attempts_observed'] = api if type(api) is int and api >= 0 else None
    incomplete |= 'api_attempts_observed' not in value or (api is not None and safe['api_attempts_observed'] is None)
    safe['unknown_additional_cost'] |= incomplete
    return safe, incomplete


def _total(costs):
    total = {key: sum(row[key] for row in costs) for key in _COUNTS}
    total['generation_usage_observed'] = {
        key: sum(row['generation_usage_observed'][key] for row in costs) for key in _USAGE}
    total.update({key: any(row[key] for row in costs) for key in _FLAGS})
    observed = [row['api_attempts_observed'] for row in costs if row['api_attempts_observed'] is not None]
    total['api_attempts_observed'] = sum(observed) if observed or not costs else None
    total['api_counts_missing'] = len(costs) - len(observed)
    return total


def _files(value):
    if type(value) is not dict:
        return {}
    return {name: {'sha256': item['sha256'], 'bytes': item['bytes']}
        for name, item in value.items() if type(name) is str and name in {'actor.jsonl', 'episode.json', 'report.json'}
        and type(item) is dict and utility._sha(item.get('sha256'))
        and type(item.get('bytes')) is int and item['bytes'] >= 0}


def run_batch(partition, tasks, candidate, placebo, protocol, prior_records, *, phase,
              batch_index, output, run_one, audit_one, emit):
    """Execute exactly one admitted batch; never retry a callback or output root.

    ``run_one(task, experience, *, arm, protocol, output, emit)`` must use the
    frozen branch and emit only its public metadata. ``audit_one(directory,
    task, experience, protocol, *, arm, expected_public_events)`` must independently
    audit those files; bind its expected backend with a closure/partial. ``emit``
    must durably write before returning. Per-branch events are wrapped as
    ``branch_event`` with an unmodified inner event dictionary for later audits.
    A caller must authenticate the whole prior batch, including its terminal
    journal status, before passing its records into a later batch.
    """
    utility._protocol(partition, tasks, protocol)
    _require(protocol['design'] == utility.COVERAGE_CONTRACT_DESIGN)
    utility._validate_protocol_provenance(protocol, candidate)
    _require(isinstance(candidate, Experience) and isinstance(placebo, Experience)
        and fingerprint(asdict(placebo)) == protocol.get('placebo_sha256')
        and placebo.experience_id == protocol.get('placebo_id')
        and placebo.source_scenarios == [] and placebo.source_episode_hashes == [])
    utility._validate_placebo_provenance(protocol.get('placebo_provenance'), protocol['design'], placebo=placebo)
    _require(all(callable(value) for value in (run_one, audit_one, emit)))
    batches = utility.batch_schedules(phase)
    _require(type(batch_index) is int and 0 <= batch_index < len(batches))
    start = (0 if phase == 'controls' else 27) + sum(len(row) for row in batches[:batch_index])
    records = copy.deepcopy(prior_records)
    decision = utility.phase_decision(partition, tasks, protocol, records)
    allowed = {'awaiting_controls'} if phase == 'controls' else {'ready_for_prompted', 'awaiting_prompted'}
    _require(len(records) == start and decision['status'] in allowed)
    protocol = copy.deepcopy(protocol)
    schedule, protocol_sha = batches[batch_index], fingerprint(protocol)
    output = _reserve(output)
    costs, private_files = [], {}
    attempted, audited, intents = 0, 0, 0
    journal_failed, error_type, interrupted = False, None, None

    def remember(exc):
        nonlocal journal_failed, error_type, interrupted
        error_type = error_type or _exception_type(exc)
        journal_failed |= isinstance(exc, TraceWriteError)
        if interrupted is None and isinstance(exc, _INTERRUPTS):
            interrupted = exc

    def public(event, **fields):
        nonlocal journal_failed
        try:
            emit(event, **copy.deepcopy(fields))
        except BaseException as exc:
            journal_failed = True
            if isinstance(exc, _INTERRUPTS):
                raise
            raise TraceWriteError('utility_batch_public_write_failed') from None

    def unknown(index, released=False):
        name, arm = utility.branch_schedule()[index]
        return {'task_id': name, 'scenario_id': name.rsplit('_', 1)[0], 'arm': arm,
            'outcome': None, 'valid_for_continuation': False, 'released': released,
            'protocol_sha256': protocol_sha}

    try:
        private_files['manifest.json'] = _write_private(output / 'manifest.json', {
            'phase': phase, 'batch_index': batch_index, 'start_global_index': start,
            'schedule': [list(pair) for pair in schedule], 'protocol_sha256': protocol_sha,
            'partition_sha256': fingerprint(partition), 'task_sha256': protocol['task_sha256'],
            'candidate_sha256': protocol['candidate_sha256'], 'placebo_sha256': protocol['placebo_sha256'],
            'prior_records': records, 'prior_records_sha256': fingerprint(records),
            'maximum_attempts_per_branch': 1, 'automatic_retry': False, 'training_authorized': False})
        public('batch_started', phase=phase, batch_index=batch_index,
               start_global_index=start, scheduled_branches=len(schedule), protocol_sha256=protocol_sha)
        by_id = {task.task_id: task for task in tasks}
        for offset, (name, arm) in enumerate(schedule):
            index = start + offset
            decision = utility.phase_decision(partition, tasks, protocol, records)
            _require(decision['status'] in allowed and len(records) == index
                     and utility.branch_schedule()[index] == (name, arm))
            directory = output / f'branch-{index:03d}'
            intent = {'global_index': index, 'task_id': name, 'arm': arm,
                'protocol_sha256': protocol_sha, 'prior_records_sha256': fingerprint(records)}
            private_files[f'attempt-{index:03d}.json'] = _write_private(
                output / f'attempt-{index:03d}.json', intent)
            intents += 1
            public('branch_attempt', **intent)
            events, raw, audit, run_failed = [], None, None, False

            def branch_emit(event, _index=index, _events=events, **fields):
                row = {'event': event, **copy.deepcopy(fields)}
                public('branch_event', global_index=_index, branch_event=row)
                _events.append(row)

            experience = None if arm == utility.ARMS[0] else candidate if arm == utility.ARMS[1] else placebo
            attempted += 1
            try:
                raw = run_one(copy.deepcopy(by_id[name]), copy.deepcopy(experience), arm=arm,
                    protocol=copy.deepcopy(protocol), output=directory, emit=branch_emit)
            except BaseException as exc:
                run_failed = True
                remember(exc)
            row = unknown(index)
            try:
                audit = audit_one(directory, copy.deepcopy(by_id[name]), copy.deepcopy(experience),
                    copy.deepcopy(protocol), arm=arm, expected_public_events=copy.deepcopy(events))
                _require(type(audit) is dict and audit.get('audit') == 'passed'
                         and audit.get('status') in {'complete', 'unknown'})
                row = copy.deepcopy(audit['record'])
                utility.phase_decision(partition, tasks, protocol, records + [row])
                audited += 1
            except BaseException as exc:
                remember(exc)
                audit, row = None, unknown(index)
            observed = audit.get('cost') if audit is not None else raw.get('cost') if type(raw) is dict else None
            observed_cost, incomplete = _cost(observed)
            if audit is None:
                observed_cost['unknown_additional_cost'] = True
            known = (audit is not None and audit['status'] == 'complete' and not run_failed
                and not journal_failed and not incomplete and not observed_cost['unknown_additional_cost']
                and type(row['outcome']) is bool and row['valid_for_continuation'] and row['released'])
            if not known:
                row = unknown(index, row['released'])
            costs.append(observed_cost)
            entry = {'global_index': index, 'record': row, 'cost': observed_cost,
                'audit_passed': audit is not None, 'private_files': _files(audit.get('private_files'))
                    if audit is not None else {}, 'public_events_sha256': fingerprint(events),
                'public_event_count': len(events), 'run_callback_returned': not run_failed}
            try:
                private_files[f'result-{index:03d}.json'] = _write_private(
                    output / f'result-{index:03d}.json', entry)
                public('branch_audited', global_index=index, record=row, cost=observed_cost,
                       result_file=private_files[f'result-{index:03d}.json'])
            except BaseException as exc:
                remember(exc)
                journal_failed = True
                row = unknown(index, row['released'])
                known = False
            records.append(row)
            decision = utility.phase_decision(partition, tasks, protocol, records)
            if not known:
                break
    except BaseException as exc:
        remember(exc)

    def summary():
        current = utility.phase_decision(partition, tasks, protocol, records)
        new = copy.deepcopy(records[start:])
        known = sum(type(row['outcome']) is bool and row['valid_for_continuation'] and row['released'] for row in new)
        return {'status': 'complete' if known == len(schedule) and not journal_failed and error_type is None
                    else 'stopped_unknown', 'phase': phase, 'batch_index': batch_index,
            'start_global_index': start, 'protocol_sha256': protocol_sha,
            'prior_records_sha256': fingerprint(prior_records), 'scheduled_branches': len(schedule),
            'attempt_intents': intents, 'attempted_branches': attempted, 'audited_branches': audited,
            'known_branches': known, 'unknown_branches': attempted - known,
            'unattempted_branches': len(schedule) - attempted, 'new_records': new,
            'phase_decision': current, 'cost': _total(costs), 'journal_failed': journal_failed,
            'error_type': error_type, 'private_files': copy.deepcopy(private_files),
            'maximum_attempts_per_branch': 1, 'automatic_retry': False,
            'accepted_experiences': 0, 'training_authorized': False,
            'scope': 'One fixed batch only. Costs are observed lower bounds, never sums of overlapping '
                'trace/step meters. Prior records require outer authentication; physical release and '
                'service cost reconciliation remain with the outer controller.'}

    result = summary()
    try:
        _write_private(output / 'report.json', result)
    except BaseException as exc:
        remember(exc)
        journal_failed = True
        if len(records) > start:
            records[-1] = unknown(len(records) - 1, records[-1]['released'])
        result = summary()
    if interrupted is not None:
        if isinstance(interrupted, SystemExit):
            raise SystemExit(interrupted.code if type(interrupted.code) is int else 1) from None
        raise type(interrupted)() from None
    return result
