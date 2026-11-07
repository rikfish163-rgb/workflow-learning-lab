"""Fixed writer/review cases over externally authenticated CodeAct inputs.

No source authentication, registration, resource admission or file writing occurs
here. The outer coordinator binds inputs, NativeModel004, HTTP timeout120 and
the service lease. All record callbacks and returned private values are private.
Only the recomputed public projection may be exposed; no result authorizes
freezing, acceptance or training.
"""
import copy
import json
import math
import time

from scripts import codeact_within_experience_001 as experience
from scripts import native_experience_content_001 as content
from scripts.codeact_within_sources_001 import TASK_IDS, _hash
from scripts.verification_token_counter import _message_payload

ID = 'codeact-within-writer-cases-001'
STAGE_SECONDS = 240
HTTP_TIMEOUT_SECONDS = 120
_ERROR = 'codeact_writer_cases_invalid'
_COUNT = ('health_checks', 'CPU_count_attempts', 'CPU_counts_returned', 'generation_attempts',
          'generation_results', 'returned_prompt_tokens_lower_bound', 'returned_completion_tokens_lower_bound')
_REASONS = {'not_attempted', 'parsed_candidate', 'context_overflow', 'incomplete_or_malformed_reply',
            'invalid_candidate_schema', 'model_abstained', 'journal_failure', 'health_failure',
            'count_failure', 'generation_failure', 'response_failure', 'invalid_review_schema',
            'plausible_pending_utility', 'reject', 'insufficient_evidence', 'model_factory_failure',
            'invalid_stage_result', 'stage_deadline', 'clock_failure', 'screen_failure',
            'deterministic_screen_rejected', 'outer_deadline', 'case_failure'}
_REASONS.add('interrupted')


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _inputs(prepared, registered):
    _require(type(prepared) is dict and set(prepared) == {'contexts', 'preparation_manifest'}
             and type(registered) is dict and set(registered) == {
                 'preparation_manifest_sha256', 'origin_ids', 'selection'})
    manifest, contexts = prepared['preparation_manifest'], prepared['contexts']
    origins = [task[:-2] for task in TASK_IDS]
    _require(type(manifest) is dict and set(manifest) == {'origin_ids', 'source_authentication', 'selection'}
             and _same(manifest['origin_ids'], origins) and _same(registered['origin_ids'], origins)
             and registered['preparation_manifest_sha256'] == _hash(manifest)
             and _same(registered['selection'], manifest['selection'])
             and type(contexts) is list and 1 <= len(contexts) <= 12
             and type(manifest['selection']['selected_source_count']) is int
             and manifest['selection']['selected_source_count'] == len(contexts)
             and len(manifest['selection']['selections']) == len(contexts))
    indices = []
    for ctx, selection in zip(contexts, manifest['selection']['selections'], strict=True):
        _, lineage = experience._context(ctx)
        index = lineage['source_index']
        _require(_same(lineage, dict(**selection, source_scenario=origins[index])))
        indices.append(index)
    _require(indices == sorted(set(indices)))
    return copy.deepcopy(prepared), copy.deepcopy(registered)


def _checked_inputs(prepared, registered):
    try:
        return _inputs(prepared, registered)
    except Exception:
        raise ValueError(_ERROR) from None


def validate_preparation(source_private, registered_inputs):
    """Check frozen values and context structure, without authenticating origins."""
    prepared, registered = _checked_inputs(source_private, registered_inputs)
    return dict(preparation_manifest_sha256=registered['preparation_manifest_sha256'],
                selected_source_indices=[ctx['lineage']['source_index'] for ctx in prepared['contexts']],
                origin_ids_sha256=_hash(registered['origin_ids']), source_authenticated=None)


class _RecordFailure(Exception):
    pass


class _TrackedModel:
    def __init__(self, model, stage, now):
        self.model, self.stage, self.now = model, stage, now
        self.max_tokens = stage['max_tokens']
        self.digest = _message_payload(stage['messages'])[1]
        _require(type(model.max_tokens) is int and model.max_tokens == self.max_tokens)

    def _call(self, operation, messages=None, seed=None):
        _require(type(self.model.max_tokens) is int and self.model.max_tokens == self.max_tokens
                 and self.now() < self.stage['deadline'])
        snapshot = None if messages is None else copy.deepcopy(messages)
        if messages is not None:
            _require(_message_payload(messages)[1] == self.digest)
        if operation == 'generate':
            _require(type(seed) is int and seed == self.stage['seed'])
        entry = dict(operation=operation, returned=False, result=None, error=False)
        self.stage['operations'].append(entry)
        try:
            if operation == 'health':
                value = self.model.check_health()
            elif operation == 'count':
                value = self.model.count_input_tokens(snapshot)
            else:
                value = self.model.generate(snapshot, seed=seed)
            entry.update(returned=True, result=copy.deepcopy(value))
            _require((snapshot is None or _message_payload(snapshot)[1] == self.digest)
                     and self.now() <= self.stage['deadline'])
            return value
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            entry['error'] = True
            for key in ('count_audit', 'generation_audit'):
                if type(getattr(error, key, None)) is dict:
                    entry[key] = copy.deepcopy(getattr(error, key))
            raise

    def check_health(self):
        return self._call('health')

    def count_input_tokens(self, messages):
        return self._call('count', messages)

    def generate(self, messages, seed):
        return self._call('generate', messages, seed)


def _valid_result(result, stage):
    field = 'candidate' if stage == 'writer' else 'verdict'
    _require(type(result) is dict and set(result) == {'status', 'reason', field, 'accepted',
             'content_certified', 'training_authorized', 'cost'}
             and result['accepted'] is result['content_certified'] is result['training_authorized'] is False
             and result['status'] in ({'candidate_unvalidated', 'abstained', 'unknown'} if stage == 'writer'
                                     else {'plausible_pending_utility', 'rejected', 'abstained', 'unknown'})
             and result['reason'] in _REASONS and type(result['cost']) is dict
             and set(result['cost']) == {*_COUNT, 'unknown_cost'}
             and type(result['cost']['unknown_cost']) is bool
             and all(type(result['cost'][key]) is int and result['cost'][key] >= 0 for key in _COUNT))


def _cost(stage):
    result = stage['result']
    raw = result.get('cost') if type(result) is dict else None
    valid = (type(raw) is dict and set(raw) == {*_COUNT, 'unknown_cost'}
             and type(raw['unknown_cost']) is bool
             and all(type(raw[key]) is int and raw[key] >= 0 for key in _COUNT))
    cost = {key: raw[key] if valid else 0 for key in _COUNT}
    ops = stage['operations']
    for key, operation, returned in [('health_checks', 'health', False), ('CPU_count_attempts', 'count', False),
            ('CPU_counts_returned', 'count', True), ('generation_attempts', 'generate', False),
            ('generation_results', 'generate', True)]:
        cost[key] = max(cost[key], sum(op['operation'] == operation and (not returned or op['returned']) for op in ops))
    for key in ('prompt_tokens', 'completion_tokens'):
        values = [op['result'].get(key) for op in ops if op['operation'] == 'generate'
                  and op['returned'] and type(op['result']) is dict]
        measured = sum(value for value in values if type(value) is int and value > 0)
        name = 'returned_' + key + '_lower_bound'
        cost[name] = max(cost[name], measured)
    known = (valid and raw['unknown_cost'] is False and stage['error'] is None
             and stage['status'] != 'unknown' and _finite(stage['wall_seconds'])
             and cost['health_checks'] == cost['CPU_count_attempts'] == cost['CPU_counts_returned'] == 1
             and cost['generation_attempts'] == cost['generation_results'] <= 1
             and all(op['returned'] and not op['error'] for op in ops))
    for key in ('prompt_tokens', 'completion_tokens'):
        lower = cost['returned_' + key + '_lower_bound']
        cost[key] = lower if known else None
        cost['returned_' + key + '_lower_bound'] = lower if lower > 0 or known else None
    return dict(cost, known=known, wall_seconds=stage['wall_seconds'])


def _stage(ctx, name, factory, emit, now, deadline, candidate=None):
    index = ctx['lineage']['source_index']
    messages = (experience.build_writer_messages(ctx) if name == 'writer'
                else experience.build_review_messages(ctx, candidate))
    started = now()
    receipt = dict(stage=name, max_tokens=128 if name == 'writer' else 256,
                   seed=(17 if name == 'writer' else 117) + index,
                   deadline=min(deadline, started + STAGE_SECONDS), messages=messages,
                   result=None, operations=[], events=[], status='unknown', reason='model_factory_failure',
                   error=None, wall_seconds=None)
    def record(event):
        receipt['events'].append(copy.deepcopy(event))
        emit(dict(event='stage_event', source_index=index, stage=name, payload=event))
    try:
        emit(dict(event='stage_start', source_index=index, stage=name, seed=receipt['seed'],
                  max_tokens=receipt['max_tokens'], deadline=receipt['deadline'], messages=messages))
        model = factory(max_tokens=receipt['max_tokens'], source_index=index, stage=name, deadline=receipt['deadline'])
        tracked = _TrackedModel(model, receipt, now)
        receipt['reason'] = 'invalid_stage_result'
        result = (experience.write_once(ctx, tracked, seed=receipt['seed'], record=record) if name == 'writer'
                  else experience.review_once(ctx, candidate, tracked, seed=receipt['seed'], record=record))
        receipt['result'] = result
        _valid_result(result, name)
        receipt.update(status=result['status'], reason=result['reason'])
    except _RecordFailure:
        receipt.update(error='journal_failure', reason='journal_failure')
    except (KeyboardInterrupt, SystemExit):
        receipt.update(status='unknown', error='interrupted', reason='interrupted')
    except Exception:
        receipt['error'] = receipt['reason']
    finally:
        try:
            receipt['wall_seconds'] = now() - started
            if now() > receipt['deadline']:
                receipt.update(status='unknown', reason='stage_deadline', error='stage_deadline')
        except Exception:
            receipt.update(status='unknown', reason='clock_failure', error='clock_failure')
    return receipt


def _pending(receipts):
    rows = []
    for item in receipts:
        if item['status'] == 'plausible_pending_utility':
            candidate = item['writer']['result']['candidate']
            rows.append(dict(source_index=item['source_index'], source_scenario=item['context']['lineage']['source_scenario'],
                candidate=copy.deepcopy(candidate), insertion=content.insertion(candidate),
                candidate_sha256=item['screen']['candidate_sha256'], insertion_sha256=item['screen']['insertion_sha256'],
                review_result_sha256=_hash(item['review']['result']), status='plausible_pending_utility',
                accepted=False, content_certified=False, training_authorized=False))
    return rows


def _projection(private):
    slots, all_stages = [], []
    contexts = private['source_private']['contexts']
    _require(len(private['receipts']) == len(contexts)
             and type(private['completed']) is type(private['journal_failed']) is bool)
    for item, ctx in zip(private['receipts'], contexts, strict=True):
        _require(type(item['source_index']) is int and item['source_index'] == ctx['lineage']['source_index']
                 and _same(item['context'], ctx))
        _require(item['status'] in {'unattempted', 'unknown', 'abstained', 'screen_rejected',
                                    'plausible_pending_utility', 'rejected'} and item['reason'] in _REASONS)
        slot = dict(source_index=item['source_index'], status=item['status'], reason=item['reason'],
                    writer=None, review=None, screen=None, receipt_sha256=_hash(item))
        for name in ('writer', 'review'):
            stage = item[name]
            if stage is not None:
                _require(stage['reason'] in _REASONS and stage['stage'] == name
                         and stage['status'] in {'candidate_unvalidated', 'abstained', 'unknown',
                                                 'plausible_pending_utility', 'rejected'}
                         and type(stage['max_tokens']) is int and stage['max_tokens'] == (128 if name == 'writer' else 256)
                         and (stage['wall_seconds'] is None or _finite(stage['wall_seconds']))
                         and type(stage['seed']) is int and stage['seed'] == (17 if name == 'writer' else 117) + item['source_index'])
                if stage['result'] is not None and stage['error'] is None:
                    _valid_result(stage['result'], name)
                cost = _cost(stage)
                all_stages.append(cost)
                slot[name] = dict(status=stage['status'], reason=stage['reason'], cost=cost,
                                  messages_sha256=_message_payload(stage['messages'])[1],
                                  result_sha256=None if stage['result'] is None else _hash(stage['result']))
        if item['screen'] is not None:
            screen = item['screen']
            _require(screen['status'] in {'passed', 'rejected'} and type(screen['reason_codes']) is list
                     and all(code in content.SCREEN_REASON_CODES for code in screen['reason_codes'])
                     and all(experience._sha(screen[key]) for key in ('candidate_sha256', 'insertion_sha256', 'history_sha256'))
                     and type(screen['inventory_values']) is int and screen['inventory_values'] >= 0)
            slot['screen'] = {key: screen[key] for key in ('status', 'reason_codes', 'candidate_sha256',
                                                          'insertion_sha256', 'history_sha256', 'inventory_values')}
        slots.append(slot)
    whole = (private['completed'] and not private['journal_failed'] and bool(all_stages)
             and all(row['status'] not in {'unknown', 'unattempted'} for row in slots)
             and all(cost['known'] for cost in all_stages))
    totals = {key: sum(cost[key] for cost in all_stages) if all_stages else None for key in _COUNT[:5]}
    for key in ('prompt_tokens', 'completion_tokens'):
        values = [cost['returned_' + key + '_lower_bound'] for cost in all_stages
                  if cost['returned_' + key + '_lower_bound'] is not None]
        lower = sum(values) if values else None
        totals['returned_' + key + '_lower_bound'] = lower
        totals[key] = lower if whole else None
    totals['wall_seconds'] = private['wall_seconds'] if whole else None
    return dict(id=ID, status='complete' if whole else 'unknown', sources=slots, costs=totals,
                whole_cost_known=whole, pending_count=len(_pending(private['receipts'])),
                planned_source_count=len(slots), attempted_source_count=sum(row['status'] != 'unattempted' for row in slots),
                preparation_manifest_sha256=private['registered_inputs']['preparation_manifest_sha256'],
                origin_ids_sha256=_hash(private['registered_inputs']['origin_ids']),
                source_authenticated=None, service_lease_authenticated=None,
                cost_scope='case client-method operations and wall only; external source/service costs excluded',
                canfreeze=False, accepted=False, training_authorized=False)


def run_cases(source_private, registered_inputs, *, model_factory, record, deadline, _clock=time.monotonic):
    prepared, registered = _checked_inputs(source_private, registered_inputs)
    _require(callable(model_factory) and callable(record) and _finite(deadline) and deadline > 0)
    previous = None
    def now():
        nonlocal previous
        value = _clock()
        _require(_finite(value) and (previous is None or value >= previous))
        previous = float(value)
        return previous
    started = now()
    private = dict(source_private=prepared, registered_inputs=registered, events=[], journal_failed=False,
                   completed=False, wall_seconds=None, receipts=[dict(source_index=ctx['lineage']['source_index'],
                   context=ctx, status='unattempted', reason='not_attempted', writer=None, screen=None, review=None)
                   for ctx in prepared['contexts']])
    def emit(event):
        if private['journal_failed']:
            raise _RecordFailure
        entry = dict(payload=copy.deepcopy(event), recorded=False)
        private['events'].append(entry)
        try:
            record(copy.deepcopy(event))
            entry['recorded'] = True
        except (Exception, KeyboardInterrupt, SystemExit):
            private['journal_failed'] = True
            raise _RecordFailure from None
    try:
        emit(dict(event='cases_start', preparation_manifest_sha256=registered['preparation_manifest_sha256']))
        for item in private['receipts']:
            if now() >= deadline:
                break
            item.update(status='unknown', reason='case_failure')
            try:
                emit(dict(event='case_start', source_index=item['source_index'], context=item['context']))
                item['writer'] = _stage(item['context'], 'writer', model_factory, emit, now, deadline)
                writer = item['writer']
                if not _cost(writer)['known']:
                    item['reason'] = writer['reason']
                elif writer['status'] == 'abstained':
                    item.update(status='abstained', reason=writer['reason'])
                else:
                    item['reason'] = 'screen_failure'
                    candidate = writer['result']['candidate']
                    item['screen'] = experience.screen(item['context'], candidate)
                    if item['screen']['status'] == 'rejected':
                        item.update(status='screen_rejected', reason='deterministic_screen_rejected')
                    else:
                        item['review'] = _stage(item['context'], 'review', model_factory, emit, now, deadline, candidate)
                        review = item['review']
                        if _cost(review)['known']:
                            item.update(status=review['status'], reason=review['reason'])
                        else:
                            item['reason'] = review['reason']
                emit(dict(event='case_stop', source_index=item['source_index'], receipt=item))
            except _RecordFailure:
                item.update(status='unknown', reason='journal_failure')
            except Exception:
                item['status'] = 'unknown'
            if item['status'] == 'unknown':
                break
        else:
            private['completed'] = True
        emit(dict(event='cases_stop', completed=private['completed']))
    except (Exception, KeyboardInterrupt, SystemExit):
        private['completed'] = False
    try:
        private['wall_seconds'] = now() - started
    except Exception:
        private['completed'] = False
    private['pending_bundle_rows'] = _pending(private['receipts'])
    return dict(private=private, public=_projection(private))


def verify_public_projection(result, registered_inputs):
    try:
        _require(type(result) is dict and set(result) == {'public', 'private'})
        private = result['private']
        _inputs(private['source_private'], registered_inputs)
        _require(_same(private['registered_inputs'], registered_inputs)
                 and _same(private['pending_bundle_rows'], _pending(private['receipts']))
                 and _same(result['public'], _projection(private)))
        return True
    except Exception:
        raise ValueError(_ERROR) from None
