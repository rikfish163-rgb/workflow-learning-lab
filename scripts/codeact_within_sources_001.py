"""Pure qualification over externally authenticated CodeAct owner envelopes.

No receipt, lifecycle or transport authentication happens here. The caller must
authenticate complete runner/client/owner/model-lease evidence before supplying
the four verified flags. Writer inputs are private; only ``public`` is exportable.
AST equivalence is syntactic deduplication, never semantic distinctness or repair.
"""
import ast
import copy
import hashlib
import json
import math

from scripts.codeact_within_episode_001 import _response_valid, _worker_valid
from wll.runner import extract_code

TASK_IDS = ('22cc237_3', '27e1026_3', '287e338_3', '2a163ab_3', '60d0b5b_3', '692c77d_3',
            '82e2fac_3', 'b0a8eae_3', 'b7a9ee9_3', 'c901732_3', 'cf6abd2_3', 'd0b1f43_3')
ERROR_TYPES = frozenset({'NameError', 'TypeError', 'ValueError', 'KeyError', 'IndexError',
                         'AttributeError', 'ZeroDivisionError', 'AssertionError', 'Exception'})
_VERIFIED = {'owner_lifecycle', 'container_release', 'model_lease_release', 'all_costs'}
_COUNTS = {'generation_attempts', 'generation_returns', 'prompt_tokens', 'completion_tokens',
           'actions', 'api_request_dispatches', 'budget_rejections'}
_KNOWN = {'prompt_tokens', 'completion_tokens', 'api_request_dispatches', 'budget_rejections'}
_FATAL = {'timeout', 'unknown', 'api_budget_rejected'}
_SCORED_STOPS = {'completed', 'initial_completion', 'generation_budget', 'action_budget',
                 'actor_wall', 'generation_length', 'generation_unknown_finish',
                 'generation_format_error', 'context_budget'}


def _require(value):
    if not value:
        raise ValueError('codeact_within_sources_invalid')


def _integer(value):
    return type(value) is int and value >= 0


def _hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _messages(value):
    _require(type(value) is list and 1 <= len(value) <= 256)
    for message in value:
        _require(type(message) is dict and set(message) == {'role', 'content'}
                 and type(message['role']) is str and message['role'] in {'system', 'user', 'assistant'}
                 and type(message['content']) is str)
        message['content'].encode('utf-8')


def _outcome(value):
    _require(type(value) is dict and set(value) == {'counts', 'known_costs', 'termination',
        'model_reported_complete', 'official_success', 'costs_known', 'actor_wall_seconds'})
    _require(type(value['costs_known']) is bool and type(value['termination']) is str)
    for field in ('model_reported_complete', 'official_success'):
        _require(value[field] is None or type(value[field]) is bool)
    if value['official_success'] is not None:
        _require(value['termination'] in _SCORED_STOPS)
    counts, known = value['counts'], value['known_costs']
    _require(type(counts) is dict and set(counts) == _COUNTS
             and type(known) is dict and set(known) == _KNOWN
             and all(_integer(v) for v in known.values()))
    for key, number in counts.items():
        _require(_integer(number) or (key in _KNOWN and number is None and not value['costs_known']))
    _require(counts['actions'] <= counts['generation_returns'] <= counts['generation_attempts'] <= 40)
    if value['costs_known']:
        wall = value['actor_wall_seconds']
        _require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= 1500
                 and all(counts[key] == known[key] for key in _KNOWN)
                 and counts['api_request_dispatches'] <= 500)
    return counts


def _trace(private, outcome):
    """Check preserved history/counters, returning parent-only local candidates."""
    _require(type(private) is dict and set(private) == {'messages', 'steps'})
    steps = private['steps']
    _require(type(steps) is list and len(steps) <= 40)
    history = copy.deepcopy(steps[0]['messages_before'] if steps else private['messages'])
    _messages(history)
    measured = dict.fromkeys(_KNOWN, 0)
    returns = actions = 0
    errors, candidates = [], []
    pending = False
    for index, step in enumerate(steps):
        _require(type(step) is dict and set(step) == {'index', 'seed', 'messages_before', 'response', 'action', 'execution'}
                 and type(step['index']) is int and step['index'] == index
                 and type(step['seed']) is int and step['seed'] == 17 + index)
        _messages(step['messages_before'])
        _require(not pending and step['messages_before'] == history)
        response, action, executed = step['response'], step['action'], step['execution']
        if response is None:
            _require(action is None and executed is None and index == len(steps) - 1)
            pending = True
            continue
        _require(_response_valid(response))
        returns += 1
        for key in ('prompt_tokens', 'completion_tokens'):
            measured[key] += response[key]
        if action is None:
            _require(executed is None and index == len(steps) - 1
                     and (response['finish_reason'] != 'stop' or response['format_error'] is not None))
            continue
        _require(type(action) is str and response['finish_reason'] == 'stop'
                 and response['format_error'] is None and extract_code(response['text']) == action
                 and _worker_valid(executed, measured))
        actions += 1
        requests_before = measured['api_request_dispatches']
        measured['api_request_dispatches'] = executed['api']['request_dispatches']
        measured['budget_rejections'] = executed['api']['budget_rejections']
        metadata = executed['execution']
        if metadata['status'] in _FATAL:
            _require(index == len(steps) - 1 and outcome['official_success'] is None)
        if (metadata['status'] in {'error_before_exec', 'error_in_exec'}
                and metadata['error_type'] in ERROR_TYPES and metadata['hook_calls'] == 1
                and (metadata['error_type'] != 'Exception'
                     or measured['api_request_dispatches'] > requests_before)):
            errors.append(index)
        elif metadata['status'] == 'success' and metadata['hook_calls'] == 1 and errors:
            canonical = ast.dump(ast.parse(action), annotate_fields=True, include_attributes=False)
            candidates.append((canonical, {'messages_before': copy.deepcopy(history),
                'raw_response': response['text'], 'action': action, 'observation': executed['observation'],
                'error_index': errors[0], 'target_index': index}))
        history.extend([{'role': 'assistant', 'content': response['text']},
                        {'role': 'user', 'content': executed['observation']}])
    _messages(private['messages'])
    _require(history == private['messages'])
    counts = outcome['counts']
    _require(counts['generation_returns'] == returns and counts['actions'] == actions
             and all(counts[key] == measured[key] for key in _KNOWN))
    attempts = counts['generation_attempts']
    _require(attempts == len(steps) or (pending and attempts == len(steps) - 1
                                      and outcome['termination'] in {'actor_wall', 'scoring_failure'}))
    if outcome['official_success'] is not None and outcome['model_reported_complete'] is True:
        _require(outcome['termination'] == 'completed' and not pending and actions == len(steps))
    return errors, candidates


def qualify_sources(episodes):
    """Return public hashes/counts/gate and, only on passage, private writer rows."""
    try:
        return _qualify(episodes)
    except Exception:
        raise ValueError('codeact_within_sources_invalid') from None


def _qualify(episodes):
    _require(type(episodes) is list and len(episodes) == 12)
    sources, targets, writer, seen = [], [], [], set()
    eligible_scenes = known_episodes = duplicates = 0
    for index, envelope in enumerate(episodes):
        _require(type(envelope) is dict and set(envelope) == {'source_index', 'task_id', 'outcome', 'private', 'verified'}
                 and type(envelope['source_index']) is int and envelope['source_index'] == index
                 and envelope['task_id'] == TASK_IDS[index])
        verified, outcome = envelope['verified'], envelope['outcome']
        _require(type(verified) is dict and set(verified) == _VERIFIED
                 and all(type(v) is bool for v in verified.values()))
        _outcome(outcome)
        errors, candidates = _trace(envelope['private'], outcome) if outcome['costs_known'] else ([], [])
        known = (all(verified.values()) and outcome['costs_known']
                 and type(outcome['official_success']) is bool
                 and type(outcome['model_reported_complete']) is bool)
        eligible, reason = None, 'unverified_or_unknown'
        if known:
            known_episodes += 1
            eligible = (outcome['model_reported_complete'] and outcome['official_success']
                        and bool(errors) and bool(candidates))
            reason = 'eligible' if eligible else 'known_ineligible'
        unique = 0
        if eligible:
            eligible_scenes += 1  # The fixed cohort contains one task per distinct scene.
            for canonical, row in candidates:
                if canonical in seen:
                    duplicates += 1
                    continue
                seen.add(canonical)
                unique += 1
                writer.append(row)
                targets.append({'source_index': index, 'target_index': row['target_index'],
                    'error_index': row['error_index'], 'input_sha256': _hash(row),
                    'messages_sha256': _hash(row['messages_before']),
                    'raw_response_sha256': _hash(row['raw_response']),
                    'canonical_sha256': _hash(canonical)})
        sources.append({'source_index': index, 'task_id_sha256': _hash(envelope['task_id']),
            'episode_sha256': _hash(envelope), 'eligible': eligible, 'reason': reason,
            'eligible_error_count': len(errors), 'target_candidate_count': len(candidates),
            'unique_target_count': unique})
    qualified = known_episodes == 12 and eligible_scenes >= 4 and len(writer) >= 20
    public = {'gate_kind': 'codeact_within_workflow_gate', 'qualified': qualified,
        'episode_count': 12, 'known_episode_count': known_episodes,
        'eligible_scene_count': eligible_scenes, 'unique_target_count': len(writer),
        'duplicate_target_count': duplicates, 'sources': sources, 'targets': targets,
        'canonicalization': 'syntactic_ast_not_semantic_distinctness',
        'authentication': 'external_precondition_not_performed_here'}
    return {'public': public, 'writer_inputs': writer if qualified else []}
