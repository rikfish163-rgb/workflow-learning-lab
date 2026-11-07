"""Extract one coding-assistant-authored contract from fixed visible frames.

The caller must authenticate the messages and trace hashes separately. This
module parses actor actions as syntax only; it never executes them or reads files.
The fixed grammar was authored outside the experiment, not by its B3 writer,
and its construction is not a claim of model learning.
"""

import ast
import json

from scripts import recovery_evidence_lesson as lesson
from scripts.completion_lesson import _json
from scripts.write_partial_lesson import _check_api_uses, _check_names
from wll.protocol import fingerprint

CREATOR = 'assistant_authored_structural_extractor'
CONTRACT_VERSION = 'selected_batch_coverage_v1'
_SOURCE_COVERAGE = 'fixed_partial_actor_visible_frames'
_HASH_SEMANTICS = 'private_trace_sha256_not_episode_objects'
_EVIDENCE_LIMITS = (
    'Fixed partial source coverage does not prove global intended target coverage, '
    'causal benefit, unchanged hidden state or safe repetition. Unknown outcomes are not known missing calls.'
)
_EXPERIENCE_FIELDS = (
    ('failure_mode', 'A loop can omit a selected target from a batch operation while returning results for others.'),
    ('applies_when', 'The task requests the current batch operation for selected targets and their relevant '
     'dispatch history can be checked.'),
    ('avoid_when', 'Avoid issuing the current requested batch operation if its complete relevant dispatch '
     'history is missing, a relevant previously attempted operation has an unknown outcome, or task '
     'scope, authorization or action safety is unknown.'),
    ('rule', 'For the current requested batch operation, reconcile selected targets with task requirements '
     'and examine the complete relevant per-target dispatch history. Separately track acknowledged, '
     'demonstrably unattempted and unknown-outcome operations. Issue this operation only for a target '
     'known not to have been attempted for that operation from the complete relevant history, and only '
     'when authorized and safe. Never repeat its acknowledged or uncertain writes. If scope, relevant '
     'history, a previously attempted operation outcome, authorization, or safety remains unknown, '
     'seek read-only evidence or stop. '
     'This restriction does not prohibit a separate authorized operation. A selected list alone cannot '
     'establish task completion.'),
)
CONSTRUCTION_SHA256 = fingerprint({'creator': CREATOR, 'contract_version': CONTRACT_VERSION,
                                   'experience_fields': dict(_EXPERIENCE_FIELDS)})
_SOURCE_KEYS = frozenset({
    'task_objective', 'visible_steps', 'public_api', 'source_coverage',
    'authentication_binding_omitted', 'same_authentication_argument_ast',
    'observed_actor_selected_items', 'observed_prefix_write_results', 'observed_followup_write_results',
})


def _require(condition):
    if not condition:
        raise ValueError('coverage_contract_ineligible')


def _provenance(source_scenarios, source_trace_sha256):
    if (type(source_scenarios) is not list or len(source_scenarios) != 2
            or any(type(value) is not str for value in source_scenarios)
            or source_scenarios != list(lesson.SOURCE_SCENARIOS)):
        raise ValueError('coverage_contract_invalid_provenance') from None
    try:
        lesson._hashes(source_trace_sha256, 4)
    except (ValueError, TypeError):
        raise ValueError('coverage_contract_invalid_provenance') from None
    return list(source_scenarios), list(source_trace_sha256)


def _metadata(scenarios, hashes):
    return {
        'creator': CREATOR,
        'contract_version': CONTRACT_VERSION,
        'construction_sha256': CONSTRUCTION_SHA256,
        'source_scenarios': list(scenarios),
        'source_trace_sha256': list(hashes),
        'source_kind': lesson.SOURCE_KIND,
        'evidence_kind': lesson.EVIDENCE_KIND,
        'source_hash_kind': lesson.SOURCE_HASH_KIND,
        'source_episode_hashes_semantics': _HASH_SEMANTICS,
        'source_coverage': _SOURCE_COVERAGE,
        'source_provenance_authenticated': False,
        'content_review_required': True,
        'accepted': False,
        'content_certified': False,
        'utility_verified': False,
        'training_authorized': False,
    }


def _candidate(scenarios, hashes):
    experience = {
        'experience_id': fingerprint([CONSTRUCTION_SHA256, scenarios, hashes])[:16],
        **dict(_EXPERIENCE_FIELDS),
        'source_scenarios': list(scenarios),
        'source_episode_hashes': list(hashes),
    }
    evidence = {
        'cases': [
            {'source_index': 0, 'selected_items': 5, 'loop_result_mappings': 4,
             'followup_result_mappings': 1},
            {'source_index': 1, 'selected_items': 4, 'loop_result_mappings': 3,
             'followup_result_mappings': 1},
        ],
        'scope': 'actor_selected_items_only',
        'omission': 'within_observed_loop_only',
        'prior_dispatch_history': 'unknown',
        'task_target_scope': 'unknown',
        'authentication_identity': 'not_proven',
        'write_repeat_safety': 'unknown',
        'terminal_causality': 'unknown',
        'mapping_confirms_write_success': False,
    }
    return {**_metadata(scenarios, hashes), 'status': 'candidate_unvalidated',
            'evidence': evidence, 'experience': experience}


def _same_plain(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (all(type(key) is str for key in actual) and actual.keys() == expected.keys()
                and all(_same_plain(actual[key], item) for key, item in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_plain(first, second) for first, second in zip(actual, expected, strict=True))
    return actual == expected


def validate_contract(value):
    """Validate fixed candidate fields, not the authenticity of its source bytes."""
    try:
        _require(type(value) is dict and all(type(key) is str for key in value))
        scenarios, hashes = _provenance(value.get('source_scenarios'), value.get('source_trace_sha256'))
        expected = _candidate(scenarios, hashes)
        _require(_same_plain(value, expected))
        return value
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError, MemoryError, OverflowError):
        raise ValueError('coverage_contract_invalid_candidate') from None


def _print_call(node, n):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'print' and len(node.args) == n and not node.keywords)


def _same(first, second):
    return ast.dump(first, include_attributes=False) == ast.dump(second, include_attributes=False)


def _direct_write(node, endpoint):
    return isinstance(node, ast.Call) and lesson._api_call(node) == endpoint


def _capture_and_print(body, target, endpoint, forbidden):
    _require(len(body) == 2 and isinstance(body[0], ast.Assign)
             and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name)
             and _direct_write(body[0].value, endpoint)
             and isinstance(body[1], ast.Expr) and _print_call(body[1].value, 2))
    printed = body[1].value.args
    _require(body[0].targets[0].id not in forbidden
             and _same(printed[0], target) and isinstance(printed[1], ast.Name)
             and printed[1].id == body[0].targets[0].id)


def _strict_write_shapes(scenario, trees, doc):
    _, _, pivot_index, followup_index, _ = lesson._SPECS[scenario]
    pivot, followup = trees[pivot_index], trees[followup_index]
    endpoint, target_key, auth_key = lesson._public_api(doc)
    target, auth = lesson._call(pivot, endpoint, target_key, auth_key)
    later_target, _ = lesson._call(followup, endpoint, target_key, auth_key)
    loop = pivot.body[0]
    if scenario == lesson.SOURCE_SCENARIOS[0]:
        _require(len(loop.body) == 1 and isinstance(loop.body[0], ast.Expr)
                 and _print_call(loop.body[0].value, 2))
        args = loop.body[0].value.args
        _require(_same(args[0], target) and _direct_write(args[1], endpoint))
        _require(len(followup.body) == 1 and isinstance(followup.body[0], ast.Expr)
                 and _print_call(followup.body[0].value, 1)
                 and _direct_write(followup.body[0].value.args[0], endpoint))
    else:
        collection = loop.iter.value.args[0]
        _require(isinstance(collection, ast.Name))
        forbidden = {loop.target.id, collection.id, auth.id, 'apis', 'print', 'list', 'sorted'}
        _capture_and_print(loop.body, target, endpoint, forbidden)
        _capture_and_print(followup.body, later_target, endpoint, forbidden)


def _inspect_source(source, scenario):
    _require(type(source) is dict and set(source) == _SOURCE_KEYS)
    lesson._text(source['task_objective'])
    _require(source['source_coverage'] == _SOURCE_COVERAGE
             and source['authentication_binding_omitted'] is True
             and source['same_authentication_argument_ast'] is True)
    visible = source['visible_steps']
    _require(type(visible) is list and len(visible) == len(lesson.FRAME_INDICES[scenario]))
    rows, trees = {}, {}
    for row, expected_index in zip(visible, lesson.FRAME_INDICES[scenario], strict=True):
        _require(type(row) is dict and set(row) == {'step_index', 'action', 'observation'}
                 and type(row['step_index']) is int and row['step_index'] == expected_index)
        action = lesson._text(row['action'])
        observation = lesson._text(row['observation'], empty=True)
        tree, classification = lesson._parse(action)
        _require(classification['status'] == 'no_match')
        lesson._check_visible_bindings(tree)
        lesson._check_sorted_bindings(tree)
        _check_names(tree, {'apis', 'print'})
        _check_api_uses(tree)
        rows[expected_index], trees[expected_index] = {'action': action, 'observation': observation}, tree
    _, snapshot_index, _, _, expected_count = lesson._SPECS[scenario]
    # The original shape helper parses this JSON; precheck duplicate and nonfinite keys.
    selected = _json(rows[snapshot_index]['observation'])
    if scenario == lesson.SOURCE_SCENARIOS[0]:
        _require(type(selected) is list and all(type(value) is int for value in selected)
                 and selected == sorted(selected))
    count = lesson._visible_shape(scenario, rows, trees, source['public_api'])
    _strict_write_shapes(scenario, trees, source['public_api'])
    _require(count == expected_count)
    for name, expected in (('observed_actor_selected_items', count),
                           ('observed_prefix_write_results', count - 1),
                           ('observed_followup_write_results', 1)):
        _require(type(source[name]) is int and source[name] == expected)


def extract_contract(messages, *, source_scenarios, source_trace_sha256):
    """Recheck two visible projections; return a candidate or typed abstention.

    Hashes and messages are caller-provided metadata, not authenticated here.
    The candidate does not assert task completion, prior dispatch, write safety,
    identity equivalence, or causal benefit.
    """
    scenarios, hashes = _provenance(source_scenarios, source_trace_sha256)
    try:
        _require(type(messages) is list and len(messages) == 2
                 and all(type(message) is dict and all(type(key) is str for key in message)
                         and set(message) == {'role', 'content'}
                         and type(message['role']) is str and type(message['content']) is str
                         for message in messages)
                 and messages[0] == {'role': 'system', 'content': lesson.SYSTEM}
                 and messages[1]['role'] == 'user')
        _require(len(messages[1]['content']) <= lesson.MAX_MESSAGE_BYTES)
        encoded = json.dumps(messages, ensure_ascii=False, allow_nan=False)
        _require(len(encoded.encode('utf-8')) <= lesson.MAX_MESSAGE_BYTES)
        payload = _json(messages[1]['content'])
        _require(type(payload) is dict and set(payload) == {'evidence_kind', 'sources', 'evidence_limits'}
                 and payload['evidence_kind'] == lesson.EVIDENCE_KIND
                 and payload['evidence_limits'] == _EVIDENCE_LIMITS
                 and type(payload['sources']) is list and len(payload['sources']) == 2)
        for source, scenario in zip(payload['sources'], scenarios, strict=True):
            _inspect_source(source, scenario)
        return _candidate(scenarios, hashes)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, UnicodeError,
            SyntaxError, RecursionError, MemoryError, OverflowError):
        return {**_metadata(scenarios, hashes), 'status': 'abstained',
                'reason': 'unsupported_or_ambiguous_evidence'}
