"""Pure preparation of CodeAct writer histories from externally authenticated rows.

The caller must authenticate the complete source gate, all twelve source episodes,
their initial histories and lineage before calling. This module performs no file
I/O, source qualification, model call, scoring or execution of actor code. The
history returned here is complete through the selected target, not the rest of
the task. Only ``context['history']`` is intended as model-visible material.
"""

import ast
import copy
import re

from scripts.codeact_within_sources_001 import _hash, _messages
from wll.prompts import SYSTEMS, TUTORIAL
from wll.runner import extract_code

_ERROR = 'codeact_within_writer_inputs_invalid'
_GATE = {'gate_kind', 'qualified', 'episode_count', 'known_episode_count', 'eligible_scene_count',
         'unique_target_count', 'duplicate_target_count', 'sources', 'targets', 'canonicalization', 'authentication'}
_SOURCE = {'source_index', 'task_id_sha256', 'episode_sha256', 'eligible', 'reason',
           'eligible_error_count', 'target_candidate_count', 'unique_target_count'}
_TARGET = {'source_index', 'target_index', 'error_index', 'input_sha256', 'messages_sha256',
           'raw_response_sha256', 'canonical_sha256'}
_ROW = {'messages_before', 'raw_response', 'action', 'observation', 'error_index', 'target_index'}


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _integer(value):
    return type(value) is int and value >= 0


def _sha(value):
    return type(value) is str and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def _initial_history(history):
    _messages(history)
    _require(len(history) == len(TUTORIAL) + 2
             and history[0] == {'role': 'system', 'content': SYSTEMS['appworld-v3']}
             and history[1:-1] == TUTORIAL and history[-1]['role'] == 'user')


def _public_gate(gate, origins):
    _require(type(gate) is dict and set(gate) == _GATE
             and gate['gate_kind'] == 'codeact_within_workflow_gate' and gate['qualified'] is True
             and type(gate['episode_count']) is int and gate['episode_count'] == 12
             and type(gate['known_episode_count']) is int and gate['known_episode_count'] == 12
             and type(gate['eligible_scene_count']) is int and 4 <= gate['eligible_scene_count'] <= 12
             and type(gate['unique_target_count']) is int and gate['unique_target_count'] >= 20
             and _integer(gate['duplicate_target_count'])
             and gate['canonicalization'] == 'syntactic_ast_not_semantic_distinctness'
             and gate['authentication'] == 'external_precondition_not_performed_here'
             and type(gate['sources']) is list and len(gate['sources']) == 12
             and type(gate['targets']) is list and len(gate['targets']) == gate['unique_target_count'])
    _require(type(origins) is list and len(origins) == len(set(origins)) == 12
             and all(type(item) is str and re.fullmatch(r'[a-f0-9]{7}', item) for item in origins))
    for index, item in enumerate(gate['sources']):
        _require(type(item) is dict and set(item) == _SOURCE
                 and type(item['source_index']) is int and item['source_index'] == index
                 and item['task_id_sha256'] == _hash(origins[index] + '_3')
                 and _sha(item['episode_sha256']) and type(item['eligible']) is bool
                 and item['reason'] == ('eligible' if item['eligible'] else 'known_ineligible')
                 and all(_integer(item[key]) for key in
                         ('eligible_error_count', 'target_candidate_count', 'unique_target_count'))
                 and item['unique_target_count'] <= item['target_candidate_count']
                 and (item['eligible_error_count'] > 0 and item['target_candidate_count'] > 0
                      if item['eligible'] else item['unique_target_count'] == 0))
    _require(sum(item['eligible'] for item in gate['sources']) == gate['eligible_scene_count'])
    observed_counts = [0] * 12
    hashes, canonicals, positions = set(), set(), []
    for target in gate['targets']:
        _require(type(target) is dict and set(target) == _TARGET
                 and type(target['source_index']) is int and 0 <= target['source_index'] < 12
                 and gate['sources'][target['source_index']]['eligible'] is True
                 and type(target['target_index']) is int and 1 <= target['target_index'] < 40
                 and type(target['error_index']) is int and 0 <= target['error_index'] < target['target_index']
                 and all(_sha(target[key]) for key in
                         ('input_sha256', 'messages_sha256', 'raw_response_sha256', 'canonical_sha256'))
                 and target['input_sha256'] not in hashes and target['canonical_sha256'] not in canonicals)
        hashes.add(target['input_sha256'])
        canonicals.add(target['canonical_sha256'])
        observed_counts[target['source_index']] += 1
        positions.append((target['source_index'], target['target_index']))
    _require(positions == sorted(set(positions))
             and observed_counts == [item['unique_target_count'] for item in gate['sources']]
             and gate['duplicate_target_count'] == sum(
                 item['target_candidate_count'] - item['unique_target_count']
                 for item in gate['sources'] if item['eligible']))


def _row_history(row, target, initial):
    _require(type(row) is dict and set(row) == _ROW
             and all(type(row[key]) is str for key in ('raw_response', 'action', 'observation'))
             and type(row['target_index']) is int and row['target_index'] == target['target_index']
             and type(row['error_index']) is int and row['error_index'] == target['error_index']
             and _hash(row) == target['input_sha256']
             and _hash(row['raw_response']) == target['raw_response_sha256']
             and extract_code(row['raw_response']) == row['action'])
    tree = ast.parse(row['action'])  # Parse syntax only; never execute model code.
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    _require(_hash(canonical) == target['canonical_sha256'])
    history = row['messages_before']
    _messages(history)
    _require(type(initial) is list and len(history) == len(initial) + 2 * row['target_index']
             and history[:len(initial)] == initial and _hash(history) == target['messages_sha256']
             and all(history[len(initial) + i]['role'] == ('assistant' if i % 2 == 0 else 'user')
                     for i in range(2 * row['target_index'])))
    result = copy.deepcopy(history)
    result.extend([{'role': 'assistant', 'content': row['raw_response']},
                   {'role': 'user', 'content': row['observation']}])
    _messages(result)
    return result


def prepare_writer_inputs(gate_public, writer_rows, initial_histories, origin_ids):
    """Return private contexts and a content-free deterministic selection projection.

    ``initial_histories`` and ``origin_ids`` are twelve-item lists ordered by
    source index; an unused initial history may be None. All hashes use the
    original source gate's JSON hash. The caller remains responsible for proving
    that these arguments are the frozen, authenticated source outputs.
    """
    try:
        _public_gate(gate_public, origin_ids)
        _require(type(initial_histories) is list and len(initial_histories) == 12
                 and type(writer_rows) is list and len(writer_rows) == len(gate_public['targets']))
        for history in initial_histories:
            if history is not None:
                _initial_history(history)
        by_hash = {}
        for row in writer_rows:
            _require(type(row) is dict and set(row) == _ROW)
            digest = _hash(row)
            _require(digest not in by_hash)
            by_hash[digest] = row
        _require(set(by_hash) == {item['input_sha256'] for item in gate_public['targets']})
        chosen = {}
        for target in gate_public['targets']:
            index = target['source_index']
            initial = initial_histories[index]
            _require(initial is not None)
            history = _row_history(by_hash[target['input_sha256']], target, initial)
            if index not in chosen:
                initial_length = len(initial)
                projection = {'source_index': index, 'error_index': target['error_index'],
                    'target_index': target['target_index'], 'input_sha256': target['input_sha256'],
                    'initial_history_sha256': _hash(initial), 'history_sha256': _hash(history),
                    'task_message_index': initial_length - 1, 'evidence_start_index': initial_length,
                    'evidence_end_exclusive': len(history),
                    'error_observation_index': initial_length + 2 * target['error_index'] + 1,
                    'target_observation_index': initial_length + 2 * target['target_index'] + 1}
                chosen[index] = {'history': history,
                    'lineage': {**projection, 'source_scenario': origin_ids[index]}}
        indices = sorted(chosen)
        _require(bool(indices))
        selections = [{key: value for key, value in chosen[index]['lineage'].items()
                       if key != 'source_scenario'} for index in indices]
        public = {'gate_kind': gate_public['gate_kind'], 'gate_sha256': _hash(gate_public),
            'writer_rows_sha256': _hash(writer_rows), 'origin_ids_sha256': _hash(origin_ids),
            'eligible_source_count': gate_public['eligible_scene_count'],
            'qualified_target_count': gate_public['unique_target_count'], 'selected_source_count': len(indices),
            'skipped_source_indices': [index for index in range(12) if index not in chosen],
            'eligible_without_unique_target_indices': [index for index in range(12)
                if gate_public['sources'][index]['eligible'] and index not in chosen],
            'selections': selections}
        private = {'origin_ids': list(origin_ids), 'contexts': [chosen[index] for index in indices]}
        return {'public': public, 'private': private}
    except Exception:
        raise ValueError(_ERROR) from None
