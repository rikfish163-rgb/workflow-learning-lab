"""Read-only, terminal-only export preparation for the fixed CodeAct source.

Import performs no I/O. Production reads the same desktop data4 mounts as the
source parent; injected dependencies are only for synthetic CPU tests. Only
private.contexts[*].history is writer model input. The preparation manifest is
provenance metadata, and no downstream registration or model call happens here.
"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ID = 'codeact-within-source-export-001'
SOURCE_ID = 'qwen38-codeact-within-source-001'
SOURCE_PLAN = 'artifacts/qwen38-codeact-within-source-plan-001.json'
SOURCE_PLAN_SHA256 = 'd97afc544dc53ae35af596776c2c5d2b15287677da32d241e711d9ee6849a41b'
ERROR = 'codeact_source_export_invalid'
_PREPARATION = {'gate_kind', 'gate_sha256', 'writer_rows_sha256', 'origin_ids_sha256',
                'eligible_source_count', 'qualified_target_count', 'selected_source_count',
                'skipped_source_indices', 'eligible_without_unique_target_indices', 'selections'}
_SELECTION = {'source_index', 'error_index', 'target_index', 'input_sha256',
              'initial_history_sha256', 'history_sha256', 'task_message_index',
              'evidence_start_index', 'evidence_end_exclusive', 'error_observation_index',
              'target_observation_index'}


def _require(value):
    if not value:
        raise ValueError(ERROR)


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _dependencies():
    from scripts import codeact_within_registration_001 as registration
    from scripts import run_codeact_within_source_001 as coordinator
    from scripts import run_codeact_within_source_study_001 as parent
    from scripts.codeact_within_sources_001 import qualify_sources
    from scripts.codeact_within_writer_inputs_001 import prepare_writer_inputs
    from scripts.remote_runtime_storage import verify_storage
    _require(parent.ROOT == coordinator.ROOT == registration.ROOT == ROOT
             and parent.OUTPUT == coordinator.OUTPUT == registration.OUTPUT
             and parent.PRIVATE == coordinator.PRIVATE == registration.PRIVATE)
    return SimpleNamespace(registration=registration, coordinator=coordinator, parent=parent,
                           qualify_sources=qualify_sources, prepare_writer_inputs=prepare_writer_inputs,
                           verify_storage=verify_storage)


def prepare_source_export(expected_report_sha256: str, *, _deps=None) -> dict:
    """Reauthenticate all twelve sources and return private writer preparation.

    The caller supplies the exact terminal report byte hash. This function never
    writes files and never promotes an incomplete or partial source snapshot.
    """
    try:
        return _prepare(expected_report_sha256, _deps or _dependencies())
    except Exception:
        raise ValueError(ERROR) from None


def _prepare(expected, deps):
    _require(_sha(expected))
    reg, c, parent = deps.registration, deps.coordinator, deps.parent
    root, output, private = parent.ROOT, parent.OUTPUT, parent.PRIVATE
    deps.verify_storage(root, reg.TMPDIR)
    _require(reg.PLAN == SOURCE_PLAN and len(reg.TASK_IDS) == 12)
    snapshots = []
    def read(path, digest=None):
        raw, observed = c._read(path, expected=digest, maximum=128_000_000)
        snapshots.append((path, observed))
        return json.loads(raw, object_pairs_hook=c._pairs, parse_constant=c._reject_number), observed
    report, _ = read(output / 'report.json', expected)
    _require(report['id'] == SOURCE_ID and report['plan_sha256'] == SOURCE_PLAN_SHA256
             and report['status'] == 'complete' and report['automatic_retry'] is False
             and report['utility_evaluated'] is False and report['unattempted_batches'] == []
             and type(report['fixed_denominator']) is int and report['fixed_denominator'] == 12
             and type(report['sources']) is list and len(report['sources']) == 12
             and all(row['auth_complete'] is row['costs_known'] is row['attempted'] is True
                     and type(row['source_index']) is int and row['source_index'] == index
                     and row['task_id'] == reg.TASK_IDS[index]
                     and type(row['official_success']) is type(row['model_reported_complete']) is bool
                     for index, row in enumerate(report['sources']))
             and type(report['batches']) is list and len(report['batches']) == 4
             and all(row['auth_complete'] is row['accepted'] is True and row['status'] == 'complete'
                     and all(row['release'][key] is True for key in ('cpu_client', 'containers', 'model_lease'))
                     for row in report['batches'])
             and report['costs']['all_costs_known'] is True and report['qualification']['qualified'] is True
             and type(report['qualification']['known_episode_count']) is int
             and report['qualification']['known_episode_count'] == 12)
    # Incomplete public snapshots must not trigger even private Task-asset hashes.
    plan = reg.load_registration(root, SOURCE_PLAN_SHA256, private=True)
    reg.check_predecessor(root, plan)
    _require(plan['id'] == SOURCE_ID and plan['task_ids'] == list(reg.TASK_IDS)
             and plan['batches'] == reg.BATCHES == [plan['task_ids'][n:n + 3] for n in range(0, 12, 3)]
             and plan['private_directory'] == str(private))
    who = parent._identity(report['parent'])
    _require(_sha(report['study_intent_sha256']))
    intent, intent_sha = read(private / 'study-intent.json', report['study_intent_sha256'])
    _require(parent._same(intent, dict(id=SOURCE_ID, plan_sha256=SOURCE_PLAN_SHA256, parent=who,
                                      started_epoch=report['started_epoch'], automatic_retry=False)))
    raw, journal_sha = c._read(output / 'events.jsonl')
    snapshots.append((output / 'events.jsonl', journal_sha))
    _require(raw and raw.endswith(b'\n'))
    journal = [json.loads(line, object_pairs_hook=c._pairs, parse_constant=c._reject_number)
               for line in raw.splitlines()]
    kinds = ['study_start'] + [kind for _ in range(4) for kind in ('batch_start', 'batch_complete')] + ['study_stop']
    _require(len(journal) == len(kinds)
             and all(row['event'] == kinds[n] and type(row['event_index']) is int and row['event_index'] == n
                     and row['plan_sha256'] == SOURCE_PLAN_SHA256 and type(row['epoch']) in (int, float)
                     and math.isfinite(row['epoch']) and row['epoch'] >= 0 for n, row in enumerate(journal))
             and parent._same(journal[0]['parent'], who) and journal[-1]['status'] == 'complete'
             and all(type(journal[1 + 2 * n + k]['batch_index']) is int
                     and journal[1 + 2 * n + k]['batch_index'] == n for n in range(4) for k in (0, 1)))
    envelopes, sources, batches = [], [], []
    for index in range(4):
        lifecycle, _ = read(output / f'batch-{index}/lifecycle.json', report['batches'][index]['lifecycle_sha256'])
        group, rows, batch = parent._inspect_batch(deps, plan, SOURCE_PLAN_SHA256, index, lifecycle, who)
        envelopes.extend(group)
        sources.extend(rows)
        batches.append(batch)
    _require(parent._same(report['sources'], sources) and parent._same(report['batches'], batches)
             and parent._same(report['costs'], parent._costs({'sources': sources, 'batches': batches})))
    qualified = deps.qualify_sources(envelopes)
    _require(set(qualified) == {'public', 'writer_inputs'} and qualified['public']['qualified'] is True
             and parent._same(qualified['public'], report['qualification']))
    binding = report['writer_inputs']
    _require(type(binding) is dict and set(binding) == {'file', 'sha256', 'count'}
             and binding['file'] == 'writer-inputs.json' and _sha(binding['sha256'])
             and type(binding['count']) is int and binding['count'] == len(qualified['writer_inputs']) >= 20)
    writer, writer_sha = read(private / 'writer-inputs.json', binding['sha256'])
    _require(parent._same(writer, qualified['writer_inputs']))
    histories = [copy.deepcopy(item['private']['steps'][0]['messages_before'])
                 if item['private']['steps'] else None for item in envelopes]
    origins = [task.rsplit('_', 1)[0] for task in plan['task_ids']]
    # Keep authenticated source values separate from the called preparer's data.
    inputs = copy.deepcopy((qualified['public'], writer, histories, origins))
    prepared = deps.prepare_writer_inputs(*inputs)
    _require(type(prepared) is dict and set(prepared) == {'public', 'private'}
             and type(prepared['public']) is dict and set(prepared['public']) == _PREPARATION
             and type(prepared['private']) is dict and set(prepared['private']) == {'contexts', 'origin_ids'}
             and parent._same(prepared['private']['origin_ids'], origins))
    metadata, contexts = prepared['public'], prepared['private']['contexts']
    _require(metadata['gate_kind'] == qualified['public']['gate_kind']
             and metadata['gate_sha256'] == _hash(qualified['public'])
             and metadata['writer_rows_sha256'] == _hash(writer) and metadata['origin_ids_sha256'] == _hash(origins)
             and metadata['eligible_source_count'] == qualified['public']['eligible_scene_count']
             and metadata['qualified_target_count'] == binding['count']
             and type(metadata['selected_source_count']) is int and 1 <= metadata['selected_source_count'] <= 12
             and type(contexts) is list and type(metadata['selections']) is list
             and len(contexts) == len(metadata['selections']) == metadata['selected_source_count'])
    first_targets = {}
    for target in qualified['public']['targets']:
        first_targets.setdefault(target['source_index'], target)
    indices = [item['source_index'] for item in metadata['selections']]
    _require(parent._same(indices, sorted(first_targets))
             and parent._same(metadata['skipped_source_indices'], [i for i in range(12) if i not in first_targets])
             and parent._same(metadata['eligible_without_unique_target_indices'], [i for i in range(12)
                              if qualified['public']['sources'][i]['eligible'] and i not in first_targets]))
    by_hash = {_hash(row): row for row in writer}
    for selection, context in zip(metadata['selections'], contexts, strict=True):
        _require(type(selection) is dict and set(selection) == _SELECTION
                 and type(selection['source_index']) is int and 0 <= selection['source_index'] < 12
                 and all(_sha(selection[key]) for key in ('input_sha256', 'initial_history_sha256', 'history_sha256'))
                 and all(type(selection[key]) is int and selection[key] >= 0 for key in _SELECTION
                         - {'input_sha256', 'initial_history_sha256', 'history_sha256'})
                 and type(context) is dict and set(context) == {'history', 'lineage'})
        index = selection['source_index']
        target, row = first_targets[index], by_hash[selection['input_sha256']]
        _require(all(selection[key] == target[key] for key in ('input_sha256', 'error_index', 'target_index'))
                 and selection['initial_history_sha256'] == _hash(histories[index]))
        actor_history = row['messages_before'] + [{'role': 'assistant', 'content': row['raw_response']},
                                                  {'role': 'user', 'content': row['observation']}]
        _require(parent._same(context['lineage'], dict(**selection, source_scenario=origins[index]))
                 and parent._same(context['history'], actor_history)
                 and type(context['history']) is list and _hash(context['history']) == selection['history_sha256']
                 and all(type(msg) is dict and set(msg) == {'role', 'content'}
                         and msg['role'] in ('system', 'user', 'assistant') and type(msg['content']) is str
                         for msg in context['history']))
    for path, digest in snapshots:
        c._read(path, expected=digest, maximum=128_000_000)
    proof = 'artifacts/' + SOURCE_ID + '/report.json'
    auth = dict(study_id=SOURCE_ID, plan=SOURCE_PLAN, plan_sha256=SOURCE_PLAN_SHA256,
                qualification_proof=proof, report_sha256=expected, study_intent_sha256=intent_sha,
                parent_journal_sha256=journal_sha, writer_inputs_sha256=writer_sha, writer_input_count=binding['count'],
                gate_sha256=_hash(qualified['public']), costs_sha256=_hash(report['costs']),
                episode_count=12, qualified=True, all_costs_known=True, origin_ids=origins,
                batches=[{key: item[key] for key in ('batch_index', 'lifecycle_sha256', 'report_sha256', 'release')}
                         for item in batches])
    return {'public': dict(id=ID, status='authenticated_writer_preparation_only',
                           source_authentication=copy.deepcopy(auth), preparation=copy.deepcopy(metadata)),
            'private': dict(contexts=copy.deepcopy(contexts), preparation_manifest={
                'origin_ids': list(origins), 'source_authentication': copy.deepcopy(auth),
                'selection': copy.deepcopy(metadata)})}
