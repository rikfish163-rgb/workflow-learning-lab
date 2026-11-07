"""Private authentication of completed writer and fallible local content review.

No model, worker, scorer, task or queue operation is started. Complete source
qualification is recomputed once, then recorded writer/reviewer calls are
replayed against their recorded responses in CPU memory. A reviewer verdict is
only a hypothesis worth testing on the full frozen utility cohort. It is never
correctness, privacy-completeness, training or learning-gain certification.

Returned text/manifests are protected. Only an explicit caller may persist them
under the writer private root. Importing this module reads no private payload.
"""

import copy
import hashlib
import math
from pathlib import Path

from scripts import native_experience_content_001 as content
from scripts import native_experience_registration_001 as registration
from scripts import native_experience_writer_001 as writer
from scripts.native_appworld_worker_client import _decode, _encode
from scripts.verification_token_counter import _message_payload

_ERROR = 'native_experience_review_invalid'
_SCOPE = 'fallible_local_model_review_pending_utility_not_correctness_or_privacy_certification'


def _require(value):
    if not value:
        raise ValueError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _same(left, right):
    return _encode(left) == _encode(right)


def _journal(raw):
    _require(type(raw) is bytes and raw.endswith(b'\n'))
    result = [_decode(line) for line in raw.splitlines()]
    _require(all(type(row) is dict and type(row.get('event')) is str for row in result))
    return result


def _one(rows, event, index=None):
    found = [row for row in rows if row['event'] == event
             and (index is None or row.get('source_index') == index)]
    _require(len(found) == 1)
    return found[0]


def _private_read(root, path):
    """Reuse descriptor-relative strict readers for the two fixed private roots."""
    from scripts import native_source_episode_006 as source
    from scripts import run_native_experience_writer_001 as coordinator

    path = Path(path)
    writer_root = Path(registration.PRIVATE_RUN_ROOT)
    source_reader = source._get_core()
    source_root = Path(source_reader.PRIVATE_RUN_ROOT)
    _require(path.is_absolute() and '..' not in path.parts)
    if writer_root in path.parents:
        return coordinator._services(root)[1]._read_private(path, maximum=64_000_000)
    _require(source_root in path.parents)
    return source_reader._read_private(path, maximum=64_000_000)


def _rehash(root, bindings):
    return all(_hash(_private_read(root, path)) == digest for path, digest in bindings.items())


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class _RecordedModel:
    """CPU evidence checker: returns a recorded response, never invokes a model."""

    def __init__(self, events, *, messages, seed, max_tokens, prefix, backend, model_id):
        self.events, self.messages, self.seed = events, messages, seed
        self.max_tokens, self.prefix = max_tokens, prefix
        self.backend, self.model_id = backend, model_id

    def check_health(self):
        _one(self.events, self.prefix + '_health_checked')

    def count_input_tokens(self, messages):
        _require(_same(messages, self.messages))
        return _one(self.events, self.prefix + '_count_result')['input_tokens']

    def generate(self, messages, seed):
        _require(_same(messages, self.messages) and seed == self.seed)
        intent = _one(self.events, 'generation_intent')
        _require(_same(intent['messages'], messages) and intent['seed'] == seed
                 and intent['max_new_tokens'] == self.max_tokens)
        response = _one(self.events, 'generation_result')['response']
        raw = response['raw_reply']
        count = _one(self.events, self.prefix + '_count_result')['input_tokens']
        _require(set(response) == {'text', 'prompt_tokens', 'completion_tokens', 'format_error', 'backend', 'raw_reply'}
                 and response['format_error'] is None and raw['model'] == self.model_id
                 and raw.get('format_error') is None
                 and _same(response['backend'], self.backend) and _same(raw['backend'], self.backend)
                 and type(raw['choices']) is list and len(raw['choices']) == 1
                 and type(raw['choices'][0]) is dict
                 and raw['choices'][0]['finish_reason'] in {'stop', 'length', 'unknown'}
                 and type(raw['usage']) is dict and set(raw['usage']) == {'prompt_tokens', 'completion_tokens'}
                 and type(raw['usage']['prompt_tokens']) is int
                 and raw['usage']['prompt_tokens'] == count == response['prompt_tokens']
                 and type(raw['usage']['completion_tokens']) is int
                 and raw['usage']['completion_tokens'] == response['completion_tokens']
                 and 1 <= response['completion_tokens'] <= self.max_tokens
                 and count + self.max_tokens <= 16384
                 and raw['choices'][0]['message'] == {'role': 'assistant', 'content': response['text']})
        return copy.deepcopy(response)


def _replay(events, messages, *, seed, backend, model_id, reviewer=False):
    _require(type(events) is list and bool(events))
    prefix = 'review' if reviewer else 'writer'
    model = _RecordedModel(events, messages=messages, seed=seed,
                           max_tokens=256 if reviewer else 128, prefix=prefix,
                           backend=backend, model_id=model_id)
    recorded = []
    function = content.review_once if reviewer else writer.write_once
    result = function(messages, model, seed=seed, record=lambda row: recorded.append(copy.deepcopy(row)))
    _require(result['status'] != 'unknown' and result['cost']['unknown_cost'] is False
             and result['accepted'] is False and result['content_certified'] is False
             and result['training_authorized'] is False and _same(recorded, events))
    return result


def prepare_reviewed_manifest(root, writer_plan_sha256, writer_result_sha256):
    """Recompute a private manifest without writing it or accepting a public flag."""
    try:
        _require(registration.public._sha(writer_plan_sha256)
                 and registration.public._sha(writer_result_sha256))
        plan, qualified = registration.private_preflight(root, writer_plan_sha256)
        from scripts import run_native_experience_writer_001 as coordinator

        coordinator._qualified(plan, qualified)
        base = Path(plan['private_run_root'])
        _require(base == Path(registration.PRIVATE_RUN_ROOT))
        bindings = dict(qualified['input_manifest']['private_files_sha256'])

        def read(name):
            path = base / name
            raw = _private_read(root, path)
            digest = _hash(raw)
            _require(str(path) not in bindings or bindings[str(path)] == digest)
            bindings[str(path)] = digest
            return raw

        raw = read('study-result.json')
        _require(_hash(raw) == writer_result_sha256)
        report = _decode(raw)
        _require(report['id'] == registration.ID and report['plan_sha256'] == writer_plan_sha256
                 and report['status'] == 'complete' and report['automatic_retry'] is False
                 and type(report['training_jobs']) is int and report['training_jobs'] == 0
                 and type(report['utility_or_dev_or_test_calls']) is int
                 and report['utility_or_dev_or_test_calls'] == 0)
        rows = _journal(read('study-journal.jsonl'))
        _require(_same(_one(rows, 'study_stop')['report'], report))
        intent = _decode(read('study-intent.json'))
        inputs = qualified['writer_inputs']
        metadata = [{key: item[key] for key in ('source_index', 'source_scenario', 'messages_sha256')}
                    for item in inputs]
        expected_initial = [{**item, 'status': 'unstarted', 'review_status': 'unstarted'} for item in metadata]
        _require(intent['plan_sha256'] == writer_plan_sha256 and intent['automatic_retry'] is False
                 and _same(intent['sources'], expected_initial)
                 and len(report['sources']) == len(inputs))
        manifest_raw = read('input-manifest.json')
        _require(_same(_decode(manifest_raw), qualified['input_manifest']))
        frozen = _one(rows, 'writer_inputs_frozen')
        _require(frozen['manifest'] == {'path': str(base / 'input-manifest.json'),
                                       'sha256': _hash(manifest_raw)}
                 and len(frozen['inputs']) == len(inputs))
        costs, jobs = report['costs'], report['costs']['GPU_jobs']
        _require(len(jobs) == 1 and jobs[0]['unknown_cost'] is False
                 and jobs[0]['release']['verified'] is True
                 and jobs[0]['release'].get('runtime_overrun') is False
                 and _seconds(jobs[0]['release']['GPU_seconds'])
                 and jobs[0]['release']['GPU_seconds'] <= plan['gpu_seconds']
                 and _one(rows, 'GPU_enqueue_intent')['job'] == jobs[0]['job']
                 and _one(rows, 'GPU_ready')['job'] == jobs[0]['job'])
        from scripts import native_experience_gpu_queue_001 as queue

        _require(_same(jobs[0]['job'], queue.make_job(0, jobs[0]['job']['owner'], writer_plan_sha256)))
        backend = _one(rows, 'GPU_ready')['service']['backend']
        _require(_same(registration.public.static_backend(backend), plan['expected_backend_static']['B3']))
        _verify_schedule_and_costs(rows, report, plan, [item['source_index'] for item in inputs])
        _require(len(costs['writer_calls']) == len(inputs))
        candidate_indices, decisions, texts = [], [], []
        for item, slot, frozen_input, writer_cost in zip(
                inputs, report['sources'], frozen['inputs'], costs['writer_calls'], strict=True):
            index = item['source_index']
            _require({key: slot[key] for key in metadata[0]} == {key: item[key] for key in metadata[0]})
            input_raw = read(f'writer-input-{index:02d}.json')
            _require(_same(_decode(input_raw), item)
                     and _same(frozen_input, {**{key: item[key] for key in metadata[0]},
                         'path': str(base / f'writer-input-{index:02d}.json'), 'sha256': _hash(input_raw)}))
            call = _one(rows, 'writer_call_intent', index)
            _require(call['seed'] == 17 + index and call['messages_sha256'] == item['messages_sha256'])
            events = [row['writer_event'] for row in rows
                      if row['event'] == 'writer_event' and row['source_index'] == index]
            result = _replay(events, item['messages'], seed=17 + index,
                             backend=backend, model_id=plan['model_id'])
            result_raw = read(f'writer-result-{index:02d}.json')
            _require(_same(_decode(result_raw), result)
                     and _one(rows, 'writer_call_finished', index)['result_binding'] == {
                         'path': str(base / f'writer-result-{index:02d}.json'), 'sha256': _hash(result_raw)}
                     and writer_cost['source_index'] == index and writer_cost['attempted'] is True
                     and writer_cost['unknown_cost'] is False and _seconds(writer_cost['wall_seconds'])
                     and writer_cost['wall_seconds'] <= plan['writer']['per_call_seconds']
                     and _same(writer_cost['result'], result) and slot['status'] == result['status'])
            if result['status'] != 'candidate_unvalidated':
                _require(slot['review_status'] == 'not_applicable'
                         and not any(row['event'] in {'review_event', 'review_call_intent', 'review_call_finished'}
                                     and row.get('source_index') == index for row in rows))
                decisions.append({'source_index': index, 'status': 'writer_abstained',
                                  'candidate_sha256': None, 'review_result_sha256': None})
                continue
            candidate_indices.append(index)
            candidate = result['candidate']
            screened = content.screen(item['messages'], candidate)
            review_raw = read(f'review-result-{index:02d}.json')
            reviewed = _decode(review_raw)
            text = content.insertion(candidate)
            expected = {'screen': screened, 'messages_sha256': None, 'result': None,
                        'status': 'rejected', 'candidate_sha256': _hash(_encode(candidate)),
                        'insertion_sha256': _hash(text.encode('utf-8'))}
            r_events = [row['review_event'] for row in rows
                        if row['event'] == 'review_event' and row['source_index'] == index]
            r_cost = [row for row in costs['review_calls'] if row['source_index'] == index]
            _require(len(r_cost) == 1 and r_cost[0]['unknown_cost'] is False
                     and _seconds(r_cost[0]['wall_seconds'])
                     and r_cost[0]['wall_seconds'] <= plan['writer']['content_review']['per_call_seconds'])
            if screened['status'] == 'passed':
                messages = content.build_review_messages(item['messages'], candidate)
                digest = _message_payload(messages)[1]
                r_intent = _one(rows, 'review_call_intent', index)
                _require(r_intent == {'event': 'review_call_intent', 'source_index': index,
                    'messages_sha256': digest, 'candidate_sha256': expected['candidate_sha256'],
                    'insertion_sha256': expected['insertion_sha256'], 'seed': 117 + index}
                    and r_cost[0]['attempted'] is True)
                reviewed_result = _replay(r_events, messages, seed=117 + index,
                                          backend=backend, model_id=plan['model_id'], reviewer=True)
                expected.update(messages_sha256=digest, result=reviewed_result,
                                status=reviewed_result['status'])
            else:
                _require(screened['status'] == 'rejected' and not r_events
                         and r_cost[0]['attempted'] is False
                         and not any(row['event'] == 'review_call_intent' and row['source_index'] == index
                                     for row in rows))
            _require(_same(reviewed, expected) and _same(r_cost[0]['result'], reviewed)
                     and slot['review_status'] == reviewed['status']
                     and _one(rows, 'review_call_finished', index)['result_binding'] == {
                         'path': str(base / f'review-result-{index:02d}.json'), 'sha256': _hash(review_raw)})
            decisions.append({'source_index': index, 'status': reviewed['status'],
                              'candidate_sha256': expected['candidate_sha256'],
                              'review_result_sha256': _hash(review_raw)})
            if reviewed['status'] == 'plausible_pending_utility':
                texts.append((index, text))
        _require([row['source_index'] for row in costs['review_calls']] == candidate_indices and bool(texts))
        _require(not any(row.get('source_index') not in {item['source_index'] for item in inputs}
                         for row in rows if row['event'] in {
                             'writer_event', 'review_event', 'writer_call_intent', 'review_call_intent',
                             'writer_call_finished', 'review_call_finished'}))
        insertion = '\n\n'.join(text for _, text in texts)
        bundle_id = 'native-experience-001:' + _hash(insertion.encode('utf-8'))
        tasks = registration._source_plan(root)['task_ids']
        origins = sorted({task[:-2] for task in tasks})
        _require(len(tasks) == len(origins) == 12 and all(task.endswith('_2') for task in tasks)
                 and _rehash(root, bindings))
        manifest = {'id': 'native-experience-review-manifest-001', 'private': True,
                    'source_gate_kind': plan['source']['source_gate_kind'],
                    'source_plan_sha256': plan['source']['plan_sha256'],
                    'source_result_sha256': plan['source']['result_sha256'],
                    'writer_plan_sha256': writer_plan_sha256, 'writer_result_sha256': writer_result_sha256,
                    'private_files_sha256': bindings, 'candidate_decisions': decisions,
                    'retained_source_indices': [index for index, _ in texts],
                    'insertion_sha256': _hash(insertion.encode('utf-8')),
                    'experience_bundle_id': bundle_id, 'source_scenario_ids': origins, 'scope': _SCOPE}
        return {'manifest': manifest, 'insertion_text': insertion}
    except Exception:
        raise ValueError(_ERROR) from None


def load_reviewed_bundle(root, writer_plan_sha256, writer_result_sha256, review_manifest_sha256):
    """Authenticate frozen private review bytes and return exact insertion text."""
    try:
        _require(registration.public._sha(review_manifest_sha256))
        prepared = prepare_reviewed_manifest(root, writer_plan_sha256, writer_result_sha256)
        manifest = prepared['manifest']
        path = Path(registration.PRIVATE_RUN_ROOT) / 'review-manifest.json'
        raw = _private_read(root, path)
        _require(_hash(raw) == review_manifest_sha256 and _same(_decode(raw), manifest))
        bindings = {**manifest['private_files_sha256'], str(path): review_manifest_sha256}
        _require(_rehash(root, bindings))
        return {'private': True, 'status': 'plausible_pending_utility',
                **{key: manifest[key] for key in ('source_gate_kind', 'source_plan_sha256', 'source_result_sha256',
                    'writer_plan_sha256', 'writer_result_sha256', 'experience_bundle_id', 'source_scenario_ids')},
                'review_manifest_sha256': review_manifest_sha256,
                'insertion_text': prepared['insertion_text'], 'private_files_sha256': bindings}
    except Exception:
        raise ValueError(_ERROR) from None


def _verify_schedule_and_costs(rows, report, plan, indices):
    """Verify the complete outer schedule and bounded, known recorded costs.

    Actor-visible content and response semantics are checked by the separate
    replay. Heartbeat and successful release-fence events may interleave, but
    every writer/review call has one ordered source slot and the stop is final.
    """
    try:
        _require(type(rows) is list and bool(rows) and type(report) is dict and type(plan) is dict
                 and type(indices) in (list, tuple) and 1 <= len(indices) <= 12
                 and all(type(index) is int and 0 <= index < 12 for index in indices)
                 and list(indices) == sorted(set(indices))
                 and report.get('status') == 'complete' and report.get('automatic_retry') is False
                 and _same(plan.get('writer'), registration.writer_policy())
                 and type(plan.get('gpu_seconds')) is int and plan['gpu_seconds'] == 7200)
        policy = plan['writer']
        sources, costs = report['sources'], report['costs']
        _require(type(sources) is list and len(sources) == len(indices) and type(costs) is dict
                 and all(type(slot) is dict and type(slot.get('source_index')) is int
                         and slot['source_index'] == index for slot, index in zip(sources, indices, strict=True)))
        writers, reviewers = costs['writer_calls'], costs['review_calls']
        _require(type(writers) is list and len(writers) == len(indices) and type(reviewers) is list)
        candidates = [slot['source_index'] for slot in sources if slot.get('status') == 'candidate_unvalidated']
        _require(all(slot.get('status') in {'candidate_unvalidated', 'abstained'} for slot in sources)
                 and len(reviewers) == len(candidates)
                 and all(type(entry) is dict and type(entry.get('source_index')) is int
                         and entry['source_index'] == index for entry, index in zip(reviewers, candidates, strict=True)))

        def seconds(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                return False

        def known_attempt(result):
            if type(result) is not dict or type(result.get('cost')) is not dict:
                return False
            cost = result['cost']
            return (cost.get('unknown_cost') is False and type(cost.get('generation_attempts')) is int
                    and 0 <= cost['generation_attempts'] <= 1
                    and type(cost.get('generation_results')) is int
                    and cost['generation_results'] == cost['generation_attempts'])

        call_seconds = 0.0
        review_by_index = {entry['source_index']: entry for entry in reviewers}
        for slot, entry, index in zip(sources, writers, indices, strict=True):
            _require(type(entry) is dict and type(entry.get('source_index')) is int and entry['source_index'] == index
                     and entry.get('attempted') is True and entry.get('unknown_cost') is False
                     and seconds(entry.get('wall_seconds')) and entry['wall_seconds'] <= policy['per_call_seconds']
                     and known_attempt(entry.get('result')) and entry['result'].get('status') == slot['status'])
            if slot['status'] == 'candidate_unvalidated':
                _require(entry['result']['cost']['generation_attempts'] == 1)
            else:
                _require(slot.get('review_status') == 'not_applicable')
            call_seconds += entry['wall_seconds']
            if index not in review_by_index:
                continue
            item = review_by_index[index]
            _require(type(item.get('attempted')) is bool and item.get('unknown_cost') is False
                     and seconds(item.get('wall_seconds'))
                     and item['wall_seconds'] <= policy['content_review']['per_call_seconds']
                     and type(item.get('result')) is dict and item['result'].get('status') == slot.get('review_status'))
            if item['attempted']:
                nested = item['result'].get('result')
                _require(slot.get('review_status') in {'plausible_pending_utility', 'rejected', 'abstained'}
                         and known_attempt(nested))
                if slot['review_status'] == 'plausible_pending_utility':
                    _require(nested['cost']['generation_attempts'] == 1)
            else:
                _require(slot.get('review_status') == 'rejected' and item['result'].get('result') is None)
            call_seconds += item['wall_seconds']

        audits = costs['source_verifications']
        _require(type(audits) is list and len(audits) == len(indices) + 1)
        fields = {'stage', 'wall_seconds', 'returned', 'timeout_seconds', 'child_returncode',
                  'stdout_bytes', 'stdout_sha256', 'stderr_bytes', 'stderr_sha256'}
        verification_seconds = 0.0
        for i, item in enumerate(audits):
            _require(type(item) is dict and set(item) == fields
                     and item['stage'] == ('preflight' if i == 0 else 'hash')
                     and item['returned'] is True and type(item['child_returncode']) is int
                     and item['child_returncode'] == 0 and seconds(item['wall_seconds'])
                     and seconds(item['timeout_seconds']) and item['timeout_seconds'] > 0
                     and item['wall_seconds'] <= item['timeout_seconds']
                     and item['timeout_seconds'] <= policy['source_verification_seconds'] - verification_seconds + 1e-6
                     and type(item['stdout_bytes']) is int and 1 <= item['stdout_bytes'] <= 32_000_000
                     and type(item['stderr_bytes']) is int and item['stderr_bytes'] >= 0
                     and registration.public._sha(item['stdout_sha256'])
                     and registration.public._sha(item['stderr_sha256']))
            verification_seconds += item['wall_seconds']
        _require(verification_seconds <= policy['source_verification_seconds']
                 and seconds(costs.get('parent_wall_seconds'))
                 and costs['parent_wall_seconds'] >= verification_seconds + call_seconds)
        jobs = costs['GPU_jobs']
        _require(type(jobs) is list and len(jobs) == 1 and type(jobs[0]) is dict
                 and jobs[0].get('unknown_cost') is False and type(jobs[0].get('release')) is dict
                 and jobs[0]['release'].get('verified') is True
                 and jobs[0]['release'].get('runtime_overrun') is False
                 and seconds(jobs[0]['release'].get('GPU_seconds'))
                 and jobs[0]['release']['GPU_seconds'] <= plan['gpu_seconds'])

        _require(all(type(row) is dict and type(row.get('event')) is str for row in rows)
                 and rows[-1].get('event') == 'study_stop' and _same(rows[-1].get('report'), report))
        background = {'GPU_heartbeat', 'GPU_released', 'release_fence_started', 'release_fence_monitor_witness'}
        events = [row for row in rows if row['event'] not in background]
        cursor = 0

        def consume(event, index=None):
            nonlocal cursor
            _require(cursor < len(events) and events[cursor]['event'] == event)
            if index is not None:
                _require(type(events[cursor].get('source_index')) is int
                         and events[cursor]['source_index'] == index)
            cursor += 1

        def consume_events(event, index):
            nonlocal cursor
            start = cursor
            while cursor < len(events) and events[cursor]['event'] == event:
                row = events[cursor]
                _require(type(row.get('source_index')) is int and row['source_index'] == index
                         and type(row.get(event)) is dict and type(row[event].get('event')) is str)
                cursor += 1
            _require(cursor > start)

        for event in ('writer_inputs_frozen', 'GPU_enqueue_intent', 'GPU_ready'):
            consume(event)
        for index in indices:
            consume('writer_call_intent', index)
            consume_events('writer_event', index)
            consume('writer_call_finished', index)
            if index in review_by_index:
                if review_by_index[index]['attempted']:
                    consume('review_call_intent', index)
                    consume_events('review_event', index)
                consume('review_call_finished', index)
        consume('study_stop')
        _require(cursor == len(events))
    except Exception:
        raise ValueError(_ERROR) from None
