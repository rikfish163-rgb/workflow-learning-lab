"""Rebuild a model-proposed origin prefix without generating or adding actions."""
import json
import time
from dataclasses import asdict
from pathlib import Path

from scripts.report_timed_tasks import finite_time
from scripts.run_foundation_pilot import digest, read, require
from scripts.run_timed_tasks import CleanupTrackedWorld, RolloutDeadline
from scripts.semantic_repair_proposal import project_failure, validate_proposal
from wll.protocol import fingerprint
from wll.runner import ReplayMismatch
from wll.storage import atomic_json, require_space


def identity(source, proposal, projection, environment_id):
    require(proposal == validate_proposal(proposal['proposal'], projection)
            and proposal['proposal']['status'] == 'repair'
            and proposal['source_hash'] == fingerprint(source.to_dict())
            and source.metadata.get('environment_id') == environment_id, 'proposal source or environment differs')
    length = proposal['proposal']['prefix_length']
    return {'source_hash': fingerprint(source.to_dict()), 'proposal_sha256': fingerprint(proposal),
        'projection_sha256': fingerprint(projection), 'prefix_length': length,
        'prefix_sha256': fingerprint([asdict(s) for s in source.steps[:length]]),
        'environment_id': environment_id, 'script_sha256': digest(__file__)}


def replay_prefix(source, proposal, projection, world_factory, *, output, origin_scenarios,
                  environment_id, seconds=120):
    require(projection == project_failure(source, origin_scenarios, max_chars=32768), 'visible source differs')
    expected = identity(source, proposal, projection, environment_id)
    deadline = RolloutDeadline(seconds)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'manifest.json', {**expected, 'origin_scenarios': sorted(origin_scenarios),
        'seconds': seconds, 'LLM_calls': 0, 'new_actions': 0,
        'deadline_scope': 'replay and terminal scoring; environment startup and cleanup separately bounded'})
    stats = {'status': 'running', 'attempted_steps': 0, 'returned_observations': 0,
        'verified_observations': 0, 'prefix_success': None, 'prefix_completed': None,
        'cleanup_verified': False, 'LLM_calls': 0, 'new_actions': 0}
    atomic_json(output / 'summary.json', stats)
    world = None
    started = time.monotonic()
    try:
        with (output / 'steps.jsonl').open('x') as journal:
            world = CleanupTrackedWorld(world_factory(source.task), deadline)
            with deadline:
                for index, step in enumerate(source.steps[:expected['prefix_length']]):
                    stats['attempted_steps'] += 1
                    before = time.monotonic()
                    observation = world.execute(step.action)
                    elapsed = time.monotonic() - before
                    require(isinstance(observation, str), 'non-text replay observation')
                    matched = observation == step.observation
                    journal.write(json.dumps({'index': index, 'action': step.action, 'observation': observation,
                        'observation_matches': matched, 'execute_seconds': elapsed}, ensure_ascii=False) + '\n')
                    journal.flush()
                    stats['returned_observations'] += 1
                    stats['verified_observations'] += int(matched)
                    if not matched:
                        raise ReplayMismatch('proposed prefix observation differs')
                completed, success = world.completed(), world.evaluate()
                require(type(completed) is bool and type(success) is bool, 'unknown prefix state')
                stats.update(prefix_completed=completed, prefix_success=success, status='complete')
    except Exception as exc:
        stats.update(status='incomplete', error_type=type(exc).__name__)
    except BaseException as exc:
        stats.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        try:
            if world is not None:
                world.close()
                stats['cleanup_verified'] = world.closed
        except BaseException as exc:
            stats.update(status='incomplete', cleanup_error_type=type(exc).__name__)
            raise
        finally:
            stats['wall_seconds'] = time.monotonic() - started
            atomic_json(output / 'summary.json', stats)
    return stats


def audit_record(directory, source, proposal, projection, *, environment_id):
    directory = Path(directory)
    manifest, summary = read(directory / 'manifest.json'), read(directory / 'summary.json')
    expected = identity(source, proposal, projection, environment_id)
    require(all(manifest.get(k) == v for k, v in expected.items()), 'prefix manifest differs')
    require(projection == project_failure(source, set(manifest['origin_scenarios']), max_chars=32768),
            'audited prefix source differs')
    RolloutDeadline(manifest['seconds'])
    require(all(x['LLM_calls'] == x['new_actions'] == 0 for x in (manifest, summary)), 'new execution is forbidden')
    require(summary['status'] in {'complete', 'incomplete', 'interrupted'}, 'nonterminal prefix')
    rows = [json.loads(s) for s in (directory / 'steps.jsonl').read_text().splitlines()]
    require(len(rows) <= expected['prefix_length'], 'too many replay observations')
    matched = 0
    for index, row in enumerate(rows):
        step = source.steps[index]
        require(row['index'] == index and row['action'] == step.action
            and type(row['observation_matches']) is bool
            and row['observation_matches'] == (row['observation'] == step.observation), 'raw replay record differs')
        require(matched == index, 'continued after a replay mismatch')
        matched += int(row['observation_matches'])
        finite_time(row['execute_seconds'])
    keys = ('attempted_steps', 'returned_observations', 'verified_observations')
    require(all(type(summary[k]) is int and 0 <= summary[k] <= expected['prefix_length'] for k in keys)
        and summary['returned_observations'] == len(rows) and summary['verified_observations'] == matched
        and len(rows) <= summary['attempted_steps'] <= len(rows) + 1, 'replay accounting differs')
    complete = summary['status'] == 'complete'
    if complete:
        require(matched == len(rows) == summary['attempted_steps'] == expected['prefix_length']
            and type(summary['prefix_success']) is bool and type(summary['prefix_completed']) is bool
            and summary['cleanup_verified'] is True, 'complete prefix has unknown or mismatched state')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], **expected,
        'prefix_success': summary['prefix_success'], 'prefix_completed': summary['prefix_completed'],
        'cleanup_verified': summary['cleanup_verified'], 'verified_observations': matched,
        'returned_observations': len(rows), 'attempted_steps': summary['attempted_steps'],
        'unknown_execution_result': summary['attempted_steps'] > len(rows),
        'eligible_for_continuation': complete and not summary['prefix_success'] and not summary['prefix_completed'],
        'execute_seconds': sum(r['execute_seconds'] for r in rows), 'wall_seconds': summary['wall_seconds'],
        'LLM_calls': 0, 'new_actions': 0, 'input_sha256': {n: digest(directory / n) for n in
            ('manifest.json', 'summary.json', 'steps.jsonl')},
        'scope': 'Exact replay of the proposed prefix only. No new recovery, proposal semantics or utility proven; '
                 'container absence must be independently checked by the caller.'}
