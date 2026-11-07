"""Pure paired analysis of normalized outcomes for the fixed official dev split.

Callers must independently audit that outcomes and protocol identities came from
the real, released runs. This module reads no task descriptions or ground truth
and does not aggregate execution costs.
"""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from wll.protocol import Episode, Task, fingerprint, paired_estimate

ROOT = Path(__file__).resolve().parent.parent
PREPARATION = ROOT / 'artifacts/qwen38-b3-dev-evaluation-preparation-001.json'
PREPARATION_SHA256 = 'e748590baba716ac3fcc3fd2eaa9d6fdfed915833c15891b9a045d8100ca33e4'
OFFICIAL_DEV_SHA256 = '9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001'
RECIPE_SHA256 = '10722a1528fb695eca8fa3bfc13316924aef0c3657d0d02ef223855766d75c2e'
EXPECTED_PROTOCOL = {
    'split': 'dev',
    'base_repo': 'Qwen/Qwen3.8-27B',
    'base_revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
    'quantization': 'NF4',
    'service_recipe_sha256': RECIPE_SHA256,
    'environment_image_id': 'sha256:2f5a87274ea53b165a7f1ec656d2f6534aa0fbdd6479f4a77efbd89e39ef7cb2',
    'environment_seed_protocol': 'wll-appworld-seed-v1',
    'observation_feedback_kind': 'silent_documentation_print_hint_v1',
    'observation_feedback_source_sha256': {
        'scripts/documentation_feedback.py': 'e7eb24a8c87a5304815514203c483cfd4c2fe5bd32b0f82e5cd07b9de032b5b2',
        'scripts/probe_observation_visibility.py': 'd9bd7d258eabdc006a9aa470eff2cc0d06bf14e8437c4b1c74d3e6287129e301',
    },
    'prompt_version': 'appworld-v3',
    'generation_seed': 17,
    'generation_seed_rule': '17 + zero-based step index, reset for every task and arm',
    'environment_seed': 100,
    'max_steps': 40,
    'max_new_tokens': 1024,
    'max_context_tokens': 16384,
    'context_check': 'prompt tokens + requested max-new tokens <= 16384; no truncation',
    'rollout_seconds': 600,
    'temperature': 0.7,
    'experience_prompt': False,
    'fresh_environment_per_task_arm': True,
    'automatic_retry': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_official_ids(*, preparation_path=PREPARATION):
    """Read only the SHA-bound metadata ID list and its analysis proposal."""
    path = Path(preparation_path)
    require(not path.is_symlink() and path.is_file(), 'preparation must be a regular file')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PREPARATION_SHA256, 'preparation hash changed')
    prep = json.loads(raw)
    dataset, proposal = prep['dataset'], prep['proposal']
    ids = dataset['dev_ids_official_order']
    require(type(ids) is list and len(ids) == 57
            and all(type(value) is str and value for value in ids)
            and len(set(ids)) == 57, 'invalid official dev IDs')
    counts = Counter(value.rsplit('_', 1)[0] for value in ids)
    require(len(counts) == 19 and set(counts.values()) == {3}, 'invalid dev scenario clusters')
    require(dataset['metadata_file_sha256']['official_dev'] == OFFICIAL_DEV_SHA256,
            'official split hash changed')
    recipe = json.dumps(proposal['service_recipe'], sort_keys=True, ensure_ascii=False).encode()
    require(hashlib.sha256(recipe).hexdigest() == RECIPE_SHA256, 'service recipe changed')
    bound = {
        'base_repo': proposal['base_repo'], 'base_revision': proposal['base_revision'],
        'prompt_version': proposal['prompt_version'],
        'generation_seed': proposal['generation_seed_each_episode'],
        'generation_seed_rule': proposal['request_seed_rule'],
        'environment_seed': proposal['proposed_explicit_environment_seed_each_episode'],
        'max_steps': proposal['max_steps'], 'max_new_tokens': proposal['max_new_tokens_per_request'],
        'max_context_tokens': proposal['max_context_tokens'], 'context_check': proposal['context_check'],
        'rollout_seconds': proposal['rollout_seconds'], 'temperature': proposal['temperature'],
        'experience_prompt': proposal['experience_prompt'],
        'fresh_environment_per_task_arm': proposal['fresh_environment_per_task_arm'],
        'automatic_retry': proposal['automatic_retry'],
        'observation_feedback_kind': proposal['observation_feedback']['kind'],
        'observation_feedback_source_sha256': proposal['observation_feedback']['source_sha256'],
    }
    require(all(type(value) is type(EXPECTED_PROTOCOL[key]) and value == EXPECTED_PROTOCOL[key]
                for key, value in bound.items()), 'preparation protocol differs')
    return list(ids)


def _validate_protocol(protocol):
    require(type(protocol) is dict and set(protocol) == set(EXPECTED_PROTOCOL) | {'environment_id'},
            'complete common protocol required')
    require(all(type(protocol[key]) is type(expected) and protocol[key] == expected
                for key, expected in EXPECTED_PROTOCOL.items()), 'protocol differs')
    require(type(protocol['environment_id']) is str
            and re.fullmatch('[0-9a-f]{64}', protocol['environment_id']) is not None,
            'invalid environment identity')


def _validate_outcome(row):
    require(type(row) is dict and set(row) == {
        'task_id', 'status', 'success', 'generation_seed', 'environment_seed'},
        'invalid outcome record fields')
    require(type(row['task_id']) is str and row['task_id'], 'invalid task ID')
    require(type(row['generation_seed']) is int and row['generation_seed'] == 17,
            'generation seed differs')
    status, success, seed = row['status'], row['success'], row['environment_seed']
    require(type(status) is str and status in {'complete', 'unknown', 'unattempted'},
            'invalid outcome status')
    require((status == 'complete' and type(success) is bool)
            or (status != 'complete' and success is None), 'status and success disagree')
    require(type(seed) is dict and set(seed) == {'requested', 'effective', 'verified'}
            and type(seed['requested']) is int and seed['requested'] == 100
            and type(seed['verified']) is bool, 'invalid environment seed provenance')
    verified = seed['verified'] is True and type(seed['effective']) is int and seed['effective'] == 100
    unverified = seed['verified'] is False and seed['effective'] is None
    require(verified or unverified, 'environment seed mismatch')
    require(status != 'complete' or verified, 'completed outcome requires verified seed')
    require(status != 'unattempted' or unverified, 'unattempted task cannot have a verified world')


def _summarize_arm(rows):
    complete = [row for row in rows if row['status'] == 'complete']
    successes = sum(row['success'] is True for row in complete)
    unknown = sum(row['status'] == 'unknown' for row in rows)
    unattempted = sum(row['status'] == 'unattempted' for row in rows)
    return {'scheduled': len(rows), 'completed': len(complete), 'successes': successes,
            'failures': len(complete) - successes, 'unknown': unknown, 'unattempted': unattempted,
            'unknown_total': unknown + unattempted,
            'success_rate_completed_only': successes / len(complete) if complete else None,
            'success_bounds_all_scheduled': [successes / len(rows),
                                             (successes + unknown + unattempted) / len(rows)]}


def analyze(expected_ids, arms):
    """Validate explicit 57-row arm records and report the fixed paired contrast.

    Each arm has ``protocol`` and an ordered ``outcomes`` list. Each outcome has
    task_id, status (complete/unknown/unattempted), success, generation_seed,
    and requested/effective/verified environment_seed. The seed on an unattempted
    row describes the planned generation seed, not a claim that generation ran.
    """
    ids = load_official_ids()
    require(type(expected_ids) is list and expected_ids == ids, 'expected IDs differ from official order')
    require(type(arms) is dict and set(arms) == {'B0', 'B3'}, 'B0 and B3 arms required')
    for arm in ('B0', 'B3'):
        data = arms[arm]
        require(type(data) is dict and set(data) == {'protocol', 'outcomes'}, 'invalid arm fields')
        _validate_protocol(data['protocol'])
        rows = data['outcomes']
        require(type(rows) is list and len(rows) == len(ids), 'explicit outcome per task required')
        for row in rows:
            _validate_outcome(row)
        require([row['task_id'] for row in rows] == ids, 'outcome IDs or order differ')
    require(arms['B0']['protocol']['environment_id'] == arms['B3']['protocol']['environment_id'],
            'paired environment identities differ')

    control, treatment = arms['B0']['outcomes'], arms['B3']['outcomes']
    pairs, differences = [], []
    lower = upper = 0
    for a, b in zip(control, treatment, strict=True):
        a_value, b_value = a['success'], b['success']
        a_possible = (0, 1) if a_value is None else (int(a_value),)
        b_possible = (0, 1) if b_value is None else (int(b_value),)
        lower += min(b_possible) - max(a_possible)
        upper += max(b_possible) - min(a_possible)
        delta = int(b_value) - int(a_value) if a_value is not None and b_value is not None else None
        if delta is not None:
            differences.append(delta)
        pairs.append({'task_id': a['task_id'], 'B0_status': a['status'], 'B0_success': a_value,
                      'B3_status': b['status'], 'B3_success': b_value, 'delta': delta})

    primary = None
    if len(differences) == len(ids):
        episodes = {'B0': [], 'B3': []}
        for task_id, a, b in zip(ids, control, treatment, strict=True):
            task = Task(task_id, task_id.rsplit('_', 1)[0], 'dev', '')
            episodes['B0'].append(Episode(task, 'B0', 17, [], a['success'], 'normalized'))
            episodes['B3'].append(Episode(task, 'B3', 17, [], b['success'], 'normalized'))
        primary = paired_estimate(episodes['B0'], episodes['B3'], bootstrap_samples=2000, seed=0)
        primary['bootstrap_samples'] = 2000

    return {
        'scope': 'Official dev development pilot; not final test or evidence of method efficacy.',
        'input_limit': 'Normalized outcomes and protocol are caller-supplied; audit real runs, '
                       'backend/adapter/image provenance, release and all costs separately.',
        'preparation_sha256': PREPARATION_SHA256, 'official_dev_sha256': OFFICIAL_DEV_SHA256,
        'common_protocol': copy.deepcopy(arms['B0']['protocol']),
        'common_protocol_sha256': fingerprint(arms['B0']['protocol']),
        'complete': len(differences) == len(ids),
        'per_arm': {'B0': _summarize_arm(control), 'B3': _summarize_arm(treatment)},
        'missing_pairs': len(ids) - len(differences),
        'known_pairs': {'count': len(differences), 'helped': differences.count(1),
                        'harmed': differences.count(-1), 'ties': differences.count(0),
                        'descriptive_delta': sum(differences) / len(differences) if differences else None},
        'primary': primary,
        'full_schedule_worst_case_delta_bounds': [lower / len(ids), upper / len(ids)],
        'pair_outcomes': pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON with expected_ids and normalized arms')
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    require(type(data) is dict and set(data) == {'expected_ids', 'arms'}, 'invalid input fields')
    print(json.dumps(analyze(data['expected_ids'], data['arms']), indent=2))


if __name__ == '__main__':
    main()
