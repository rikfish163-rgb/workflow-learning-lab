"""CodeAct paired utility over externally authenticated metadata only.

The caller must bind the fixed cohort, source origins, episode/scorer evidence,
model identity, budgets and releases before constructing summaries. This helper
checks their shape, preserves the observed worker exit, and computes no subset
estimate. It does not authenticate inputs, run tasks, or establish unseen transfer.
"""

import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

BASE_SOURCE = Path(__file__).with_name('native_experience_utility_001.py')
BASE_SHA256 = '4e94c34fd2068b93dde7ec2243698ca20096b7abe9bd824fca2b765b5c8ee461'
DEPENDENCY_SHA256 = {
    BASE_SOURCE.with_name('exact_cluster_resampling.py'):
        '2b78767d9c0e962e53708c2cab0dbc5cdf2d3a0d9d4012bd34c2d5e6430eca5a',
    BASE_SOURCE.parent.parent / 'wll/protocol.py':
        '81cddbc5ed9f43412b0b98ff21098048ca479a0a177b1ee6605611622953e0f8',
}
_ERROR = 'codeact_within_utility_invalid'
_CORE = None


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _isolated(*, raw=None):
    raw = BASE_SOURCE.read_bytes() if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == BASE_SHA256)
    _require(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                 for path, digest in DEPENDENCY_SHA256.items()))
    source = raw.decode('utf-8')
    replacements = (
        ('                         "experience_bundle_id"})\n_STATUSES =',
         '                         "experience_bundle_id", "owner_cleanup_verified", '
         '"owner_release_verified"})\n_STATUSES ='),
        ('                 and type(row["is_policy_episode"]) is bool\n',
         '                 and type(row["is_policy_episode"]) is bool\n'
         '                 and type(row["owner_cleanup_verified"]) is bool\n'
         '                 and type(row["owner_release_verified"]) is bool\n'),
        ('            or not row["is_policy_episode"] or row["worker_exit_code"] != 0\n',
         '            or not row["is_policy_episode"] or row["worker_exit_code"] not in (0, -9)\n'
         '            or not row["owner_cleanup_verified"] or not row["owner_release_verified"]\n'),
    )
    for old, new in replacements:
        _require(source.count(old) == 1)
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_utility_core', BASE_SOURCE)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    # Only pinned human-authored repository source, never actor or model code.
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def evaluate_utility(expected_pairs, summaries, *, identity, source_scenario_ids, experience_bundle_id):
    """Estimate all 27 train pairs or return a non-evaluable full denominator.

    Rows add exact booleans ``owner_cleanup_verified`` and
    ``owner_release_verified`` to the native summary shape. A known row requires
    both plus the original closed/scored/known checks and actual exit 0 or -9.
    These supplied booleans are structural claims, not authenticated receipts.
    Input rows are not rewritten; the caller retains the observed exit values.
    """
    try:
        core = _get_core()
        pairs = core._manifest(expected_pairs, source_scenario_ids)
        _require(len(pairs) == 27 and len({task for task, _, _ in pairs}) == 27
                 and len(source_scenario_ids) == 12)
        scenarios = Counter(scenario for _, scenario, _ in pairs)
        _require(len(scenarios) == 9 and all(count == 3 for count in scenarios.values()))
        result = core.evaluate_utility(expected_pairs, summaries, identity=identity,
            source_scenario_ids=source_scenario_ids, experience_bundle_id=experience_bundle_id)
        old = 'Train scenarios may have prior exposure; no unseen generalization is established.'
        new = ('This intended train cohort is treated as exposed; this helper does not authenticate '
               'exposure and no unseen generalization is established.')
        _require(result['limits'].count(old) == 1)
        result['limits'] = result['limits'].replace(old, new)
        return result
    except Exception:
        raise ValueError(_ERROR) from None
