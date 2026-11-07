"""Post-hoc origin exploration; never overwrite prospective compact outcomes."""
import copy
import json

from scripts.run_compact_critic import classify_response
from scripts.semantic_repair_proposal import require, validate_proposal
from wll.protocol import fingerprint


def explore_legacy_format(raw, projection):
    strict, _ = classify_response(raw, projection)
    record = {'strict_compact_status': strict, 'raw_response_sha256': fingerprint(raw),
        'projection_sha256': fingerprint(projection), 'legacy_schema_compatible': False,
        'posthoc_exploratory_only': True, 'accepted_experience': False, 'automatic_execution': False,
        'canonical_proposal': None, 'field_rewrites': [],
        'scope': 'Retrospective compatibility with pre-existing v1 text limits. Exact quotes, task text and '
                 'proposed decisions are unchanged; only uniquely resolved fragment addresses are added. '
                 'Not a prospective format success, semantic proof, recovery, or accepted experience.'}
    if strict in {'invalid_json', 'invalid_response_format', 'abstain'}:
        record['ineligible_reason'] = strict
        return record
    try:
        # Strict duplicate-key/nonfinite parsing already ran in classify_response.
        value = json.loads(raw['text'])
        require(isinstance(value, dict) and value.get('status') == 'repair', 'not a repair')
        require(len(json.dumps(value, ensure_ascii=False)) <= 2000, 'total compact budget exceeded')
        evidence = value.get('evidence')
        require(isinstance(evidence, list) and 1 <= len(evidence) <= 2, 'one or two citations required')
        canonical = copy.deepcopy(value)
        for row in canonical['evidence']:
            require(isinstance(row, dict) and set(row) == {'step_index', 'field', 'quote'}, 'citation schema differs')
            i, field, quote = row['step_index'], row['field'], row['quote']
            require(type(i) is int and 0 <= i < projection['step_count'] and field in ('action', 'observation')
                and isinstance(quote, str) and 10 <= len(quote) <= 100, 'invalid citation address or quote length')
            parts = [p for p in ('head', 'tail') if quote in projection['steps'][i][field][p]]
            require(len(parts) == 1, 'citation has no unique visible fragment')
            row['part'] = parts[0]
        checked = validate_proposal(canonical, projection)
    except ValueError as exc:
        record['ineligible_reason'] = str(exc)
    else:
        record.update(legacy_schema_compatible=True, canonical_proposal=checked,
            field_rewrites=['Added exact, uniquely resolved evidence[].part addresses; no original field modified.'])
    return record
