"""Read-only handoff of an unchanged utility-selected rendering, never a raw candidate.

The frozen study is re-audited in full, including live resource release. This
does not authorize training or certify the excluded writer output as compliant.
"""
import json
from pathlib import Path

from scripts.audit_memory_comparison_v2 import combine_study
from scripts.run_qwen38_memory_comparison_v2 import (
    ARMS,
    ID,
    PLAN,
    digest,
    load_registration,
    read,
    require,
)
from wll.protocol import fingerprint


def snapshot_files(directory):
    directory = Path(directory).resolve(strict=True)
    result = {}
    for path in sorted(directory.rglob('*')):
        require(path.resolve().is_relative_to(directory), 'evidence path escapes project snapshot')
        if path.is_file():
            result[str(path.relative_to(directory))] = digest(path)
    return result


def project_file(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'non-project projection path')
    result = (root / path).resolve(strict=True)
    require(result.is_relative_to(root), 'projection escapes project')
    return result


def load_registered_projection(root, plan_sha, arm):
    """Return (rendered_text, provenance); no model calls, GPU jobs or output writes."""
    require(arm in ARMS[1:], 'a candidate arm is required')
    root = Path(root).resolve(strict=True)
    plan, inputs, tasks, candidates = load_registration(root, root / PLAN, plan_sha)
    base = root / 'artifacts' / ID
    status_raw = (base / 'status.json').read_bytes()
    status = json.loads(status_raw)
    require(status.get('phase') == 'complete' and status.get('completed_batches') == len(plan['batches'])
        and status.get('plan_sha256') == plan_sha, 'complete registered memory study required')
    report_raw = (base / 'report.json').read_bytes()
    saved = json.loads(report_raw)
    require(saved == status.get('result') and saved.get('status') == 'complete'
        and saved.get('observed_branches') == len(tasks) * len(ARMS)
        and saved.get('unknown_branches') == saved.get('unattempted_branches') == 0,
        'complete known three-arm results required')
    require(saved.get('selection_rule_activated') is True and saved.get('all_resources_released') is True
        and type(saved.get('accepted_experiences')) is int and saved['accepted_experiences'] == 0
        and saved.get('automatic_training') is False and saved.get('raw_candidate_content_certified') is False
        and saved.get('full_hint_writer_instruction_violation_retained') is True,
        'selection, release and raw-content boundaries differ')
    value = inputs['candidates'][arm]
    render_path = project_file(root, value['rendered_file'])
    review_path = project_file(root, inputs['content_review'])
    require(digest(render_path) == value['rendered_sha256']
        and digest(review_path) == inputs['content_review_sha256'], 'projection or content review changed')
    rendered = render_path.read_text()
    require(rendered == candidates[arm].render(), 'existing render projection changed')
    before = snapshot_files(base)
    fresh = combine_study(root, plan, inputs, tasks, candidates, base)
    require(saved == fresh, 'saved memory report cannot be reproduced')
    require(arm in fresh['utility_selected_projections'] and fresh['candidate_gate_passes'][arm] is True,
        'utility projection rejected')
    selected = fresh['utility_selected_projections'][arm]
    require(selected == {'experience_id': candidates[arm].experience_id,
        'rendered_sha256': value['rendered_sha256'],
        'raw_candidate_fingerprint': fingerprint(value['candidate'])}, 'selected projection identity differs')
    review = read(review_path)
    record, = [r for r in review['conditions'] if r['condition'] == arm.removesuffix('_memory')]
    services = [{'model_id': read(base / f'batch-{i}/health.json')['model'],
        'backend': read(base / f'batch-{i}/health.json')['backend']} for i in range(len(plan['batches']))]
    require(load_registration(root, root / PLAN, plan_sha) == (plan, inputs, tasks, candidates),
        'registered inputs changed during audit')
    require(snapshot_files(base) == before and (base / 'status.json').read_bytes() == status_raw
        and (base / 'report.json').read_bytes() == report_raw
        and digest(render_path) == value['rendered_sha256']
        and digest(review_path) == inputs['content_review_sha256'], 'terminal evidence changed during audit')
    return rendered, {'study_id': ID, 'arm': arm, 'experience_id': candidates[arm].experience_id,
        'plan_sha256': plan_sha, 'report_hash': fingerprint(saved), 'report_file_sha256': digest(base / 'report.json'),
        'loader_sha256': digest(__file__), 'rendered_file_sha256': value['rendered_sha256'],
        'raw_candidate_fingerprint': selected['raw_candidate_fingerprint'],
        'content_review_file_sha256': inputs['content_review_sha256'],
        'full_output_writer_instruction_violation': record['full_output_writer_instruction_conformance'] is False,
        'raw_candidate_content_certified': False, 'training_authorized': False,
        'validation_scenarios': sorted({t.scenario_id for t in tasks}),
        'services': services, 'shared_control_branches': fresh['shared_control_branches'],
        'proposal_specific_utility_evidence': fresh['proposal_specific_utility_evidence'],
        'study_file_sha256': before, 'distillation_requires_separate_registration': True,
        'scope': 'Private unchanged render projection selected on reused train utility. '
            'Full writer content remains uncertified; no held-out, parameter or recursive-improvement claim. '
            'Subsequent data generation and training must register the projection, loader, scenarios and matched controls.'}
