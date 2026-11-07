"""Fourteen prospective utility leases over isolated, frozen GPU admission code.

This gate reads public metadata only. The local client must authenticate its
private input/protocol and native runtime separately. Import does not register,
reserve, enqueue, contact a host, or initialize a model.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import recovery_evidence_utility as utility
from scripts.fresh_completion_metadata import OFFICIAL_TRAIN_SHA256
from scripts.queue_recovery_evidence_lesson import (
    _public_directory,
    public_read,
    verify_public_frozen_plan,
)
from scripts.queue_recovery_evidence_lesson import (
    public_wll_fingerprint as public_wll_fingerprint,
)
from scripts.verification_readonly_registry_v3 import SOURCE_SHA256 as RUNTIME_SOURCE_SHA256
from scripts.verification_token_counter import TOKENIZER_SHA256
from wll.protocol import fingerprint

SOURCE = Path(__file__).with_name('completion_gpu_queue.py')
SOURCE_SHA256 = '928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8'
HELPER_SHA256 = 'ef0d3665d38990f637a0786db7d8e3d097c1df577c93c842b1fbdbf061f9aed1'
ID = 'qwen38-coverage-contract-utility-001'
PLAN = 'artifacts/qwen38-coverage-contract-utility-plan-001.json'
PORT = 18990
WAIT_SECONDS, GPU_SECONDS, CLIENT_SECONDS = 86400, 7200, 6000
MODEL_START_SECONDS, CLEANUP_SECONDS = 900, 180
IMAGE = utility.FIXED['environment_image']
PRIVATE_RUN_ROOT = '/home/hetaisheng/.local/share/workflow-learning-lab/coverage-contract-utility-001'
DESIGN = utility.COVERAGE_CONTRACT_DESIGN
TASK_IDS, SCENARIO_IDS = list(utility.TASK_IDS), list(utility.UTILITY_SCENARIOS)
BATCH_SCHEDULE = [
    {'batch_index': offset + index, 'phase': phase, 'phase_batch_index': index,
     'branches': [list(branch) for branch in schedule]}
    for phase, offset in (('controls', 0), ('prompted', 5))
    for index, schedule in enumerate(utility.batch_schedules(phase))
]
BATCHES = [row['branches'] for row in BATCH_SCHEDULE]
PROTOCOL_SETTINGS = {**utility.FIXED, 'design': DESIGN,
    'thresholds': dict(utility.THRESHOLDS), 'comparison_view': copy.deepcopy(utility.COMPARISON_VIEW)}
CONTEXT_PROOF = 'artifacts/coverage-contract-utility-inputs-CPU-001.json'
PROTOCOL_PROOF = 'artifacts/coverage-contract-utility-protocol-CPU-001.json'
PARTITION = 'artifacts/partitions-v1/manifest.json'
DESCRIPTORS = 'artifacts/coverage-contract-descriptor-probe-CPU-001/report.json'
PREPARATION = 'artifacts/coverage-contract-preparation-CPU-002.json'
CONTENT_REVIEW = 'artifacts/coverage-contract-root-review-CPU-002.json'
TASK_DESCRIPTORS = 'artifacts/all-utility-variants-inputs-001/tasks.jsonl'
TASK_DESCRIPTORS_SHA256 = '87f7471ca178292a4ba66c32f07816d0ad44d65dfcc25adb8a8a01d15ae2ad10'
PUBLIC_PROOFS = {
    CONTEXT_PROOF: '2d0ebad5701dbeca73ab4716253aecafefbe85afe551dba8dd0ca09557e4456a',
    PROTOCOL_PROOF: 'f29fb7fe49e00bed5785a74221e077f375f5cda0abd3c9d1d301e73d1b8b97ec',
    PARTITION: '3732b02da71c4031ba3a831c727d1956bb43872552841db275a86900e0dcf4a6',
    DESCRIPTORS: '7347ffb9c7504c1614e309b8e84636ff127f5333f51764333681d3adb30a5f40',
    PREPARATION: '5dffaf1cf42e84f93077e392d65e0d7c07f611911a9b20d24fcbd901241aa678',
    CONTENT_REVIEW: '138cad6ffc8cd221e30fb58e4d58e848db885eb5431aa9b059050667f57ba40d',
}
_PRIVATE_NAMES = {'tasks.jsonl', 'inputs.jsonl', 'candidate.jsonl', 'protocol.json',
                  'actor.jsonl', 'episode.json', 'worker-trace.jsonl', 'request.jsonl'}
_ERROR = 'coverage_contract_queue_invalid'
LAUNCH_SOURCES = frozenset({
    'scripts/coverage_contract_gpu_queue.py', 'tests/test_coverage_contract_gpu_queue.py',
    'scripts/run_coverage_contract_utility_client.py', 'tests/test_coverage_contract_utility_client.py',
    'scripts/run_coverage_contract_utility_batch.py', 'tests/test_coverage_contract_utility_batch.py',
    'scripts/run_recovery_evidence_utility_branch.py', 'tests/test_recovery_evidence_utility_branch.py',
    'scripts/report_coverage_contract_utility_branch.py', 'tests/test_coverage_contract_utility_branch_audit.py',
    'scripts/coverage_contract_token_endpoint.py', 'tests/test_coverage_contract_token_endpoint.py',
    'scripts/recovery_evidence_utility.py', 'tests/test_recovery_evidence_utility.py',
    'scripts/coverage_contract.py', 'tests/test_coverage_contract.py',
})
LAUNCH_SUITES = frozenset(name[:-3].replace('/', '.') for name in LAUNCH_SOURCES if name.startswith('tests/'))


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_same(value[key], item) for key, item in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected, strict=True))
    return value == expected


def _isolated(*, raw=None):
    if raw is None:
        raw = public_read(SOURCE.parent.parent, 'scripts/completion_gpu_queue.py')
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256)
    source = raw.decode('utf-8')
    old = 'from scripts.serve_qwen38 import make_handler'
    _require(source.count(old) == 1)
    source = source.replace(old, 'from scripts.coverage_contract_token_endpoint import make_handler')
    spec = importlib.util.spec_from_file_location('scripts._isolated_coverage_contract_queue', SOURCE)
    _require(spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    # Only pinned repository source is compiled; actor Python never enters here.
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), module.__dict__)
    return module


_CORE = _isolated()
ROOT, REMOTE, PYTHON = _CORE.ROOT, _CORE.REMOTE, _CORE.PYTHON
SERVICE_PLAN, SERVICE_SHA = _CORE.SERVICE_PLAN, _CORE.SERVICE_SHA
GATE, GATE_SHA, GATE_B3, GATE_B3_SHA = _CORE.GATE, _CORE.GATE_SHA, _CORE.GATE_B3, _CORE.GATE_B3_SHA
MODEL_IDS = dict(_CORE.MODEL_IDS)
REQUIRED_SOURCES = frozenset(_CORE.REQUIRED_SOURCES) | {
    'scripts/coverage_contract_gpu_queue.py', 'tests/test_coverage_contract_gpu_queue.py',
    'scripts/coverage_contract_token_endpoint.py', 'tests/test_coverage_contract_token_endpoint.py',
    'scripts/coverage_contract.py', 'tests/test_coverage_contract.py',
    'scripts/recovery_evidence_utility.py', 'tests/test_recovery_evidence_utility.py',
    'scripts/run_recovery_evidence_utility_branch.py', 'tests/test_recovery_evidence_utility_branch.py',
    'scripts/run_coverage_contract_utility_batch.py', 'tests/test_coverage_contract_utility_batch.py',
    'scripts/report_coverage_contract_utility_branch.py', 'tests/test_coverage_contract_utility_branch_audit.py',
    'scripts/completion_gpu_queue.py', 'scripts/queue_recovery_evidence_lesson.py',
    'scripts/queue_completion_lesson.py', 'scripts/serve_qwen38.py', 'scripts/serve_qwen38_b3.py',
    'scripts/fresh_completion_world.py', 'scripts/completion_trace.py', 'scripts/owned_appworld_v3.py',
    'scripts/run_timed_tasks.py', 'scripts/run_observed_utility_branch.py',
    'scripts/exact_cluster_resampling.py', 'scripts/utility_feasibility.py',
    'scripts/batch_cpu_token_counter.py', 'scripts/verification_token_counter.py',
    'artifacts/prepare_coverage_contract_001.py', 'tests/test_prepare_coverage_contract.py',
    'artifacts/prepare_coverage_contract_utility_inputs_001.py', 'tests/test_prepare_coverage_contract_utility_inputs.py',
    'artifacts/probe_coverage_contract_descriptors_001.py',
    'wll/runner.py', 'wll/prompts.py', 'wll/protocol.py', 'wll/experience.py', 'wll/model.py',
    'scripts/fresh_completion_metadata.py', 'scripts/verification_readonly_registry_v3.py',
} | LAUNCH_SOURCES


def make_job(index, owner, plan_sha):
    _require(type(index) is int and 0 <= index < 14 and type(owner) is str
        and re.fullmatch('[a-f0-9]{32}', owner) is not None and _sha(plan_sha))
    return {'id': f'{ID}-comparison-{index}', 'kind': 'comparison', 'batch_index': index,
        'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': GPU_SECONDS,
        'argv': [PYTHON, '-B', '-m', 'scripts.coverage_contract_gpu_queue', '--serve',
                 '--plan-sha256', plan_sha, '--batch', str(index), '--owner', owner]}


def _public_maps(plan):
    sources, inputs = plan.get('source_sha256'), plan.get('input_sha256')
    _require(type(sources) is dict and REQUIRED_SOURCES <= sources.keys() and type(inputs) is dict)
    merged = {}
    for values in (sources, inputs):
        for name, sha in values.items():
            _require(type(name) is str and bool(name) and '\0' not in name
                and not Path(name).is_absolute() and '..' not in Path(name).parts
                and os.path.normpath(name) == name and Path(name).name not in _PRIVATE_NAMES
                and _sha(sha) and (name not in merged or merged[name] == sha))
            merged[name] = sha
    _require(sources['scripts/completion_gpu_queue.py'] == SOURCE_SHA256
        and sources['scripts/queue_recovery_evidence_lesson.py'] == HELPER_SHA256)
    return sources, inputs


def _read(root, name):
    return json.loads(public_read(root, name))


def _bound_wll(root, sources):
    directory = _public_directory(Path(root) / 'wll')
    try:
        _require(all('wll/' + name in sources for name in os.listdir(directory) if name.endswith('.py')))
    finally:
        os.close(directory)


def _public_service(root, plan, sources, inputs):
    for name, sha in ((SERVICE_PLAN, SERVICE_SHA), (GATE, GATE_SHA), (GATE_B3, GATE_B3_SHA)):
        _require(inputs.get(name) == sha)
    service, gate, measured = (_read(root, name) for name in (SERVICE_PLAN, GATE, GATE_B3))
    _require(all(sources.get(name) == sha for name, sha in service['source_sha256'].items()))
    policy = _read(root, 'config/gpu-policy.json')
    _require(_same(plan.get('gpu_policy'), policy)
        and plan.get('gpu_policy_sha256') == utility.GPU_POLICY_SHA256
        == service['input_sha256']['config/gpu-policy.json']
        == sources.get('config/gpu-policy.json')
        and policy.get('idle_seconds') == 180 and policy.get('poll_seconds') == 15
        and policy.get('max_poll_gap_seconds') == 35 and policy.get('max_used_mib') == 128
        and policy.get('max_utilization_percent') == 1 and policy.get('minimum_free_mib') == 22528
        and policy.get('hostname') == 'amax' and policy.get('allowed_gpus') == utility.ALLOWED_GPUS
        and gate.get('phase') == 'complete' and gate.get('all_resources_released') is True
        and gate.get('plan_sha256') == SERVICE_SHA)
    expected = plan.get('expected_backend_static')
    _require(type(expected) is dict and set(expected) == {'B3'}
        and _same(expected['B3'], _CORE.static_backend(measured['final_health']['backend']))
        and fingerprint(expected['B3']) == utility.B3_STATIC_SHA256)


def _public_preparation(root, plan, sources, inputs):
    _require(all(inputs.get(name) == sha for name, sha in PUBLIC_PROOFS.items())
        and plan.get('context_proof') == CONTEXT_PROOF and plan.get('protocol_proof') == PROTOCOL_PROOF)
    context, proof, partition, descriptors = (_read(root, name)
        for name in (CONTEXT_PROOF, PROTOCOL_PROOF, PARTITION, DESCRIPTORS))
    _require(context.get('id') == 'coverage-contract-utility-inputs-CPU-001'
        and context.get('status') == 'prepared_initial_contexts_fit_not_registered'
        and proof.get('id') == 'coverage-contract-utility-protocol-CPU-001'
        and proof.get('status') == 'actual_inputs_and_protocol_prepared_not_registered'
        and context.get('design') == proof.get('design') == DESIGN
        and _same(context.get('task_ids'), TASK_IDS)
        and partition['task_ids']['utility'] == TASK_IDS
        and len({name.rsplit('_', 1)[0] for name in TASK_IDS}) == 9
        and all(context.get(key) is False and proof.get(key) is False
                for key in ('registered', 'training_authorized', 'utility_outcomes_read'))
        and context.get('source_scenario_overlap') == proof.get('source_scenario_overlap') == []
        and context.get('all81_initial_contexts_fit') is proof.get('all81_initial_contexts_fit') is True)
    for value in (context, proof):
        _require(type(value.get('task_count')) is int and value['task_count'] == 27
            and type(value.get('scenario_count')) is int and value['scenario_count'] == 9
            and _same(value.get('lesson_message_tokens'), {'candidate': 199, 'matched_unrelated_placebo': 199})
            and all(type(value.get(key)) is int and value[key] == 0
                    for key in ('model_calls', 'GPU_jobs', 'accepted_experiences')))
    _require(all(sources.get(name) == sha for name, sha in context['source_sha256'].items())
        and context['input_sha256'].get(TASK_DESCRIPTORS) == TASK_DESCRIPTORS_SHA256
        and all(inputs.get(name) == sha for name, sha in context['input_sha256'].items()
                if name != TASK_DESCRIPTORS)
        and all(inputs.get(name) == sha for name, sha in proof['public_evidence_sha256'].items())
        and proof.get('candidate_creator') == 'assistant_authored_structural_extractor'
        and proof.get('placebo_creator') == 'assistant_authored_unrelated_placebo'
        and proof.get('current_image_descriptors_match') is True
        and proof.get('utility_verified') is False)
    for name, expected in (('private_input_sha256', context['private_input_sha256']),
        ('private_input_bytes', context['private_input_bytes']),
        ('private_protocol_sha256', proof['private_protocol_sha256']),
        ('private_protocol_bytes', proof['private_protocol_bytes']),
        ('protocol_fingerprint', proof['protocol_fingerprint']),
        ('private_input_path', context['private_input_path']),
        ('private_protocol_path', proof['private_protocol_path'])):
        _require(type(plan.get(name)) is type(expected) and plan[name] == expected)
    _require(_sha(plan['private_input_sha256']) and _sha(plan['private_protocol_sha256'])
        and _sha(plan['protocol_fingerprint']) and plan['private_input_bytes'] > 0
        and plan['private_protocol_bytes'] > 0 and proof['private_input_sha256'] == context['private_input_sha256'])
    initial = context.get('initial_input_tokens')
    utility._contexts(initial, context['lesson_message_tokens'])
    tokenized = context.get('CPU_tokenization_proof')
    expected_keys = {name + ':' + arm for name in TASK_IDS for arm in utility.ARMS} | {'candidate', 'placebo'}
    _require(type(tokenized) is dict and tokenized.get('status') == 'measured_CPU_batch_tokenization'
        and tokenized.get('tokenizer_files_sha256') == tokenized.get('tokenizer_files_sha256_after') == TOKENIZER_SHA256
        and tokenized.get('cuda_initialized') is tokenized.get('model_weights_loaded') is False
        and tokenized.get('CUDA_VISIBLE_DEVICES') == '' and type(tokenized.get('results')) is dict
        and set(tokenized['results']) == expected_keys)
    for key, row in tokenized['results'].items():
        _require(type(row) is dict and set(row) == {'kind', 'input_tokens', 'content_sha256'}
            and _sha(row['content_sha256']) and type(row['input_tokens']) is int)
        if key in {'candidate', 'placebo'}:
            _require(row['kind'] == 'text' and row['input_tokens'] == 199)
        else:
            name, arm = key.split(':')
            _require(row['kind'] == 'messages' and row['input_tokens'] == initial[name][arm] <= 15360)
    _require(descriptors.get('id') == 'coverage-contract-descriptor-probe-CPU-001'
        and descriptors.get('status') == 'descriptors_match' and descriptors.get('valid') is True
        and descriptors.get('creator_absent') is descriptors.get('owned_container_absent') is True
        and descriptors.get('task_ids') == TASK_IDS and descriptors.get('task_descriptor_sha256') == context['task_sha256']
        and descriptors.get('environment_id') == proof['environment_id']
        and descriptors.get('image') == plan.get('image') == IMAGE
        and all(plan.get(key) == descriptors.get(key) for key in ('data_dir', 'deps_dir', 'environment_id')))


def validate_native_metadata(plan):
    """Check the declared allowlist only; the local client verifies actual bytes."""
    bindings = plan.get('native_bindings')
    _require(type(bindings) is dict)
    sources, inputs = plan['source_sha256'], plan['input_sha256']
    _require(all(bindings.get(name) == sha for values in (sources, inputs) for name, sha in values.items()))
    data, deps = Path(plan['data_dir']), Path(plan['deps_dir'])
    private = {plan['private_input_path']: plan['private_input_sha256'],
               plan['private_protocol_path']: plan['private_protocol_sha256']}
    _require(len(private) == 2 and all(bindings.get(name) == sha for name, sha in private.items()))
    appworld, docs = deps / 'appworld', data / 'api_docs'
    train, version = data / 'datasets/train.txt', data / 'version.txt'
    _require(bindings.get(str(train)) == OFFICIAL_TRAIN_SHA256)
    for relative, sha in RUNTIME_SOURCE_SHA256.items():
        _require(bindings.get(os.path.normpath(str(appworld / relative))) == sha)
    found_docs = False
    forbidden = {'tasks', 'dbs', 'ground_truth', 'groundtruth', 'solutions', 'task_solutions'}
    for name, sha in bindings.items():
        _require(type(name) is str and bool(name) and '\0' not in name
            and os.path.normpath(name) == name and '..' not in Path(name).parts and _sha(sha))
        path = Path(name)
        if not path.is_absolute():
            _require(name in sources or name in inputs)
        elif name in private:
            continue
        else:
            is_source = (path.is_relative_to(appworld) and path.suffix == '.py'
                         and not set(path.relative_to(appworld).parts) & forbidden)
            is_sibling = path == deps / 'fastapi_login/fastapi_login.py'
            is_docs = path.is_relative_to(docs) and path.suffix == '.json'
            _require(is_source or is_sibling or is_docs or path in {train, version})
            found_docs |= is_docs
    _require(found_docs)


def _public_launch(root, plan, sources, inputs):
    proof_name, junit_name = plan.get('launch_root_proof'), plan.get('launch_junit')
    _require(type(proof_name) is str and proof_name.startswith('artifacts/') and proof_name.endswith('.json')
        and type(junit_name) is str and junit_name.startswith('artifacts/') and junit_name.endswith('.xml')
        and proof_name in inputs and junit_name in inputs)
    proof, raw = _read(root, proof_name), public_read(root, junit_name)
    _require(type(proof) is dict and proof.get('status') == 'passed_CPU_only'
        and proof.get('junit_sha256') == hashlib.sha256(raw).hexdigest() == inputs[junit_name]
        and type(proof.get('source_sha256')) is dict
        and LAUNCH_SOURCES <= proof['source_sha256'].keys()
        and all(sources.get(name) == sha for name, sha in proof['source_sha256'].items()))
    tree = ET.fromstring(raw)
    _require(tree.tag in {'testsuite', 'testsuites'}
        and not list(tree.iter('failure')) and not list(tree.iter('error')) and not list(tree.iter('skipped')))
    cases, classes = list(tree.iter('testcase')), set()
    for case in cases:
        classes.add(case.get('classname'))
    _require(len({(case.get('classname'), case.get('name')) for case in cases}) == len(cases))
    for node in tree.iter():
        if node.tag in {'testsuite', 'testsuites'}:
            for key in ('failures', 'errors', 'skipped'):
                if key in node.attrib:
                    _require(re.fullmatch('[0-9]+', node.get(key)) is not None and int(node.get(key)) == 0)
    suites = [suite for suite in tree.iter('testsuite') if not list(suite.iter('testsuite'))[1:]]
    counts = 0
    for suite in suites:
        for key in ('tests', 'failures', 'errors', 'skipped'):
            value = suite.get(key)
            _require(type(value) is str and re.fullmatch('[0-9]+', value) is not None)
            _require(key == 'tests' or int(value) == 0)
        counts += int(suite.get('tests'))
    _require(bool(cases) and len(cases) == counts and LAUNCH_SUITES <= classes)
    if 'tests' in proof:
        _require(type(proof['tests']) is int and proof['tests'] == counts)


def load_registration(root, sha):
    """Validate bound public bytes and metadata; never open private/native inputs."""
    try:
        _require(_sha(sha))
        raw = public_read(root, PLAN)
        _require(hashlib.sha256(raw).hexdigest() == sha)
        plan = json.loads(raw)
        _require(type(plan) is dict)
        sources, inputs = _public_maps(plan)
        _bound_wll(root, sources)
        verify_public_frozen_plan(root, Path(root) / PLAN, sha, expected_plan=plan)
        _require(plan.get('id') == ID and plan.get('status') == 'registered'
            and plan.get('design') == DESIGN and plan.get('automatic_retry') is False
            and plan.get('training_authorized') is False
            and plan.get('private_run_root') == PRIVATE_RUN_ROOT
            and type(plan.get('accepted_experiences')) is int and plan['accepted_experiences'] == 0
            and _same(plan.get('task_ids'), TASK_IDS) and _same(plan.get('scenario_ids'), SCENARIO_IDS)
            and _same(plan.get('batch_schedule'), BATCH_SCHEDULE)
            and _same(plan.get('protocol_settings'), PROTOCOL_SETTINGS))
        for key, value in (('gpu_seconds', GPU_SECONDS), ('client_seconds', CLIENT_SECONDS),
            ('model_start_seconds', MODEL_START_SECONDS), ('cleanup_seconds', CLEANUP_SECONDS),
            ('admission_wait_seconds', WAIT_SECONDS), ('port', PORT)):
            _require(type(plan.get(key)) is int and plan[key] == value)
        _public_service(root, plan, sources, inputs)
        _public_preparation(root, plan, sources, inputs)
        validate_native_metadata(plan)
        _public_launch(root, plan, sources, inputs)
        return plan
    except (KeyError, TypeError, ValueError, OSError, UnicodeError, AttributeError, RecursionError, ET.ParseError):
        raise ValueError(_ERROR) from None


for _name, _value in {'ID': ID, 'PLAN': PLAN, 'PORT': PORT, 'WAIT_SECONDS': WAIT_SECONDS,
    'REQUIRED_SOURCES': REQUIRED_SOURCES, 'make_job': make_job, 'load_registration': load_registration}.items():
    setattr(_CORE, _name, _value)


def __getattr__(name):
    return getattr(_CORE, name)


if __name__ == '__main__':
    _CORE.main()
