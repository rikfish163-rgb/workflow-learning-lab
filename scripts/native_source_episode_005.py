"""Source-005 episode with isolated 004 machinery and registered phase budgets.

The authenticated historical episode supplies all worker, queue, replay and
failure handling. Only its fixed design, singleton job binding and per-phase
worker timing are changed in a fresh module; earlier module globals stay intact.
"""

import argparse
import ast
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_source_episode_004.py'
BASE_SHA256 = 'e86bbd2b0729ebd706e685ac8d41b8bcd75a8d088c544f561d3fd783da2b1822'
LEGACY_SOURCE = ROOT / 'scripts/native_source_episode.py'
LEGACY_SHA256 = '6b034e2b0c53050832c43384b1e3c4bc5853f5c8e940bbe52c348a0d3ce36e1b'
ID = 'qwen38-native-source-pilot-005'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-005.json'
_CORE = None
_ERROR = 'native_source_episode_005_source_invalid'


def _verified(raw, expected):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(_ERROR)
    return raw.decode('utf-8')


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(_ERROR)
    return source.replace(before, after, 1)


def _functions(source):
    tree = ast.parse(source)
    names = {'_fixed_design', '_configuration', 'run_episode'}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            if node.name in found or node.decorator_list:
                raise ValueError(_ERROR)
            found[node.name] = ast.get_source_segment(source, node)
    if set(found) != names or any(type(value) is not str for value in found.values()):
        raise ValueError(_ERROR)
    return found


def _patch_core(core, source):
    functions = _functions(source)
    design = functions['_fixed_design']
    design = _replace_once(
        design, '    design = plan["design"]\n',
        '    _require(phase in {"first_attempt", "control", "review"})\n'
        '    design = plan["design"]\n')
    design = _replace_once(
        design, '"worker_wall_seconds": 780, "parent_unit_seconds": 900}.items())',
        '"worker_wall_seconds": 780, "parent_unit_seconds": 900,\n'
        '                 "suffix_actor_generation_seconds": 1200, "suffix_worker_wall_seconds": 1380,\n'
        '                 "suffix_parent_unit_seconds": 1500, "model_ready_seconds": 600}.items())')
    design = _replace_once(
        design, 'max_generations=branch["max_generations"], max_actions=60, max_api_requests=60,',
        'max_generations=branch["max_generations"],\n'
        '        max_actions=branch["max_action_attempts" if phase == "first_attempt"\n'
        '                           else "max_action_attempts_including_replay"],\n'
        '        max_api_requests=branch["max_API_invocations" if phase == "first_attempt"\n'
        '                                else "max_API_invocations_including_replay"],')
    design = _replace_once(
        design, 'max_new_tokens=budgets["max_new_tokens"], wall_seconds=budgets["actor_generation_seconds"])',
        'max_new_tokens=budgets["max_new_tokens"],\n'
        '        wall_seconds=budgets["actor_generation_seconds" if phase == "first_attempt"\n'
        '                             else "suffix_actor_generation_seconds"])')
    configuration = _replace_once(functions['_configuration'],
                                  'queue.make_job(index // 4,', 'queue.make_job(index,')
    run = functions['run_episode']
    run = _replace_once(
        run, '            config, root, _load_plan or registration.load_registration)\n',
        '            config, root, _load_plan or registration.load_registration)\n'
        '        worker_wall_seconds = plan["design"]["budgets"][\n'
        '            "worker_wall_seconds" if config["phase"] == "first_attempt"\n'
        '            else "suffix_worker_wall_seconds"]\n')
    run = _replace_once(run, 'deadline=time.monotonic() + 780)',
                        'deadline=time.monotonic() + worker_wall_seconds)')
    run = _replace_once(
        run, '_same(kwargs, {"max_actions": 60, "max_api_requests": 60,\n'
        '                                        "rpc_timeout": 60, "wall_seconds": 780})',
        '_same(kwargs, {"max_actions": limits.max_actions, "max_api_requests": limits.max_api_requests,\n'
        '                                        "rpc_timeout": 60, "wall_seconds": worker_wall_seconds})')
    run = _replace_once(run, 'rpc_timeout=60, worker_wall_seconds=780,',
                        'rpc_timeout=60, worker_wall_seconds=worker_wall_seconds,')
    core._SUFFIX_BUDGET = {
        'max_generations': 24, 'max_action_attempts_including_replay': 84,
        'max_API_invocations_including_replay': 84, 'max_prompt_tokens': 368640,
        'max_completion_tokens': 24576,
    }
    for source_function in (design, configuration, run):
        exec(compile(source_function, str(LEGACY_SOURCE), 'exec', dont_inherit=True), core.__dict__)


def _isolated(*, raw=None, raw_legacy=None, _modules=None):
    from scripts.native_source_registration import public_read

    base_bytes = public_read(ROOT, 'scripts/native_source_episode_004.py') if raw is None else raw
    legacy_bytes = public_read(ROOT, 'scripts/native_source_episode.py') if raw_legacy is None else raw_legacy
    base_source = _verified(base_bytes, BASE_SHA256)
    legacy_source = _verified(legacy_bytes, LEGACY_SHA256)
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_005'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_005'),
        'actor': importlib.import_module('scripts.native_appworld_actor_004'),
        'controller': importlib.import_module('scripts.native_appworld_controller_004'),
        'model': importlib.import_module('scripts.native_appworld_model_client_004'),
    }
    if (type(modules) is not dict or set(modules) != {'registration', 'queue', 'actor', 'controller', 'model'}
            or any(modules[name].ID != ID or modules[name].PLAN != PLAN for name in ('registration', 'queue'))
            or not callable(modules['controller'].run_task) or not callable(modules['model'].NativeModel)):
        raise ValueError('native_source_episode_005_binding_invalid')
    spec = importlib.util.spec_from_file_location('scripts._native_source_episode_005_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    base = importlib.util.module_from_spec(spec)
    exec(compile(base_source, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    base.ID, base.PLAN = ID, PLAN
    core = base._isolated(_modules=modules)
    _patch_core(core, legacy_source)
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_episode(config_path, expected_config_sha256, *, root=None, _load_plan=None,
                _model_factory=None, _run_task=None, _queue_status=None):
    core = _get_core()
    return core.run_episode(config_path, expected_config_sha256,
        root=core.WORKSPACE if root is None else root, _load_plan=_load_plan,
        _model_factory=_model_factory, _run_task=_run_task, _queue_status=_queue_status)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.parse_args(argv)
    try:
        return _get_core().main(argv)
    except Exception:
        sys.stderr.write('native_source_episode_failed\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
