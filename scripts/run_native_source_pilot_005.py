"""Independent singleton-source schedule over authenticated 004 machinery.

Only the registered schedule, admission load limit and phase resource budgets
change. Original first-4xx selection, matched replay, fixed hint, source gates,
cancellation accounting, private initiating diagnostics and release fence stay
in force. No predecessor outcome is imported as a new example or resumed run.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/run_native_source_pilot_004.py'
BASE_SHA256 = '76e3ae6db7607a28744369e3eb3b8d8e4deab5850e98881964147d4e8d3cc6aa'
LEGACY_SOURCE = ROOT / 'scripts/run_native_source_pilot.py'
LEGACY_SHA256 = 'c9cac394a244e8764d3bbd7143d3b44e0db00e20cc5f6016d84504d89bbee7f6'
ID = 'qwen38-native-source-pilot-005'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-005.json'
PARENT_SECONDS = 12 * (86400 + 7200) + 600
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
_ERROR = 'native_source_pilot_005_source_invalid'
_CORE = None


def _require(value):
    if not value:
        raise ValueError(_ERROR) from None


def _episode_seconds(unit, arguments):
    if '--help' in arguments:
        return 900
    match = re.fullmatch(r'wll-native-source-[0-9]{2}-(first_attempt|control|review)-[a-f0-9]{32}\.service', unit)
    _require(match is not None)
    return 900 if match.group(1) == 'first_attempt' else 1500


def _v2_lifecycle(source):
    """Retain the exact authenticated V2 lifecycle edits when projecting V1 AST."""
    for before, after in (
        ('    except BaseException:\n        if not begun:',
         '    except BaseException as initiating_error:\n'
         '        _capture_initiating_exception(runtime, initiating_error)\n        if not begun:'),
        ('    except _BudgetStop as error:\n        report["status"] = "stopped_" + str(error)',
         '    except _BudgetStop as error:\n        _capture_initiating_exception(runtime, error)\n'
         '        report["status"] = "stopped_" + str(error)'),
        ('                    runtime.monitor()\n                    _queue_state(state, runtime)',
         '                    runtime.monitor()\n                    state = runtime.queue("status", job)\n'
         '                    _queue_state(state, runtime)'),
        ('                    runtime.record({"event": "GPU_released", "release": release})\n'
         '                if heartbeat_failure is not None:',
         '                    runtime.record({"event": "GPU_released", "release": release})\n'
         '                    if release.get("runtime_overrun") is True:\n'
         '                        raise _BudgetStop("GPU_runtime_overrun")\n'
         '                if heartbeat_failure is not None:'),
    ):
        _require(source.count(before) == 1)
        source = source.replace(before, after, 1)
    return source


def _call(name, *arguments):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(arguments), keywords=[])


class _PhaseCaps(ast.NodeTransformer):
    def __init__(self, name):
        self.name, self.changed = name, 0

    def visit_Constant(self, node):
        if self.name == 'unit_command':
            modules = {'scripts.native_source_episode': 'scripts.native_source_episode_005',
                       'scripts.run_native_source_pilot': 'scripts.run_native_source_pilot_005'}
            if type(node.value) is str and node.value in modules:
                self.changed += 1
                return ast.Constant(value=modules[node.value])
            if type(node.value) is int and node.value == 282600:
                self.changed += 1
                return ast.Constant(value=PARENT_SECONDS)
        if type(node.value) is int and node.value == 900:
            self.changed += 1
            if self.name == 'unit_status':
                duration = _call('_source5_episode_seconds',
                    ast.Subscript(value=ast.Name(id='spec', ctx=ast.Load()), slice=ast.Constant(value='unit'), ctx=ast.Load()),
                    ast.Subscript(value=ast.Name(id='spec', ctx=ast.Load()), slice=ast.Constant(value='command'), ctx=ast.Load()))
                return ast.IfExp(test=ast.Compare(left=ast.Subscript(value=ast.Name(id='spec', ctx=ast.Load()),
                    slice=ast.Constant(value='kind'), ctx=ast.Load()), ops=[ast.Eq()],
                    comparators=[ast.Constant(value='pilot')]), body=ast.Constant(value=PARENT_SECONDS), orelse=duration)
            return _call('_source5_episode_seconds', ast.Name(id='unit', ctx=ast.Load()),
                         ast.Name(id='arguments', ctx=ast.Load()))
        return node


class _SingletonSchedule(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict.fromkeys(('batches', 'source', 'total', 'ready'), 0)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == 'batch_index':
            _require(ast.unparse(node.iter) == 'range(3)')
            node.iter = _call('range', ast.Constant(value=12))
            self.counts['batches'] += 1
        if isinstance(node.target, ast.Name) and node.target.id == 'index':
            _require(ast.unparse(node.iter) == 'range(batch_index * 4, batch_index * 4 + 4)')
            node.iter = ast.Tuple(elts=[ast.Name(id='batch_index', ctx=ast.Load())], ctx=ast.Load())
            self.counts['source'] += 1
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) is int and node.value == 21600:
            self.counts['total'] += 1
            return ast.Constant(value=86400)
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'service' and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'ready'):
            self.counts['ready'] += 1
            check = ast.parse('''if (type(info) is dict and info.get("status") == "running"
        and runtime.epoch() - info["started_epoch"] >= 600):
    raise _BudgetStop("model_ready_budget")
''').body[0]
            return [copy.deepcopy(check), node, copy.deepcopy(check)]
        return self.generic_visit(node)


def _patch_core(core, source):
    tree = ast.parse(_v2_lifecycle(source))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    runtime_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_Runtime')
    status = next(node for node in runtime_class.body if isinstance(node, ast.FunctionDef) and node.name == 'unit_status')
    compiled = []
    for name, node, expected in (('unit_command', functions['unit_command'], 4),
                                  ('_unit', functions['_unit'], 1), ('unit_status', status, 1)):
        change = _PhaseCaps(name)
        node = change.visit(node)
        _require(change.changed == expected)
        if name == 'unit_status':
            node.name = '_source5_unit_status'
        compiled.append(node)
    schedule = _SingletonSchedule()
    compiled.append(schedule.visit(functions['run_pilot']))
    _require(schedule.counts == {'batches': 1, 'source': 1, 'total': 1, 'ready': 1})
    core._source5_episode_seconds = _episode_seconds
    module = ast.fix_missing_locations(ast.Module(body=compiled, type_ignores=[]))
    exec(compile(module, str(LEGACY_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    core._Runtime.unit_status = core._source5_unit_status


def _isolated(*, raw=None, raw_legacy=None, _modules=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, 'scripts/run_native_source_pilot_004.py') if raw is None else raw
    raw_legacy = public_read(ROOT, 'scripts/run_native_source_pilot.py') if raw_legacy is None else raw_legacy
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == BASE_SHA256
             and type(raw_legacy) is bytes and hashlib.sha256(raw_legacy).hexdigest() == LEGACY_SHA256)
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_005'),
        'episode': importlib.import_module('scripts.native_source_episode_005'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_005'),
        'actor': importlib.import_module('scripts.native_appworld_actor_004'),
    }
    _require(type(modules) is dict and set(modules) == {'registration', 'episode', 'queue', 'actor'}
             and all(modules[name].ID == ID and modules[name].PLAN == PLAN for name in ('registration', 'episode', 'queue'))
             and callable(modules['queue'].fenced_release)
             and len(modules['registration'].TASK_IDS) == 12
             and modules['registration'].BATCH_TASK_IDS == tuple((task,) for task in modules['registration'].TASK_IDS))
    source = raw.decode('utf-8')
    for old, new, expected in (
        ('qwen38-native-source-pilot-004', ID, 1),
        ('artifacts/qwen38-native-source-pilot-plan-004.json', PLAN, 1),
        ('scripts.native_source_episode_004', 'scripts.native_source_episode_005', 2),
        ('scripts.run_native_source_pilot_004', 'scripts.run_native_source_pilot_005', 1),
        ('scripts._native_source_pilot_004_core', 'scripts._native_source_pilot_005_core', 1),
    ):
        _require(source.count(old) == expected)
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._native_source_pilot_005_base', BASE_SOURCE)
    _require(spec is not None and spec.loader is not None)
    base = importlib.util.module_from_spec(spec)
    tree = ast.parse(source)
    quoted = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
              and node.value == "'" + '"_004"' + "'"]
    _require(len(quoted) == 1)
    quoted[0].value = "'" + '"_005"' + "'"
    exec(compile(ast.fix_missing_locations(tree), str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    core = base._isolated(_modules=modules)
    _patch_core(core, raw_legacy.decode('utf-8'))
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_pilot(plan_sha, root=None, _runtime=None):
    core = _get_core()
    return core.run_pilot(plan_sha, root=core.WORKSPACE if root is None else root, _runtime=_runtime)


def launch_pilot(plan_sha, *, root=None):
    core = _get_core()
    return core.launch_pilot(plan_sha, root=core.WORKSPACE if root is None else root)


def unit_command(*args, **kwargs):
    return _get_core().unit_command(*args, **kwargs)


def initial_slots():
    return _get_core().initial_slots()


def __getattr__(name):
    if name in {'_Runtime', '_Heartbeat', '_queue_state', 'NativeSourcePilotError'}:
        return getattr(_get_core(), name)
    raise AttributeError(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run', action='store_true')
    choice.add_argument('--launch', action='store_true')
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
