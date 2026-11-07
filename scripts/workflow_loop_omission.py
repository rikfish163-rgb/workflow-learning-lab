"""Select one bounded, text-preserving loop omission without executing code.

The caller must keep ``mutated_action`` private. Hashes use the project's
``fingerprint`` encoding. ``loop_index`` is the zero-based statement index in
the selected action's module body, and ``step_index`` is zero-based.

The entire supplied source prefix is scanned, including steps after index 35,
for visible ``list`` bindings and common reflective forms. This deliberately
rejects some harmless inputs and is not a proof about hidden aliases or runtime
state. A name can hold a generator: materializing it may itself have side
effects or fail to terminate, so callers must use an independently bounded
world and must not infer the intervention's effect from its syntax.
"""

import ast
import unicodedata

from scripts.completion_boundary import MAX_ACTION_BYTES, inspect_completion_action
from wll.protocol import fingerprint
from wll.runner import extract_code

MAX_STEPS = 40
MAX_SELECTED_STEP_INDEX = 35
MAX_RESPONSE_BYTES = 262_144

_REFLECTIVE_NAMES = frozenset({
    '__builtins__', '__import__', 'builtins', 'compile', 'delattr', 'eval',
    'exec', 'getattr', 'globals', 'locals', 'setattr', 'vars',
})
_INVALID_ACTION_REASONS = frozenset({
    'invalid_input_type', 'invalid_utf8', 'input_too_large', 'syntax_error',
    'parse_resource_error', 'ast_too_large', 'ast_too_deep',
})


def _reject():
    raise ValueError('loop_omission_ineligible') from None


def _parse(action):
    if type(action) is not str:
        _reject()
    try:
        if len(action) > MAX_ACTION_BYTES or len(action.encode('utf-8')) > MAX_ACTION_BYTES:
            _reject()
        classification = inspect_completion_action(action)
        if classification['reason'] in _INVALID_ACTION_REASONS:
            _reject()
        tree = ast.parse(action, mode='exec')
        # Compile for structural validity only; the resulting code is never run.
        compile(tree, '<loop-omission>', 'exec', dont_inherit=True)
    except (SyntaxError, TypeError, UnicodeError, RecursionError, MemoryError):
        _reject()
    return tree, classification


def _check_visible_bindings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in _REFLECTIVE_NAMES or (
                    node.id == 'list' and isinstance(node.ctx, (ast.Store, ast.Del))):
                _reject()
        elif isinstance(node, ast.arg) and node.arg == 'list':
            _reject()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == 'list':
                _reject()
        elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'builtins':
            _reject()
        elif isinstance(node, ast.alias):
            if (node.name == '*' or node.name.split('.')[0] == 'builtins'
                    or (node.asname or node.name.split('.')[0]) == 'list'):
                _reject()
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and 'list' in node.names:
            _reject()
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name == 'list':
                _reject()
        elif isinstance(node, ast.MatchMapping) and node.rest == 'list':
            _reject()
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') or (
                    node.attr == 'list' and isinstance(node.ctx, (ast.Store, ast.Del))):
                _reject()


def _replace_iterator(action, iterator):
    try:
        raw = action.encode('utf-8')
        lines = raw.splitlines(keepends=True)
        if (type(iterator.lineno) is not int or iterator.lineno < 1
                or iterator.end_lineno != iterator.lineno
                or iterator.lineno > len(lines)
                or type(iterator.col_offset) is not int
                or type(iterator.end_col_offset) is not int
                or not 0 <= iterator.col_offset < iterator.end_col_offset):
            _reject()
        line = lines[iterator.lineno - 1]
        if iterator.end_col_offset > len(line):
            _reject()
        token = line[iterator.col_offset:iterator.end_col_offset]
        name = token.decode('utf-8')
        if not name.isidentifier() or unicodedata.normalize('NFKC', name) != iterator.id:
            _reject()
        start = sum(map(len, lines[:iterator.lineno - 1])) + iterator.col_offset
        end = start + len(token)
        return (raw[:start] + b'list(' + token + b')[:-1]' + raw[end:]).decode('utf-8')
    except (UnicodeError, RecursionError, MemoryError):
        _reject()


def build_loop_omission(steps):
    """Return a private mutation, None for no candidate, or a fixed ValueError.

    ``steps`` must be a list of at most 40 dictionaries with string ``action``
    fields. The selected dictionary must additionally have a string
    ``raw_response`` whose extracted code exactly equals its action. Only a
    top-level ``for <Name> in <Name>`` in an action with no visible completion
    reference, with no ``else``, ``break``, or ``continue``, is eligible.
    Selection is the last eligible top-level statement in the last eligible
    action at index 0 through 35, inclusive.
    """
    if type(steps) is not list or len(steps) > MAX_STEPS:
        _reject()
    selected = None
    for index, step in enumerate(steps):
        if type(step) is not dict:
            _reject()
        action = step.get('action')
        tree, classification = _parse(action)
        _check_visible_bindings(tree)
        if index > MAX_SELECTED_STEP_INDEX or classification['status'] != 'no_match':
            continue
        for statement_index, statement in enumerate(tree.body):
            if (isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
                    and isinstance(statement.iter, ast.Name) and not statement.orelse
                    and not any(isinstance(node, (ast.Break, ast.Continue))
                                for node in ast.walk(statement))):
                selected = (index, statement_index, action, statement.iter)
    if selected is None:
        return None
    index, statement_index, action, iterator = selected
    response = steps[index].get('raw_response')
    try:
        if (type(response) is not str or len(response) > MAX_RESPONSE_BYTES
                or len(response.encode('utf-8')) > MAX_RESPONSE_BYTES
                or extract_code(response) != action):
            _reject()
        mutated = _replace_iterator(action, iterator)
        _parse(mutated)
        return {'step_index': index, 'loop_index': statement_index, 'mutated_action': mutated,
                'original_action_sha256': fingerprint(action),
                'mutated_action_sha256': fingerprint(mutated)}
    except (TypeError, UnicodeError, RecursionError, MemoryError):
        _reject()
