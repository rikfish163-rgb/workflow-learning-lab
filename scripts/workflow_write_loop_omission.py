"""Build one private, syntactic write-loop omission without executing code.

Only top-level loops over a direct ``sorted(...)``-shaped call are considered.
The caller supplies the externally bound public write-call catalog and owns
fresh-world execution, replay, safety and scoring. This certificate does not
prove that ``sorted`` resolves to the builtin, that a listed call executes or
changes state, that the omitted iteration matters, or that materialization is
safe. ``mutated_action`` and the original history must remain private.
"""

import ast
import keyword

from scripts.workflow_loop_omission import (
    MAX_RESPONSE_BYTES,
    MAX_STEPS,
    _check_visible_bindings,
    _parse,
)
from wll.protocol import fingerprint
from wll.runner import extract_code

MAX_SELECTED_STEP_INDEX = 27
_TYPE_PARAMETERS = tuple(getattr(ast, name) for name in ('TypeVar', 'TypeVarTuple', 'ParamSpec')
                         if hasattr(ast, name))


def _reject():
    raise ValueError('write_loop_omission_ineligible') from None


def _catalog(write_calls):
    if type(write_calls) is not frozenset:
        _reject()
    for pair in write_calls:
        if (type(pair) is not tuple or len(pair) != 2
                or any(type(value) is not str or not value.isidentifier() or keyword.iskeyword(value)
                       for value in pair)):
            _reject()
    return write_calls


def _check_sorted_bindings(tree):
    """Conservatively reject visible binding/deletion of the added builtin name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id == 'sorted' and isinstance(node.ctx, (ast.Store, ast.Del)):
                _reject()
        elif isinstance(node, ast.arg) and node.arg == 'sorted':
            _reject()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == 'sorted':
                _reject()
        elif isinstance(node, ast.alias):
            if (node.asname or node.name.split('.')[0]) == 'sorted':
                _reject()
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and 'sorted' in node.names:
            _reject()
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name == 'sorted':
                _reject()
        elif isinstance(node, ast.MatchMapping) and node.rest == 'sorted':
            _reject()
        elif isinstance(node, _TYPE_PARAMETERS) and node.name in {'list', 'sorted'}:
            _reject()
        elif isinstance(node, ast.Attribute):
            if node.attr == 'sorted' and isinstance(node.ctx, (ast.Store, ast.Del)):
                _reject()


def _contains_direct_write(loop, write_calls):
    """Look only at syntactically direct calls, excluding deferred definitions."""
    pending = list(loop.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            app = node.func.value
            if (isinstance(app, ast.Attribute) and isinstance(app.value, ast.Name)
                    and app.value.id == 'apis' and (app.attr, node.func.attr) in write_calls):
                return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def _replace_iterator(action, iterator):
    """Wrap the precise UTF-8 byte span, retaining every surrounding byte."""
    raw = action.encode('utf-8')
    lines = raw.splitlines(keepends=True)
    first, last = iterator.lineno, iterator.end_lineno
    left, right = iterator.col_offset, iterator.end_col_offset
    if (type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines)
            or type(left) is not int or type(right) is not int
            or not 0 <= left <= len(lines[first - 1]) or not 0 <= right <= len(lines[last - 1])):
        _reject()
    start = sum(map(len, lines[:first - 1])) + left
    end = sum(map(len, lines[:last - 1])) + right
    if not 0 <= start < end <= len(raw):
        _reject()
    original = raw[start:end].decode('utf-8')
    parsed = ast.parse(original, mode='eval').body
    if ast.dump(parsed, include_attributes=False) != ast.dump(iterator, include_attributes=False):
        _reject()
    return (raw[:start] + b'list(' + raw[start:end] + b')[:-1]' + raw[end:]).decode('utf-8')


def _verify_rewrite(tree, statement_index, original_iterator):
    statement = tree.body[statement_index]
    if not isinstance(statement, ast.For) or not isinstance(statement.iter, ast.Subscript):
        _reject()
    selected = statement.iter
    if (not isinstance(selected.value, ast.Call) or not isinstance(selected.value.func, ast.Name)
            or selected.value.func.id != 'list' or len(selected.value.args) != 1
            or selected.value.keywords or not isinstance(selected.slice, ast.Slice)
            or selected.slice.lower is not None or selected.slice.step is not None
            or not isinstance(selected.slice.upper, ast.UnaryOp)
            or not isinstance(selected.slice.upper.op, ast.USub)
            or not isinstance(selected.slice.upper.operand, ast.Constant)
            or type(selected.slice.upper.operand.value) is not int
            or selected.slice.upper.operand.value != 1
            or ast.dump(selected.value.args[0], include_attributes=False)
               != ast.dump(original_iterator, include_attributes=False)):
        _reject()


def build_write_loop_omission(steps, *, write_calls):
    """Select the last eligible pivot through index 27, or return ``None``.

    All 0..40 actions and observations must be UTF-8 strings and are scanned
    for visible ``list``/``sorted`` bindings and reflective contexts. Only the
    selected step's bounded raw response is required to extract to its exact
    action; extra row metadata and all observation values are ignored for
    selection and never returned. Invalid inputs raise one content-free error.
    """
    try:
        write_calls = _catalog(write_calls)
        if type(steps) is not list or len(steps) > MAX_STEPS:
            _reject()
        selected = None
        for index, step in enumerate(steps):
            if type(step) is not dict or type(step.get('observation')) is not str:
                _reject()
            step['observation'].encode('utf-8')
            action = step.get('action')
            tree, completion = _parse(action)
            _check_visible_bindings(tree)
            _check_sorted_bindings(tree)
            if index > MAX_SELECTED_STEP_INDEX or completion['status'] != 'no_match':
                continue
            for statement_index, statement in enumerate(tree.body):
                if (not isinstance(statement, ast.For)
                        or not isinstance(statement.target, (ast.Name, ast.Tuple))
                        or not isinstance(statement.iter, ast.Call)
                        or not isinstance(statement.iter.func, ast.Name)
                        or statement.iter.func.id != 'sorted' or statement.orelse
                        or any(isinstance(node, (ast.Break, ast.Continue)) for node in ast.walk(statement))
                        or not _contains_direct_write(statement, write_calls)):
                    continue
                selected = (index, statement_index, action, statement.iter)
        if selected is None:
            return None
        index, statement_index, action, iterator = selected
        response = steps[index].get('raw_response')
        if (type(response) is not str or len(response) > MAX_RESPONSE_BYTES
                or len(response.encode('utf-8')) > MAX_RESPONSE_BYTES or extract_code(response) != action):
            _reject()
        mutated = _replace_iterator(action, iterator)
        changed, completion = _parse(mutated)
        if completion['status'] != 'no_match':
            _reject()
        _verify_rewrite(changed, statement_index, iterator)
        return {'step_index': index, 'loop_index': statement_index,
                'original_action_sha256': fingerprint(action),
                'mutated_action_sha256': fingerprint(mutated), 'mutated_action': mutated}
    except (ValueError, TypeError, UnicodeError, SyntaxError, RecursionError, MemoryError, OverflowError):
        _reject()
