"""Restricted actor-code checks and static variable candidates, not a sandbox.

The worker must still isolate trusted state and independently enforce its own
read-only protocol. Candidate names alone never prove an assignment executed or
that a value came from actor-visible history.
"""

import ast
import re

from wll.code_policy import violation

_PUBLIC_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
_CONTAINER_METHODS = frozenset({
    'update', 'clear', 'pop', 'popitem', 'setdefault', 'append', 'extend',
    'insert', 'remove', 'discard', 'reverse', 'sort', 'get', 'keys', 'items',
    'values', 'copy', 'fromkeys',
})
_FRAMEWORK_NAMES = frozenset({
    'get_ipython', 'shell', 'user_ns', 'request_tracker', 'builtins', 'apicollection',
})


def _protected_name(name):
    lowered = name.casefold()
    return name == 'apis' or lowered in _FRAMEWORK_NAMES or 'requester' in lowered


def _direct_api_call(node, parents):
    app = parents.get(node)
    endpoint = parents.get(app)
    call = parents.get(endpoint)
    return (isinstance(node.ctx, ast.Load)
            and isinstance(app, ast.Attribute) and app.value is node and isinstance(app.ctx, ast.Load)
            and app.attr.casefold() != 'admin'
            and isinstance(endpoint, ast.Attribute) and endpoint.value is app
            and isinstance(endpoint.ctx, ast.Load) and endpoint.attr not in _CONTAINER_METHODS
            and isinstance(call, ast.Call) and call.func is endpoint)


def actor_code_restriction(code):
    """Return a fixed reason or None; permit only direct three-part API calls."""
    if type(code) is not str:
        return 'invalid code type'
    try:
        base = violation(code)
        if base is not None:
            return base
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Preserve existing worker behavior for ordinary syntax errors.
    except (ValueError, RecursionError, MemoryError):
        return 'code cannot be inspected'

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id != 'apis' and _protected_name(node.id):
                return 'framework reference restricted'
            if node.id == 'apis' and not _direct_api_call(node, parents):
                return 'apis must be a direct endpoint call'
        elif isinstance(node, ast.Attribute) and _protected_name(node.attr):
            return 'framework reference restricted'
        elif isinstance(node, ast.arg) and _protected_name(node.arg):
            return 'protected name binding restricted'
        elif isinstance(node, ast.alias):
            bound = node.asname or node.name.split('.')[0]
            if _protected_name(bound) or 'requester' in node.name.casefold():
                return 'protected name binding restricted'
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _protected_name(node.name):
                return 'protected name binding restricted'
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if any(_protected_name(name) for name in node.names):
                return 'protected name binding restricted'
        elif isinstance(node, ast.ExceptHandler) and node.name and _protected_name(node.name):
            return 'protected name binding restricted'
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name and _protected_name(node.name):
            return 'protected name binding restricted'
        elif isinstance(node, ast.MatchMapping) and node.rest and _protected_name(node.rest):
            return 'protected name binding restricted'
    return None


class _AssignedNames(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def target(self, target):
        if isinstance(target, ast.Name):
            name = target.id
            if _PUBLIC_NAME.fullmatch(name) and not _protected_name(name):
                self.names.add(name)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                self.target(child)
        elif isinstance(target, ast.Starred):
            self.target(target.value)

    def visit_Assign(self, node):
        for target in node.targets:
            self.target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.target(node.target)
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self.target(node.target)
        self.visit(node.value)

    def visit_For(self, node):
        self.target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self.target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_NamedExpr(self, node):
        self.target(node.target)
        self.visit(node.value)

    def _skip_scope(self, _node):
        pass

    visit_FunctionDef = _skip_scope
    visit_AsyncFunctionDef = _skip_scope
    visit_ClassDef = _skip_scope
    visit_Lambda = _skip_scope
    visit_ListComp = _skip_scope
    visit_SetComp = _skip_scope
    visit_DictComp = _skip_scope
    visit_GeneratorExp = _skip_scope


def assigned_actor_names(code):
    """Return syntactic public assignment candidates, never executed-name proof."""
    if type(code) is not str or len(code) > 100_000:
        return frozenset()
    try:
        tree = ast.parse(code)
        visitor = _AssignedNames()
        visitor.visit(tree)
        return frozenset(visitor.names)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return frozenset()
