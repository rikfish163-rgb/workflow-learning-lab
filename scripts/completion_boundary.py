"""Bounded syntax inspection only; never execute an action or infer task outcome.

A candidate is a shape match, not proof of API identity, execution, argument
validity, task completion or replay safety. ``no_match`` cannot rule out aliases
from earlier actions or arbitrary dynamic execution. No cross-action state is kept.
"""

import ast

MAX_ACTION_BYTES = 65_536
MAX_AST_NODES = 4_096
MAX_AST_DEPTH = 128

_COMPLETION_NAMES = frozenset({"complete_task", "task_completed"})
_DIRECT_NAME = ("apis", "supervisor", "complete_task")


def _name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return (node.id, *reversed(parts))
    return None


def _dynamic_lookup(call):
    if isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2:
        owner, key = call.args[:2]
        return _name(owner) == ("apis", "supervisor") or (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value in _COMPLETION_NAMES
        )
    if isinstance(call.func, ast.Attribute) and call.func.attr == "__getattribute__":
        return _name(call.func.value) == ("apis", "supervisor")
    if isinstance(call.func, ast.Subscript):
        key = call.func.slice
        return isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in _COMPLETION_NAMES
    return False


def inspect_completion_action(action) -> dict:
    """Return exactly ``status`` and a fixed ``reason``; export no action content."""
    if type(action) is not str:
        return {"status": "abstain", "reason": "invalid_input_type"}
    if len(action) > MAX_ACTION_BYTES:
        return {"status": "abstain", "reason": "input_too_large"}
    try:
        encoded = action.encode("utf-8")
    except UnicodeEncodeError:
        return {"status": "abstain", "reason": "invalid_utf8"}
    if len(encoded) > MAX_ACTION_BYTES:
        return {"status": "abstain", "reason": "input_too_large"}
    try:
        tree = ast.parse(action, mode="exec")
    except (SyntaxError, ValueError):
        return {"status": "abstain", "reason": "syntax_error"}
    except (RecursionError, MemoryError):
        return {"status": "abstain", "reason": "parse_resource_error"}

    pending = [(tree, 1)]
    nodes = []
    while pending:
        node, depth = pending.pop()
        nodes.append(node)
        if len(nodes) > MAX_AST_NODES:
            return {"status": "abstain", "reason": "ast_too_large"}
        if depth > MAX_AST_DEPTH:
            return {"status": "abstain", "reason": "ast_too_deep"}
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))

    calls = [node for node in nodes if isinstance(node, ast.Call)]
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        call = tree.body[0].value
        if (
            isinstance(call, ast.Call)
            and _name(call.func) == _DIRECT_NAME
            and len(calls) == 1
            and all(isinstance(arg, ast.Constant) for arg in call.args)
            and all(keyword.arg is not None and isinstance(keyword.value, ast.Constant) for keyword in call.keywords)
        ):
            return {"status": "candidate", "reason": "isolated_constant_completion_call"}

    if any(_dynamic_lookup(call) for call in calls):
        return {"status": "abstain", "reason": "dynamic_completion_lookup"}
    if any(
        (isinstance(call.func, ast.Attribute) and call.func.attr in _COMPLETION_NAMES)
        or (isinstance(call.func, ast.Name) and call.func.id in _COMPLETION_NAMES)
        for call in calls
    ):
        return {"status": "abstain", "reason": "completion_not_isolated_or_literal"}
    if any(
        (isinstance(node, ast.Attribute) and node.attr in _COMPLETION_NAMES)
        or (isinstance(node, ast.Name) and node.id in _COMPLETION_NAMES)
        or (isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] in _COMPLETION_NAMES)
        for node in nodes
    ):
        return {"status": "abstain", "reason": "completion_reference_uncertain"}
    return {"status": "no_match", "reason": "no_visible_completion_reference"}
