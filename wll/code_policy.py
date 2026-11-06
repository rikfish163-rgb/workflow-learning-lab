"""Additional benchmark-integrity restrictions, not a Python security sandbox.

Docker remains the host boundary. This intentionally restricted Python subset
excludes reflection and environment internals. Apply identically to all arms.
"""
import ast

ALLOWED_IMPORTS = frozenset({"builtins", "calendar", "collections", "copy", "datetime",
                             "functools", "itertools", "json", "math", "pendulum",
                             "random", "re", "statistics"}) - {"builtins"}
FORBIDDEN_NAMES = frozenset({"open", "eval", "exec", "compile", "globals", "locals", "vars",
                            "dir", "getattr", "setattr", "delattr", "hasattr", "type", "super",
                            "object", "help", "breakpoint", "input", "memoryview"})
FORBIDDEN_ATTRIBUTES = frozenset({"f_globals", "f_locals", "f_back", "gi_frame", "gi_code",
                                 "cr_frame", "cr_code", "tb_frame", "tb_next",
                                 "sys", "os", "io", "builtins", "importlib", "inspect",
                                 "modules", "tool", "format", "format_map",
                                 "codecs", "pathlib", "Path", "open", "read_text", "read_bytes",
                                 "write_text", "write_bytes"})


def violation(code: str) -> str | None:
    if len(code) > 100_000:
        return "code exceeds length budget"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Let AppWorld report syntax errors normally.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name not in ALLOWED_IMPORTS for a in node.names):
                return "import outside allowed standard-library subset"
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module not in ALLOWED_IMPORTS:
                return "import outside allowed standard-library subset"
            if any(a.name.startswith("_") or a.name in FORBIDDEN_ATTRIBUTES for a in node.names):
                return "private import"
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            return "reflection or private name"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                return "reflection or private attribute"
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                return "attribute mutation"
        if isinstance(node, ast.ClassDef):
            return "class definitions are outside the workflow subset"
    return None
