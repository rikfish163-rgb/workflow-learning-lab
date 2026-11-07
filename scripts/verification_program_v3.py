"""Frozen verification syntax and result rules with the V3 read-only registry.

The policy module is instantiated privately so its endpoint binding changes
without mutating the frozen policy or any other module's globals. Validation
parses source only; this module never executes submitted programs.
"""

import importlib.util
from pathlib import Path

from scripts import verification_readonly_registry_v3 as registry


def _load_policy():
    source = Path(__file__).with_name('verification_program.py')
    spec = importlib.util.spec_from_file_location('scripts._verification_program_v3_core', source)
    if spec is None or spec.loader is None:
        raise ImportError('frozen verification policy unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ALLOWED_APIS = registry.ALLOWED_APIS
    return module


_CORE = _load_policy()
ALLOWED_APIS = registry.ALLOWED_APIS
VerificationProgramError = _CORE.VerificationProgramError
validate_program = _CORE.validate_program
validate_result = _CORE.validate_result
RESULT_NAME = _CORE.RESULT_NAME
SAFE_BUILTINS = _CORE.SAFE_BUILTINS
MAX_SOURCE_BYTES = _CORE.MAX_SOURCE_BYTES
MAX_AST_NODES = _CORE.MAX_AST_NODES
MAX_AST_DEPTH = _CORE.MAX_AST_DEPTH
MAX_LITERAL_STRING_BYTES = _CORE.MAX_LITERAL_STRING_BYTES
MAX_LITERAL_ITEMS = _CORE.MAX_LITERAL_ITEMS
MAX_RANGE_ITEMS = _CORE.MAX_RANGE_ITEMS
MAX_LOOP_NESTING = _CORE.MAX_LOOP_NESTING
MAX_RESULT_BYTES = _CORE.MAX_RESULT_BYTES
AUDIT_LIMITATION = _CORE.AUDIT_LIMITATION
