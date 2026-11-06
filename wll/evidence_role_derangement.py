"""Deterministic structural P-control construction, without runtime data.

The caller freezes role order (tuple order), whole public slot descriptors and
per-role allowed sets before utility evaluation. This helper checks structure,
not public provenance, semantic truth, privacy, token equality or usefulness.
Different slots may alias or return equal values. Nothing here runs an actor,
writer, tokenizer, tool, model, scorer or training job.
"""

import re
from dataclasses import dataclass

MAX_ROLES = 32
MAX_ALLOWED_PER_ROLE = 32
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})
_ERROR = "invalid_evidence_role_metadata"


@dataclass(frozen=True, order=True, slots=True)
class EvidenceSlot:
    """Public metadata only; no source record ID, observation or value field.

    Tuple field order defines canonical lexicographic order. Path segments are
    names, with '*' allowed for collection items; numeric indices are not used.
    An empty path denotes the whole response. Strings are preserved exactly.
    """

    app: str
    api: str
    output_path: tuple[str, ...]
    value_type: str
    public_scope: str


def _valid_slot(slot):
    if type(slot) is not EvidenceSlot:
        return False
    if any(type(x) is not str or _NAME.fullmatch(x) is None
           for x in (slot.app, slot.api, slot.public_scope)):
        return False
    return (type(slot.value_type) is str and len(slot.value_type) <= 7
            and slot.value_type in _TYPES
            and type(slot.output_path) is tuple and len(slot.output_path) <= 8
            and all(type(x) is str and (x == "*" or _NAME.fullmatch(x) is not None)
                    for x in slot.output_path))


def canonical_derangement(bindings, allowed):
    """Return the smallest legal whole-slot permutation, or None if impossible.

    Inputs must be tuples, with a frozen set of allowed slots for each role.
    Each original binding must belong to its allowed set. A P binding must
    differ by slot value, preserve the role's type AND public scope, and belong
    to that same allowed set. The entire original multiset is preserved.

    Duplicate occurrences supply capacities, never an artificial change of
    binding. Zero/one-role and unsatisfiable cases return None; malformed or
    oversized metadata raises a content-free ValueError. Limits bound only
    this CPU helper, not scientific inclusion criteria or candidate acceptance.
    """
    if (type(bindings) is not tuple or type(allowed) is not tuple
            or len(bindings) != len(allowed) or len(bindings) > MAX_ROLES
            or any(not _valid_slot(x) for x in bindings)
            or any(type(group) is not frozenset or len(group) > MAX_ALLOWED_PER_ROLE
                   or any(not _valid_slot(x) for x in group) for group in allowed)):
        raise ValueError(_ERROR)
    if any(old not in legal for old, legal in zip(bindings, allowed, strict=True)):
        raise ValueError(_ERROR)
    size = len(bindings)
    if size < 2:
        return None
    columns = tuple(sorted(range(size), key=lambda j: (bindings[j], j)))
    edges = tuple(frozenset(j for j, new in enumerate(bindings)
                           if new != old and new in legal
                           and new.value_type == old.value_type
                           and new.public_scope == old.public_scope)
                  for old, legal in zip(bindings, allowed, strict=True))

    def feasible(first_row, remaining):
        # Standard augmenting-path matching over occurrence columns. Equal
        # descriptors have identical eligibility and distinct unit capacity.
        matched = {}

        def augment(row, seen):
            for column in remaining:
                if column in seen or column not in edges[row]:
                    continue
                seen.add(column)
                if column not in matched or augment(matched[column], seen):
                    matched[column] = row
                    return True
            return False

        return all(augment(row, set()) for row in range(first_row, size))

    if not feasible(0, columns):
        return None
    result = []
    for row in range(size):
        for column in columns:
            if column not in edges[row]:
                continue
            rest = tuple(j for j in columns if j != column)
            if feasible(row + 1, rest):
                result.append(bindings[column])
                columns = rest
                break
    return tuple(result)
