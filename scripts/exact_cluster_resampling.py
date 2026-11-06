"""Exact conditional cluster-bootstrap distribution for small binary-success studies.

Independent CPU analysis only. It is not wired into the frozen utility gate or
the running calibration, and exact resampling does not ensure population coverage.
"""
from collections import Counter
from fractions import Fraction


def exact_cluster_resampling(groups, *, max_states=100000):
    if (not groups or type(max_states) is not int or max_states < 1
            or any(not group or any(type(x) is not int or x not in (-1, 0, 1)
                                   for x in group) for group in groups)):
        raise ValueError("nonempty clusters of integer binary-success differences required")
    # A sampled cluster contributes its sum and its number of task/seed pairs.
    # Keep both to match the original task-weighted mean with unequal clusters.
    cluster_types = Counter((sum(group), len(group)) for group in groups)
    states = {(0, 0): 1}
    for _ in groups:
        updated = Counter()
        for (total, size), count in states.items():
            for (delta, width), multiplicity in cluster_types.items():
                updated[total + delta, size + width] += count * multiplicity
                if len(updated) > max_states:
                    raise ValueError("exact resampling state limit exceeded")
        states = updated
    distribution = Counter()
    for (total, size), count in states.items():
        distribution[Fraction(total, size)] += count
    total_count = len(groups) ** len(groups)
    if sum(distribution.values()) != total_count:
        raise RuntimeError("resampling mass is not conserved")
    ordered = sorted(distribution.items())

    def quantile(level):
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative * level.denominator >= total_count * level.numerator:
                return float(value)
        raise RuntimeError("missing resampling quantile")

    return {"method": "exact_conditional_cluster_resampling", "scenario_count": len(groups),
            "pair_count": sum(map(len, groups)), "resample_count": total_count,
            "ci95": [quantile(Fraction(1, 40)), quantile(Fraction(39, 40))],
            "degenerate": len(distribution) == 1,
            "distribution": [{"numerator": value.numerator, "denominator": value.denominator,
                              "count": count} for value, count in ordered],
            "warning": "Exact calculation of the empirical resampling distribution removes Monte Carlo noise. "
                       "It does not establish population coverage, independence, causal attribution or an upper harm-risk guarantee."}
