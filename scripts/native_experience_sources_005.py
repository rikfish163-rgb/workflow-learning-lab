"""Read-only source005 qualification over the authenticated source001 loader.

All returned histories and manifests are PRIVATE. The caller freezes the output
manifest, this adapter, its authenticated base and the source005 code/plan in a
separate writer registration before any call. No actual plan or protected file
is accessed on import; no task, scorer, model, queue or rerun is started here.
Post-run file hashes bind consistency, not an original journal precommitment.
"""

import hashlib
import importlib.util
import math
from pathlib import Path

from scripts import native_source_gpu_queue_005 as queue5
from scripts.native_appworld_worker_client import _encode

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_experience_sources_001.py"
BASE_SHA256 = "9ae4b37e518cda5fab989d9b3555410b6fe07f0f603f4a6cfe2343030a8114ae"
_ERROR = "native_experience_sources_invalid"
_CORE = None


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_jobs(jobs, plan_sha):
    _require(type(jobs) is list and len(jobs) == 12)
    duration = 0
    for index, value in enumerate(jobs):
        _require(type(value) is dict and value.get("unknown_cost") is False)
        job, release = value.get("job"), value.get("release")
        _require(type(job) is dict and _encode(job) == _encode(queue5.make_job(index, job.get("owner"), plan_sha))
                 and type(release) is dict and release.get("verified") is True
                 and release.get("runtime_overrun") is False and _finite(release.get("GPU_seconds"))
                 and release["GPU_seconds"] <= 7200 and _finite(release.get("service_wall_seconds"))
                 and release["service_wall_seconds"] == release["GPU_seconds"])
        duration += release["GPU_seconds"]
    _require(duration <= 86400)


def _validate_cost_rows(report):
    expected = {(index, phase) for index, slot in enumerate(report["slots"])
                for phase in ("first_attempt", "control", "review") if slot[phase]["status"] == "known"}
    for name in ("episodes", "scorers"):
        rows, found = report["costs"][name], set()
        _require(type(rows) is list)
        for row in rows:
            _require(type(row) is dict and type(row.get("source_index")) is int
                     and type(row.get("phase")) is str and row.get("unknown_cost") is False)
            key = row["source_index"], row["phase"]
            _require(key in expected and key not in found)
            found.add(key)
            if name == "episodes":
                _require(row.get("counts_complete") is True and row.get("generation_accounting_valid") is True
                         and _finite(row.get("parent_wall_seconds")))
            else:
                _require(all(row.get(key) is True for key in ("attempted", "result_validated", "cleanup_verified"))
                         and _finite(row.get("wall_seconds")))
        _require(found == expected)
    expected_pairs = {index for index, slot in enumerate(report["slots"])
                      if slot["pair_class"] != "ineligible_no_boundary"}
    paired, found = report["costs"]["paired_CPU_counts"], set()
    _require(type(paired) is list)
    for row in paired:
        _require(type(row) is dict and type(row.get("source_index")) is int
                 and row["source_index"] in expected_pairs and row["source_index"] not in found
                 and row.get("unknown_cost") is False and row.get("counts_complete") is True
                 and type(row.get("CPU_calls")) is int and row["CPU_calls"] == 2
                 and _finite(row.get("wall_seconds")) and type(row.get("counts")) is dict
                 and set(row["counts"]) == {"control", "review"})
        found.add(row["source_index"])
    _require(found == expected_pairs)


def _validate_phase_cost(report, core, index, phase, receipt, journal, limits, context):
    cost = core._costs(receipt, journal)
    declared = next(row for row in report["costs"]["episodes"]
                    if (row["source_index"], row["phase"]) == (index, phase))
    _require(cost.get("unknown_cost") is False and cost.get("counts_complete") is True
             and cost.get("generation_accounting_valid") is True
             and all(core.registration._same(declared.get(key), value) for key, value in cost.items())
             and _finite(cost.get("controller_elapsed_seconds"))
             and cost.get("CPU_counting_setup_save_cleanup_included_in_parent_wall") is True)
    for key, maximum in (("generations", limits.max_generations), ("prompt_tokens", limits.max_prompt_tokens_total),
            ("completion_tokens", limits.max_completion_tokens_total), ("action_attempts", limits.max_actions),
            ("api_invocations", limits.max_api_requests), ("upstream_requests", limits.max_api_requests)):
        _require(type(cost.get(key)) is int and 0 <= cost[key] <= maximum)
    if phase != "first_attempt":
        paired = next(row for row in report["costs"]["paired_CPU_counts"] if row["source_index"] == index)
        _require(core.registration._same(paired["counts"][phase], context))


def _replace_once(source, before, after):
    _require(source.count(before) == 1)
    return source.replace(before, after, 1)


def _isolated(*, raw=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, "scripts/native_experience_sources_001.py") if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == BASE_SHA256)
    source = raw.decode("utf-8")
    for before, after in (
        ("native_source_episode_004 as episode4", "native_source_episode_005 as episode4"),
        ("native_source_registration_004 as registration4", "native_source_registration_005 as registration4"),
        ("run_native_source_pilot_004 as pilot4", "run_native_source_pilot_005 as pilot4"),
        ("len(jobs) == 3", "len(jobs) == 12"),
        ("jobs[index // 4]", "jobs[index]"),
        ("        context_binding = plan[\"context_proof\"]",
         "        _validate_jobs(jobs, plan_sha256)\n"
         "        _validate_cost_rows(report)\n"
         "        context_binding = plan[\"context_proof\"]"),
        ("            metadata, records, generations = core.actor_evidence(",
         "            _validate_phase_cost(report, core, index, phase, receipt, journal, limits, config[\"initial_context\"])\n"
         "            metadata, records, generations = core.actor_evidence("),
    ):
        source = _replace_once(source, before, after)
    spec = importlib.util.spec_from_file_location("scripts._native_experience_sources_005_base", BASE_SOURCE)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    core.__dict__.update(_validate_jobs=_validate_jobs, _validate_cost_rows=_validate_cost_rows,
                         _validate_phase_cost=_validate_phase_cost)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    return core


def load_qualified_sources(root, plan_sha256, result_sha256):
    """Return the same private gate/input/manifest shape after complete recomputation."""
    global _CORE
    try:
        if _CORE is None:
            _CORE = _isolated()
        return _CORE.load_qualified_sources(root, plan_sha256, result_sha256)
    except Exception:
        raise ValueError(_ERROR) from None
