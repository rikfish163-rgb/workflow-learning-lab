"""Private ordinary-control recovery inputs, with independent primary auditing.

The source006 policy fixes selection and target rules before outcomes. This
read-only adapter authenticates the complete terminal source and recomputes both
questions; primary failure never becomes a primary pass. All returned histories,
target records and manifests are PRIVATE post-run consistency evidence. A later
writer registration must freeze this code and its output before any model call.
Import starts no source, Task, model, queue, scorer or accelerator operation.
"""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from scripts import native_source_episode_006 as episode6
from scripts import native_source_gpu_queue_006 as queue6
from scripts import native_source_registration_006 as registration6
from scripts import run_native_source_pilot_006 as pilot6
from scripts.native_appworld_replay import _observation
from scripts.native_appworld_worker_client import NativeWorkerClientError, _decode, _encode
from scripts.native_experience_writer_001 import SYSTEM
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_experience_sources_005.py"
BASE_SHA256 = "4ede854294696a1d9e0e821a6a38d11184f6c3e90739defc9b47c7d545be2227"
SOURCE_GATE_KIND = "ordinary_control_recovery_gate"
_ERROR = "native_control_recovery_sources_invalid"
_CORE = None


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def control_targets(generations):
    """Apply the registered response/action filter; never edit control context."""
    _require(type(generations) is list)
    targets = {}
    for index, generation in enumerate(generations):
        try:
            response, action, reply = generation["response"], generation["action"], generation["reply"]
            raw = response["raw_reply"]
            choice = raw["choices"][0]
            messages = copy.deepcopy(generation["intent"]["messages"])
            _require(response.get("format_error") is None and response.get("text") == action
                     and reply.get("ok") is True and type(raw["choices"]) is list and len(raw["choices"]) == 1
                     and choice.get("finish_reason") == "stop"
                     and choice["message"] == {"role": "assistant", "content": action})
            parsed = _decode(action.encode("utf-8"))
            _require(type(parsed) is dict and set(parsed) == {"name", "arguments"}
                     and type(parsed["name"]) is str and bool(parsed["name"])
                     and type(parsed["arguments"]) is dict)
            context_sha = _message_payload(messages)[1]
            digest = hashlib.sha256(_encode(parsed)).hexdigest()
            targets.setdefault(digest, {"target_sha256": digest, "action": action, "student_messages": messages,
                "source_messages_sha256": context_sha, "student_messages_sha256": context_sha,
                "new_generation_index": index})
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, NativeWorkerClientError):
            continue
    return list(targets.values())


def build_control_writer_messages(actor_start_messages, control_records, *, prefix_length):
    """Retain the complete authenticated control history and final observation."""
    try:
        _message_payload(actor_start_messages)
        _require(type(prefix_length) is int and 1 <= prefix_length <= 54
                 and type(control_records) is list and 1 <= len(control_records) <= 24
                 and len(actor_start_messages) == 2 + 2 * prefix_length
                 and actor_start_messages[0]["role"] == "system" and actor_start_messages[1]["role"] == "user"
                 and all(actor_start_messages[index]["role"] == ("assistant" if index % 2 == 0 else "user")
                         for index in range(2, len(actor_start_messages))))
        history = copy.deepcopy(actor_start_messages)
        for row in control_records:
            _require(type(row) is dict and set(row) == {"action", "observation"}
                     and type(row["action"]) is str and bool(row["action"]))
            _observation(row["observation"])
            history.extend([{"role": "assistant", "content": row["action"]},
                            {"role": "user", "content": _encode(row["observation"]).decode("ascii")}])
        result = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(
            {"history": history}, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))}]
        _message_payload(result)
        return result
    except Exception:
        raise ValueError(_ERROR) from None


def _function(source, name):
    found = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    _require(len(found) == 1 and not found[0].decorator_list)
    segment = ast.get_source_segment(source, found[0])
    _require(type(segment) is str)
    return segment


def _replace_once(source, before, after):
    _require(source.count(before) == 1)
    return source.replace(before, after, 1)


def _isolated(*, raw=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, "scripts/native_experience_sources_005.py") if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == BASE_SHA256)
    spec = importlib.util.spec_from_file_location("scripts._native_control_recovery_sources_006_base", BASE_SOURCE)
    _require(spec is not None and spec.loader is not None)
    parent = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), "exec", dont_inherit=True), parent.__dict__)
    parent.queue5 = queue6
    core = parent._isolated()
    core.registration4, core.episode4, core.pilot4 = registration6, episode6, pilot6
    core.control_targets, core.build_control_writer_messages = control_targets, build_control_writer_messages
    core._ERROR = _ERROR
    legacy = public_read(ROOT, "scripts/native_experience_sources_001.py")
    _require(hashlib.sha256(legacy).hexdigest() == parent.BASE_SHA256)
    legacy = legacy.decode("utf-8")
    terminal = _function(legacy, "_terminal_claim")
    for before, after in (
        ('gate.get("status") == "passed"', 'gate.get("status") in {"passed", "insufficient"}'),
        ('_integer(gate.get("complete_help_scenarios"), 4)', '_integer(gate.get("complete_help_scenarios"), 0)'),
        ('_integer(gate.get("distinct_student_targets"), 20)', '_integer(gate.get("distinct_student_targets"), 0)'),
        ('len(hashes) >= 20', 'len(hashes) >= 0'),
    ):
        terminal = _replace_once(terminal, before, after)
    function = _function(legacy, "load_qualified_sources")
    for before, after in (
        ('        core, episode = pilot4._get_core(), episode4._get_core()',
         '        _require(registration4._same(plan["design"].get("secondary_analysis"), registration4.secondary_design()))\n'
         '        core, episode = pilot4._get_core(), episode4._get_core()'),
        ('len(jobs) == 3', 'len(jobs) == 12'),
        ('jobs[index // 4]', 'jobs[index]'),
        ('        context_binding = plan["context_proof"]',
         '        _validate_jobs(jobs, plan_sha256)\n        _validate_cost_rows(report)\n'
         '        context_binding = plan["context_proof"]'),
        ('            metadata, records, generations = core.actor_evidence(',
         '            _validate_phase_cost(report, core, index, phase, receipt, journal, limits, config["initial_context"])\n'
         '            metadata, records, generations = core.actor_evidence('),
        ('        targets, helped = {}, []', '        targets, helped = {}, []\n        ordinary_targets, recoveries = {}, []'),
        ('            phase_evidence(index, "control")',
         '            _, control_records, control_generations, control_start = phase_evidence(index, "control")'),
        ('            _require(slot["pair_class"] == pair)',
         '            _require(slot["pair_class"] == pair)\n'
         '            if slot["first_attempt"]["official_success"] is False and control["model_reported_complete"] is True and control["official_success"] is True:\n'
         '                for target in control_targets(control_generations):\n'
         '                    ordinary_targets.setdefault(target["target_sha256"], {"task_id": slot["task_id"], **target})\n'
         '                recoveries.append((index, slot["task_id"].rsplit("_", 1)[0], control_start, control_records, len(prefix["prefix_records"])))'),
        ('        gate = {"status": "passed", "complete_help_scenarios": len({row[1] for row in helped}),',
         '        gate = {"status": ("passed" if len({row[1] for row in helped}) >= 4 and len(targets) >= 20 else "insufficient"),\n'
         '                "complete_help_scenarios": len({row[1] for row in helped}),'),
        ('        _require(gate["complete_help_scenarios"] >= 4 and gate["distinct_student_targets"] >= 20\n'
         '                 and core.registration._same(report["data_feasibility_gate"], gate)',
         '        _require(core.registration._same(report["data_feasibility_gate"], gate)'),
        ('        writer_inputs = []',
         '        primary_gate = gate\n'
         '        gate = {"status": "passed", "complete_recovery_scenarios": len({row[1] for row in recoveries}),\n'
         '                "distinct_student_targets": len(ordinary_targets), "source_denominator": 12}\n'
         '        _require(gate["complete_recovery_scenarios"] >= 4 and gate["distinct_student_targets"] >= 20)\n'
         '        helped = recoveries\n        writer_inputs = []'),
        ('messages = build_writer_messages(start, records, prefix_length=prefix_length, fixed_hint=hint)',
         'messages = build_control_writer_messages(start, records, prefix_length=prefix_length)'),
        ('            "private": True, "source_plan_sha256": plan_sha256, "source_result_sha256": result_sha256,',
         '            "private": True, "source_gate_kind": "ordinary_control_recovery_gate",\n'
         '            "primary_data_feasibility_gate": primary_gate, "primary_candidate_student_target_sha256": sorted(targets),\n'
         '            "ordinary_control_targets": list(ordinary_targets.values()),\n'
         '            "ordinary_control_target_sha256": sorted(ordinary_targets),\n'
         '            "ordinary_targets_scope": "Post-run consistency; source006 precommits deterministic filter, not these records.",\n'
         '            "source_plan_sha256": plan_sha256, "source_result_sha256": result_sha256,'),
        ('return {"private": True, "gate": gate, "writer_inputs": writer_inputs,',
         'return {"private": True, "source_gate_kind": "ordinary_control_recovery_gate", "gate": gate, "writer_inputs": writer_inputs,'),
    ):
        function = _replace_once(function, before, after)
    for code in (terminal, function):
        exec(compile(code, str(parent.BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    return core


def load_qualified_sources(root, plan_sha256, result_sha256):
    """Require full source authentication, retain primary result, qualify control bank."""
    global _CORE
    try:
        if _CORE is None:
            _CORE = _isolated()
        return _CORE.load_qualified_sources(root, plan_sha256, result_sha256)
    except Exception:
        raise ValueError(_ERROR) from None
