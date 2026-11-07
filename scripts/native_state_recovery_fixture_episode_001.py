"""Run the native actor with one authored instruction and verify the full trace."""

import hashlib
from pathlib import Path

from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts.native_appworld_worker_client import _decode, _same
from scripts.native_state_refusal_client_001 import NativeWorkerClient


def _require(value):
    if not value:
        raise ValueError("native_state_recovery_fixture_episode_invalid")


def _case_context(row, case):
    _require(row["case"] == case["case"] and row["instruction_sha256"] == protocol._hash(protocol.instructions(case)))
    return {"metadata_sha256": row["actor_metadata_sha256"], "messages_sha256": row["messages_sha256"],
            "input_tokens": row["input_tokens"]}


def _metadata_reply(reply, case, row):
    """Keep original counters/freshness and change only the task instruction."""
    _case_context(row, case)
    _require(type(reply) is dict and set(reply) == {"ok", "result", "cost", "freshness"} and reply["ok"] is True)
    metadata = protocol.actor_metadata(reply["result"], case, row["worker_metadata_sha256"])
    _require(protocol._hash(metadata) == row["actor_metadata_sha256"])
    return {**reply, "result": metadata}


def _bound_rows(rows):
    _require(type(rows) in (list, tuple) and len(rows) == len(protocol.CASES))
    bound = {row["template_task_id"]: (row, case) for row, case in zip(rows, protocol.CASES, strict=True)}
    _require(len(bound) == len(rows))
    for row, case in bound.values():
        _case_context(row, case)
    return bound


def worker_for(rows):
    """Leave the original private worker RPC log intact."""
    bound = _bound_rows(rows)

    class FixtureWorker(NativeWorkerClient):
        def __init__(self, task, experiment, path, **kwargs):
            self.fixture_row, self.fixture_case = bound[task]
            super().__init__(task, experiment, path, **kwargs)

        def request(self, request, *, timeout=None):
            reply = super().request(request, timeout=timeout)
            if request == {"op": "metadata"}:
                return _metadata_reply(reply, self.fixture_case, self.fixture_row)
            return reply

    return FixtureWorker


def authenticate(episode, original, rows):
    """Verify the only allowed metadata substitution before normal replay checks."""
    bound = _bound_rows(rows)

    def verify(receipt, controller, worker, transport, status, fault, *, initial_context, run_dir):
        row, case = bound[receipt["task_id"]]
        _require(_same(_case_context(row, case), initial_context)
                 and type(transport) is list and len(transport) >= 10
                 and _same(transport[2], {"event": "request_intent", "request": {"op": "metadata"}})
                 and type(transport[3]) is dict and set(transport[3]) == {"event", "reply"}
                 and transport[3]["event"] == "request_result")
        reply = _metadata_reply(transport[3]["reply"], case, row)
        _require(_same(reply, episode._one(controller, "metadata_result")["reply"]))
        # The original logs are hash-bound and never edited. The in-memory
        # projection reconciles the one substitution with the actor's view.
        projected = [*transport[:3], {"event": "request_result", "reply": reply}, *transport[4:]]
        result = original(receipt, controller, worker, projected, status, fault,
                          initial_context=initial_context, run_dir=run_dir)
        _metadata, records, _generations = episode._actor_evidence(receipt, controller, initial_context)
        checked = protocol.qualify_witness(records, result["fault_provenance"], case)
        result["fixture"] = {"case": case["case"], "worker_metadata_sha256": row["worker_metadata_sha256"],
                             "actor_metadata_sha256": row["actor_metadata_sha256"], **checked}
        return result

    return verify


def install(episode, rows):
    """Install on an isolated, source-bound module only."""
    _require(episode.__name__.startswith("scripts._native_state_recovery_002_"))
    original_bind = episode.bind
    episode.authenticate_evidence = authenticate(episode, episode.authenticate_evidence, rows)

    def bind(registration, queue):
        bound = original_bind(registration, queue)
        core = bound.core
        core.NativeWorkerClient = worker_for(rows)

        def context_row(_root, _plan, index):
            _require(type(index) is int and 0 <= index < len(rows))
            return {**_case_context(rows[index], protocol.CASES[index]),
                    "task_id": rows[index]["template_task_id"], "initial_fits": True}

        core._context_row = context_row
        return bound

    episode.bind = bind


def collect(episode, core, run_dir, receipt, context):
    """Reconcile durable evidence independently of the controller's projection."""
    path = Path(run_dir)
    names = ("execution-receipt.json", "controller.jsonl", "worker/actions.jsonl", "worker/transport.jsonl", "worker/status.json")
    raw = {name: core._read_private(path / name, maximum=episode.MAX_DOCUMENT) for name in names}
    fault_raw = episode._optional_private(core, path / "worker/fault-provenance.json")
    _require(_same(_decode(raw[names[0]]), receipt))
    result = episode.authenticate_evidence(receipt, episode._journal(raw[names[1]]),
        episode._journal(raw[names[2]]), episode._journal(raw[names[3]]), _decode(raw[names[4]]),
        None if fault_raw is None else _decode(fault_raw), initial_context=context, run_dir=path)
    result["private_files_sha256"] = {str(path / name): hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    if fault_raw is not None:
        result["private_files_sha256"][str(path / "worker/fault-provenance.json")] = hashlib.sha256(fault_raw).hexdigest()
    _require(all(core._read_private(path / name, maximum=episode.MAX_DOCUMENT) == value for name, value in raw.items())
             and episode._optional_private(core, path / "worker/fault-provenance.json") == fault_raw)
    return result
