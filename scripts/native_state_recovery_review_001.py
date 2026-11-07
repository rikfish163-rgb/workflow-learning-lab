"""Authenticate terminal state-recovery writer outcomes without new model calls."""

import hashlib
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from scripts import native_state_recovery_writer_evidence_001 as evidence
from scripts import native_state_recovery_writer_stage_001 as stage
from scripts import report_native_state_recovery_source_003 as audit
from scripts import run_native_state_recovery_writer_001 as writer
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _same
from scripts.native_experience_review_001 import _RecordedModel
from scripts.native_source_registration import _sha

BUNDLE_PREFIX = "native-state-recovery-experience-001:"
SOURCE_GATE_KIND = "verified_state_transition_after_predispatch_refusal_v1"
MANIFEST = "review-manifest.json"
SOURCE_RESULT = Path(evidence.source.PRIVATE_RUN_ROOT) / "study-result.json"


@dataclass(frozen=True)
class ReviewLineage:
    writer: ModuleType
    source_gate_kind: str
    bundle_prefix: str
    manifest_id: str

    @property
    def source_result(self):
        return Path(self.writer.source.PRIVATE_RUN_ROOT) / "study-result.json"


LINEAGE = ReviewLineage(writer, SOURCE_GATE_KIND, BUNDLE_PREFIX, "native-state-recovery-review-manifest-001")


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_review_invalid")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _one(rows, event):
    result = [row for row in rows if row["event"] == event]
    _require(len(result) == 1)
    return result[0]


def _read_source_result(lineage):
    reg, queue, episode, _runner = lineage.writer.source.bindings()
    core = episode.bind(reg, queue).core
    return core._read_private(lineage.source_result, maximum=episode.MAX_DOCUMENT)


def replay(cases, events, *, backend, model_id):
    """Recompute exact writer/screen/review events from the persisted responses."""
    replayed = []

    def client(tokens):
        _require(tokens in (stage.WRITER_MAX_TOKENS, stage.REVIEW_MAX_TOKENS))
        reviewer = tokens == stage.REVIEW_MAX_TOKENS
        prefix = "review" if reviewer else "writer"
        rows = [{key: value for key, value in row.items() if key != "stage"}
                for row in events if row.get("stage") == prefix]
        messages = evidence.writer_messages(cases)
        if reviewer:
            draft = _one([row for row in events if row.get("stage") == "writer"], "generation_result")
            candidate = stage.writer.parse_candidate(draft["response"]["text"])
            messages = evidence.review_messages(cases, candidate)
        return _RecordedModel(rows, messages=messages, seed=stage.REVIEW_SEED if reviewer else stage.WRITER_SEED,
                              max_tokens=tokens, prefix=prefix, backend=backend, model_id=model_id)

    result = stage.run(cases, model_factory=client, guard=lambda: None, record=replayed.append)
    _require(result["status"] in {"plausible_pending_utility", "abstained", "structurally_rejected", "rejected"}
             and _same(replayed, events) and result["writer"]["cost"]["unknown_cost"] is False
             and (result["review"] is None or result["review"]["cost"]["unknown_cost"] is False)
             and result["training_authorized"] is result["utility_authorized"] is False)
    return result


def _terminal_writer(writer_plan_sha256, writer_result_sha256, lineage):
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and _sha(writer_plan_sha256) and _sha(writer_result_sha256))
    writer = lineage.writer
    reg, queue, runner = writer.bindings()
    plan = reg.load_registration(WORKSPACE, writer_plan_sha256, private=True)
    prepared = writer._prepare(plan)
    core = writer.core_for(runner)
    runtime = core._Runtime(WORKSPACE)
    root = Path(reg.PRIVATE_RUN_ROOT)
    snapshots = {}

    def read(name):
        path = root / name
        raw = runtime.read(path)
        snapshots[path] = _hash(raw)
        return raw

    launch = _decode(read("parent-launch-intent.json"))
    _require(launch["plan_sha256"] == writer_plan_sha256
             and re.fullmatch(r"wll-native-source-pilot-[a-f0-9]{32}\.service", launch["unit"])
             and _main_pid(launch["unit"]) == 0)
    result_raw = read("study-result.json")
    result = _decode(result_raw)
    journal = core.episode._journal(read("study-journal.jsonl"))
    _require(_hash(result_raw) == writer_result_sha256 and result["plan_sha256"] == writer_plan_sha256
             and result["id"] == writer.ID and result["status"] == result["stage"]["status"]
             and result["automatic_retry"] is False and _same(result["qualification"], plan["qualification"])
             and all(type(result[k]) is int and result[k] == 0 for k in ("source_tasks", "actions", "official_scores"))
             and journal[-1]["event"] == "writer_study_stop" and _same(journal[-1]["report"], result))
    release, = result["costs"]["GPU_jobs"]
    job = release["job"]
    ready = _one(journal, "GPU_ready")
    _require(_same(job, queue.make_job(0, job["owner"], writer_plan_sha256))
             and _same(job, _one(journal, "GPU_enqueue_intent")["job"]) and _same(job, ready["job"]))
    durable_release = _one(journal, "GPU_released")
    _require(_same(durable_release["job"], job) and _same(durable_release["release"], release["release"]))
    gpu = audit._release(queue, plan, release, _one(journal, "release_fence_started"))
    backend = ready["service"]["backend"]
    core.episode._backend(plan, backend)
    events = [row for row in journal if "stage" in row or row["event"] in {"candidate_screened", "review_binding"}]
    _require(_same(replay(prepared["cases"], events, backend=backend, model_id=plan["model_id"]), result["stage"]))
    _require(all(_hash(runtime.read(path)) == sha for path, sha in snapshots.items()))
    source_result_sha256 = plan["qualification"]["source_result_sha256"]
    _require(_hash(_read_source_result(lineage)) == source_result_sha256 and _main_pid(launch["unit"]) == 0)
    snapshots[lineage.source_result] = source_result_sha256
    return plan, result["stage"], snapshots, gpu


def report(writer_plan_sha256, writer_result_sha256, *, lineage=LINEAGE):
    """Return authenticated outcomes and cost without model text or task identifiers."""
    plan, result, _snapshots, gpu = _terminal_writer(writer_plan_sha256, writer_result_sha256, lineage)

    def project(value):
        if value is None:
            return None
        return {"status": value["status"], "reason": value["reason"], "cost": value["cost"]}

    return {"id": plan["id"], "plan_sha256": writer_plan_sha256, "result_sha256": writer_result_sha256,
            "source_witness_scenarios": plan["qualification"]["witness_scenarios"],
            "status": result["status"], "writer": project(result["writer"]),
            "review": project(result["review"]), "GPU": gpu, "utility_authorized_by_writer": False}


def prepare_manifest(writer_plan_sha256, writer_result_sha256, *, lineage=LINEAGE):
    """Verify the terminal writer and freeze only an accepted exact insertion."""
    plan, result, snapshots, _gpu = _terminal_writer(writer_plan_sha256, writer_result_sha256, lineage)
    _require(result["status"] == "plausible_pending_utility")
    insertion = stage.content.insertion(result["writer"]["candidate"])
    insertion_sha = _hash(insertion.encode())
    _require(insertion_sha == result["insertion_sha256"])
    manifest = {"id": lineage.manifest_id, "private": True,
                "source_gate_kind": lineage.source_gate_kind,
                "source_plan_sha256": plan["qualification"]["source_plan_sha256"],
                "source_result_sha256": plan["qualification"]["source_result_sha256"],
                "writer_plan_sha256": writer_plan_sha256, "writer_result_sha256": writer_result_sha256,
                "experience_bundle_id": lineage.bundle_prefix + insertion_sha, "insertion_sha256": insertion_sha,
                "source_scenario_ids": sorted(task[:-2] for task in plan["task_ids"]),
                "private_files_sha256": {str(path): digest for path, digest in snapshots.items()}}
    return {"manifest": manifest, "insertion_text": insertion}


def load_reviewed_bundle(writer_plan_sha256, writer_result_sha256, review_manifest_sha256, *, lineage=LINEAGE):
    """Bind the stored manifest and return a private bundle pending utility."""
    _require(_sha(review_manifest_sha256))
    prepared = prepare_manifest(writer_plan_sha256, writer_result_sha256, lineage=lineage)
    manifest = prepared["manifest"]
    writer = lineage.writer
    reg, _queue, runner = writer.bindings()
    runtime = writer.core_for(runner)._Runtime(WORKSPACE)
    path = Path(reg.PRIVATE_RUN_ROOT) / MANIFEST
    raw = runtime.read(path)
    _require(_hash(raw) == review_manifest_sha256 and _same(_decode(raw), manifest))
    files = {**manifest["private_files_sha256"], str(path): review_manifest_sha256}
    _require(all(_hash(_read_source_result(lineage) if Path(path) == lineage.source_result else runtime.read(path)) == sha
                 for path, sha in files.items()))
    return {"private": True, "status": "plausible_pending_utility", "review_manifest_sha256": review_manifest_sha256,
            "insertion_text": prepared["insertion_text"], "private_files_sha256": files,
            **{key: manifest[key] for key in ("source_gate_kind", "source_plan_sha256", "source_result_sha256",
                "writer_plan_sha256", "writer_result_sha256", "experience_bundle_id", "source_scenario_ids")}}
