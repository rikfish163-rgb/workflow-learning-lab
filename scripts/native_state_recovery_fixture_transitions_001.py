"""Extract ordered, anonymous transition evidence from the closed fixture cohort."""

from pathlib import Path

from scripts import native_state_recovery_evidence_001 as witness
from scripts import report_native_state_recovery_fixture_001 as audit
from scripts import run_native_state_recovery_fixture_001 as source
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _same
from wll.protocol import fingerprint

_FAMILIES = {"simple_note__update_note": "A", "todoist__update_task": "B"}


def project(records, checked, fault):
    """Project a checked trace into an ordered sequence with no private values."""
    source._require(checked["status"] == "qualified" and _same(witness.classify(records, fault), checked))
    indices = checked["private_evidence"]["indices"]
    before, refused, retried, after = (indices[key] for key in ("before", "fault", "correction", "after"))
    source._require(before < refused < retried < after)
    decoded = witness._decode_records(records)
    action, _observation = decoded[refused]
    read_name, identifier, field = witness._target_write(action)
    arguments = action["arguments"]
    initial_status, initial = witness._read_value(decoded[before][1], identifier, field, arguments)
    final_status, final = witness._read_value(decoded[after][1], identifier, field, arguments)
    source._require(initial_status == final_status == "known" and final is arguments[field]
                    and initial is not final and decoded[before][0]["name"] == read_name
                    and decoded[after][0]["name"] == read_name)
    return {"operation_family": _FAMILIES[action["name"]], "resource": "target",
            "events": [
                {"event": "read", "observed_value": initial},
                {"event": "write", "requested_value": final, "outcome": "explicitly_not_executed"},
                {"event": "write", "requested_value": final, "same_call": True, "outcome": "acknowledged"},
                {"event": "read", "observed_value": final},
            ]}


def prepare(plan_sha, result_sha):
    """Authenticate the existing cohort and build a new evidence representation."""
    old = audit.prepare(plan_sha, result_sha)
    reg, queue, episode, _runner = source.bindings()
    plan = reg.load_registration(WORKSPACE, plan_sha, private=False)
    core = episode.bind(reg, queue).core
    private = Path(reg.PRIVATE_RUN_ROOT)

    def read(path):
        return core._read_private(path, maximum=episode.MAX_DOCUMENT)

    raw = read(private / "study-result.json")
    source._require(source._hash(raw) == result_sha)
    jobs = {row["job"]["batch_index"]: row["job"] for row in _decode(raw)["costs"]["GPU_jobs"]}
    cases = []
    for row in old["private_provenance"]:
        index = row["source_index"]
        path = private / f"unit-{index:02d}-first_attempt-{jobs[index]['owner']}"
        stored = read(path / episode.EVIDENCE_NAME)
        source._require(source._hash(stored) == row["witness_sha256"])
        checked = _decode(stored)
        receipt_path, controller_path = path / "execution-receipt.json", path / "controller.jsonl"
        receipt_raw, controller_raw = read(receipt_path), read(controller_path)
        hashes = checked["private_files_sha256"]
        source._require(source._hash(receipt_raw) == hashes[str(receipt_path)]
                        and source._hash(controller_raw) == hashes[str(controller_path)]
                        and checked["fixture"]["fixture_target_verified"] is True)
        context = audit.fixture._case_context(plan["design"]["rows"][index], audit.protocol.CASES[index])
        _metadata, records, _generations = episode._actor_evidence(
            _decode(receipt_raw), episode._journal(controller_raw), context,
        )
        source._require(fingerprint(records) == checked["records_sha256"])
        cases.append(project(records, checked["witness"], checked["fault_provenance"]))
    source._require(len(cases) == len(old["cases"])
                    and {case["operation_family"] for case in cases} == {"A", "B"}
                    and {case["events"][0]["observed_value"] for case in cases} == {True, False}
                    and read(private / "study-result.json") == raw)
    return {"cases": cases, "private_provenance": old["private_provenance"],
            "private_source_result_sha256": result_sha}
