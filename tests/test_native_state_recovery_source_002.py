from enum import StrEnum

from scripts import native_state_recovery_registration_001 as old
from scripts import run_native_state_recovery_source_002 as source


class QuantizationMethod(StrEnum):
    BITS_AND_BYTES = "bitsandbytes"


def test_backend_enum_matches_wire_type_without_dropping_fields():
    expected = {"quantization": {"configuration": {"quant_method": "bitsandbytes", "load_in_4bit": True}},
                "recipe": {"temperature": 0.7}}
    live = {**expected, "quantization": {"configuration": {
        "quant_method": QuantizationMethod.BITS_AND_BYTES, "load_in_4bit": True}}}
    assert not old._same(live, expected)
    assert old._same(source.wire_backend(live), expected)
    assert not old._same(source.wire_backend({**live, "recipe": {"temperature": 0.8}}), expected)
    assert not old._same(source.wire_backend({**live, "extra": True}), expected)


def test_new_identity_keeps_exact_prior_sources_and_fixed_cohort():
    reg, queue, episode, runner = source.bindings()
    assert reg.ID == queue.ID == episode.ID == runner.ID == source.ID != old.ID
    assert reg.PLAN == queue.PLAN == episode.PLAN == source.PLAN != old.PLAN
    assert reg.PRIVATE_RUN_ROOT == source.PRIVATE_RUN_ROOT != old.PRIVATE_RUN_ROOT
    assert reg.TASK_IDS == episode.TASK_IDS == old.TASK_IDS
    assert reg.FIXED_INPUTS[source.PRIOR_PLAN] == source.PRIOR_INPUTS[source.PRIOR_PLAN]
    assert source.PRIOR_TERMINAL not in old.FIXED_INPUTS
    job = queue.make_job(0, "a" * 32, "b" * 64)
    queue._core()._validate_job(job)
    assert job["argv"][3] == source.MODULE
    assert job["max_runtime_seconds"] == 7200
