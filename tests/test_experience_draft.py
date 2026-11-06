import pytest
from test_protocol import episode

from wll.experience import ExperienceDraftError, draft_experience
from wll.protocol import Step, fingerprint


class Model:
    model_id = "test-double"

    def __init__(self, text):
        self.text = text

    def generate(self, *args, **kwargs):
        return {"text": self.text, "prompt_tokens": 42, "completion_tokens": 17}


def test_malformed_draft_keeps_consumed_tokens():
    with pytest.raises(ExperienceDraftError) as error:
        draft_experience(episode(success=False), episode(branch="recovery"), Model("not json"))
    assert error.value.cost["completion_tokens"] == 17
    assert error.value.cost["prompt_tokens"] == 42


def test_single_json_fence_is_supported_without_extra_prose():
    raw = '```json\n{"failure_mode":"paging","applies_when":"paginated",'
    raw += '"avoid_when":"single page","rule":"read all pages"}\n```'
    experience, cost = draft_experience(episode(success=False), episode(branch="recovery"), Model(raw))
    assert experience.rule == "read all pages"
    assert cost["completion_tokens"] == 17


def test_nonschema_json_is_a_costed_failure():
    with pytest.raises(ExperienceDraftError):
        draft_experience(episode(success=False), episode(branch="recovery"), Model('["wrong shape"]'))


def test_observed_correction_requires_exact_parent_provenance():
    prefix = [Step("wrong()", "Execution failed: wrong API")]
    source = episode(steps=prefix+[Step("correct()", "ok")])
    probe = episode(success=False, branch="prefix_probe", steps=prefix)
    probe.replayed_steps = len(prefix)
    with pytest.raises(ValueError, match="provenance"):
        draft_experience(probe, source, Model("{}"))
    probe.metadata["parent_episode_hash"] = fingerprint(source.to_dict())
    raw = '{"failure_mode":"paging","applies_when":"paginated",'
    raw += '"avoid_when":"single page","rule":"read all pages"}'
    _, cost = draft_experience(probe, source, Model(raw))
    assert cost["evidence_kind"] == "observed_self_correction"
