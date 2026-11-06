import pytest

from wll.training import encode_target


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False):
        text = "".join(f"<{m['role']}>" + m["content"] + "</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


def test_only_final_assistant_tokens_are_supervised():
    messages = [{"role": "system", "content": "rule"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "old action"},
                {"role": "user", "content": "tool observation"},
                {"role": "assistant", "content": "new action"}]
    encoded = encode_target(Tokenizer(), messages, 1000)
    targets = bytes(x for x in encoded["labels"] if x != -100).decode()
    assert targets == "new action</end>"
    assert len(encoded["labels"]) == len(encoded["input_ids"])


def test_overlength_samples_are_rejected_without_truncating_target():
    with pytest.raises(ValueError, match="length"):
        encode_target(Tokenizer(), [{"role": "user", "content": "a"},
                                   {"role": "assistant", "content": "b"}], 2)


def test_chat_template_boundary_must_be_exact():
    class Broken(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False):
            return [1,2] if add_generation_prompt else [3,4,5]

    with pytest.raises(ValueError, match="prefix"):
        encode_target(Broken(), [{"role": "user", "content": "a"},
                                {"role": "assistant", "content": "b"}], 100)


def test_modern_tokenizer_mapping_default_does_not_break_label_boundary():
    class Modern(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=True):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return {"input_ids": ids} if return_dict else ids

    result = encode_target(Modern(), [{"role": "user", "content": "task"},
                                     {"role": "assistant", "content": "answer"}], 1000)
    assert bytes(x for x in result["labels"] if x != -100).decode() == "answer</end>"
