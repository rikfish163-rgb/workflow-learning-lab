"""Experimental text-only Qwen3.5-family prefill with last-axis position slicing.

The installed Transformers 5.17 path slices [4, batch, sequence] position IDs on
the batch axis. Keep this adapter local to a separately registered inference
model; it changes no package file, historical server or default loading policy.
"""
from types import MethodType


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_prefill(self, input_ids, generation_config, model_kwargs, is_first_iteration=True):
    chunk_size = generation_config.prefill_chunk_size
    if chunk_size is None:
        return self._wll_original_prefill(input_ids, generation_config, model_kwargs,
                                          is_first_iteration=is_first_iteration)
    require(type(chunk_size) is int and chunk_size > 0, 'positive integer prefill chunk required')
    require(is_first_iteration and input_ids.ndim == 2 and input_ids.shape[0] == 1
            and input_ids.shape[1] > 0, 'fresh batch-one text generation required')
    require(generation_config.num_beams == generation_config.num_return_sequences == 1
            and generation_config.disable_compile is True, 'uncompiled single-sequence decoding required')
    require(all(model_kwargs.get(k) is None for k in ('inputs_embeds', 'pixel_values',
            'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'mm_token_type_ids')),
            'multimodal or embedded inputs are unsupported')
    for name in ('image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'):
        value = getattr(self.config, name, None)
        require(value is None or not bool((input_ids == value).any()), 'vision marker in text input')
    cache = model_kwargs.get('past_key_values')
    require(cache is not None and cache.get_seq_length() == 0, 'fresh cache required')
    attention = model_kwargs.get('attention_mask')
    positions = model_kwargs.get('position_ids')
    require(attention is not None and attention.ndim == 2
            and attention.shape == input_ids.shape and bool((attention == 1).all()),
            'unpadded text attention mask required')
    require(positions is not None and positions.ndim == 3
            and positions.shape == (4, 1, input_ids.shape[1]), 'four-axis Qwen position IDs required')
    for start in range(0, input_ids.shape[1], chunk_size):
        end = min(start+chunk_size, input_ids.shape[1])
        current = {**model_kwargs, 'attention_mask': attention[:, :end],
                   'position_ids': positions[..., start:end]}
        prepared = self.prepare_inputs_for_generation(input_ids[:, start:end], **current)
        outputs = self(**prepared, return_dict=True)
        model_kwargs['past_key_values'] = outputs.past_key_values
    return outputs


def install_text_chunked_prefill(model):
    require(getattr(model.config, 'model_type', None) == 'qwen3_5', 'Qwen3.5-family model required')
    require(not hasattr(model, '_wll_original_prefill'), 'chunked prefill adapter already installed')
    model._wll_original_prefill = model._prefill
    model._prefill = MethodType(text_prefill, model)
