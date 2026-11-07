"""Explicit no-thinking SFT labels, verified against the serving token prefix.

Independent from the frozen historical encoder. This module does not collect,
select or train examples; callers must first validate their corpus provenance.
"""


def encode_target(tokenizer, messages, max_length):
    if (not messages or messages[-1]['role'] != 'assistant'
            or not isinstance(messages[-1]['content'], str)
            or not messages[-1]['content'].strip()):
        raise ValueError('last message must be a nonempty assistant target')
    if type(max_length) is not int or max_length <= 0:
        raise ValueError('positive integer maximum length required')
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, return_dict=False, enable_thinking=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=False, return_dict=False, enable_thinking=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError('chat template prefix mismatch; cannot safely infer label boundary')
    # Match ChatEngine: render first, then tokenize without adding special tokens.
    served_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    served = tokenizer(served_text, add_special_tokens=False)['input_ids']
    if served != prefix:
        raise ValueError('training and serving prefix tokens differ')
    if len(tokens) > max_length or len(tokens) <= len(prefix):
        raise ValueError('invalid sample length; do not silently truncate')
    return {'input_ids': tokens, 'attention_mask': [1] * len(tokens),
            'labels': [-100] * len(prefix) + tokens[len(prefix):]}


def audit_tokens(rows, tokenizer):
    samples = []
    for row in rows:
        # Audit all supplied rows, including those beyond either training budget.
        encoded = encode_target(tokenizer, row['messages'], 1_000_000)
        total = len(encoded['input_ids'])
        targets = sum(label != -100 for label in encoded['labels'])
        samples.append({'task_id': row['task_id'], 'step_index': row['step_index'],
            'total_tokens': total, 'supervised_tokens': targets,
            'fits_8192': total <= 8192, 'fits_16384': total <= 16384})
    return {'examples': len(samples),
        'all_training_serving_prefixes_identical': True if samples else None,
        'maximum_sample_tokens': max((s['total_tokens'] for s in samples), default=None),
        'supervised_tokens_per_epoch': sum(s['supervised_tokens'] for s in samples),
        'samples_exceeding_8192': sum(not s['fits_8192'] for s in samples),
        'samples_exceeding_16384': sum(not s['fits_16384'] for s in samples),
        'samples': samples,
        'scope': 'Tokenizer-only boundary/length audit. No truncation, sample selection, '
            'model loading, CUDA, training configuration approval or efficacy claim.'}
