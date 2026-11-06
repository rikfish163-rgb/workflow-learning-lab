"""Train an explicit schedule of full-history, final-assistant SFT microbatches.

This component selects no records, loads no model and saves no adapter. A GPU
caller must separately verify corpus provenance, register its schedule, pass
memory/numerical gates and acquire the project's single-device admission.
"""
import math


def _pooled_target_loss(logits, shift_labels, num_items_in_batch):
    import torch

    losses = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), shift_labels.reshape(-1),
        ignore_index=-100, reduction='none')
    # FP32 CE sum reduction changes with ignored-history rows even when all
    # supervised logits and gradients are identical. Accumulate token scalars
    # in FP64 so selection/padding cannot change this pooled-token objective.
    return losses.sum(dtype=torch.float64) / num_items_in_batch


def validate_groups(groups, *, max_length):
    if type(max_length) is not int or max_length < 2:
        raise ValueError('positive sample length limit required')
    if not isinstance(groups, list) or not groups:
        raise ValueError('explicit nonempty update schedule required')
    totals = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError('each update needs at least one microbatch')
        total = 0
        for row in group:
            if not isinstance(row, dict) or set(row) != {'input_ids', 'attention_mask', 'labels'}:
                raise ValueError('complete encoded input, mask and labels required')
            ids, mask, labels = row['input_ids'], row['attention_mask'], row['labels']
            if (not all(isinstance(v, list) for v in (ids, mask, labels))
                    or not 2 <= len(ids) <= max_length or len(ids) != len(mask) or len(ids) != len(labels)):
                raise ValueError('invalid sample length; no padding or truncation')
            if (any(type(t) is not int or t < 0 for t in ids)
                    or any(type(t) is not int or t != 1 for t in mask)
                    or any(type(t) is not int for t in labels) or labels[0] != -100):
                raise ValueError('unmasked history, invalid tokens or padded microbatch')
            positions = [i for i, label in enumerate(labels) if label != -100]
            if (not positions or positions != list(range(positions[0], len(ids)))
                    or any(labels[i] != ids[i] for i in positions)):
                raise ValueError('one complete final assistant target required')
            total += len(positions)
        totals.append(total)
    return totals


def train_groups(model, groups, *, max_length, progress, learning_rate=1e-4, max_grad_norm=1.0):
    totals = validate_groups(groups, max_length=max_length)
    if (not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not isinstance(max_grad_norm, (int, float))
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError('finite positive optimizer settings required')
    if (not hasattr(model, 'peft_config')
            or type(model.get_base_model()).__name__ != 'Qwen3_5ForConditionalGeneration'):
        raise ValueError('prepared hybrid text LoRA model required')
    import torch

    from scripts.qwen38_training import target_batch
    from scripts.training_numerics import tensor_digest

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    devices = {str(p.device) for p in params.values()}
    if (not params or any('lora_' not in n or 'visual' in n for n in params)
            or len(devices) != 1 or not devices <= {'cpu', 'cuda:0'}
            or any(p.dtype != torch.float32 or p.grad is not None for p in params.values())):
        raise ValueError('clean FP32 text adapters on one execution device required')
    vocab = model.get_input_embeddings().num_embeddings
    if any(token >= vocab for group in groups for row in group for token in row['input_ids']):
        raise ValueError('input token exceeds model vocabulary')
    device = next(iter(params.values())).device
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    initial_sha = tensor_digest(initial)
    optimizer = torch.optim.AdamW(params.values(), lr=learning_rate, weight_decay=0, foreach=False)
    model.train()
    losses, norms = [], []
    try:
        for index, (group, total) in enumerate(zip(groups, totals, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for row_index, row in enumerate(group):
                detail = {'optimizer_step': index, 'microbatch': row_index, 'target_tokens': total}
                batch = target_batch({k: torch.tensor([v], dtype=torch.long, device=device)
                                      for k, v in row.items()})
                # Keep native forwarding of the complete update's target count,
                # including a final short group.
                batch['num_items_in_batch'] = total
                progress('forward', detail)
                output = model(**batch)
                targets = sum(v != -100 for v in row['labels'][1:])
                if tuple(output.logits.shape[:2]) != (1, targets) or not bool(torch.isfinite(output.loss)):
                    raise ValueError('invalid target projection or nonfinite loss')
                logits = output.logits
                # Release the native CE graph before computing the stable scalar
                # reduction; retain only its target logits and decoder graph.
                del output
                loss = _pooled_target_loss(logits, batch['shift_labels'], total)
                value = float(loss.detach())
                progress('backward', {**detail, 'normalized_loss': value})
                loss.backward()
                group_loss += value
                del logits, loss, batch
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params.values())
                    or any(p.grad is not None for p in model.parameters() if not p.requires_grad)):
                raise ValueError('missing/nonfinite adapter gradient or frozen base gradient')
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), max_grad_norm,
                                                 error_if_nonfinite=True)
            progress('update', {'optimizer_step': index, 'target_tokens': total,
                                'examples': len(group), 'normalized_loss': group_loss})
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params.values()):
                raise ValueError('nonfinite updated adapter')
            losses.append(group_loss)
            norms.append(float(norm))
    finally:
        optimizer.zero_grad(set_to_none=True)
    changed = [n for n, p in params.items() if not torch.equal(initial[n], p.detach().cpu())]
    kinds = sorted({kind for n in changed for kind in ['linear_attn', 'self_attn'] if kind in n})
    if not changed or kinds != ['linear_attn', 'self_attn']:
        raise ValueError('both text attention families must update')
    return {'optimizer_steps': len(groups), 'supervised_tokens': sum(totals),
            'target_tokens_per_update': totals, 'examples_per_update': [len(g) for g in groups],
            'losses': losses, 'gradient_norms_before_clipping': norms,
            'changed_adapter_tensors': len(changed), 'changed_attention_types': kinds,
            'initial_adapter_sha256': initial_sha, 'final_adapter_sha256': tensor_digest(params),
            'finite_parameters': True, 'finite_gradients': True, 'adapter_saved': False}
