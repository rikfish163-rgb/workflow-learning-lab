"""Select supervised next-token logits while retaining the complete input history.

Optional torch dependency. Qwen3's native loss consumes explicit shift_labels;
full labels stay in the batch so Trainer keeps its original accumulation count.
"""


def target_logit_inputs(labels):
    import torch

    if (labels.dtype != torch.long or labels.ndim != 2 or labels.shape[1] < 2
            or torch.any(labels[:, 0] != -100)
            or torch.any((labels < 0) & (labels != -100))):
        raise ValueError("expected batched token labels with masked first token")
    shifted = labels[:, 1:]
    positions = torch.nonzero((shifted != -100).any(dim=0), as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("no supervised next-token positions")
    return {"logits_to_keep": positions,
            "shift_labels": shifted.index_select(1, positions).contiguous()}
