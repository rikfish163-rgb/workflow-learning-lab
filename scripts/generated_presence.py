"""Generated-token presence penalty; prompt and occurrence counts are excluded.

Duck-typed Transformers logits processor. Importing this module does not import
torch, so the GPU policy can run before CUDA visibility is cached.
"""
import math


class GeneratedPresencePenalty:
    def __init__(self, penalty, prompt_length):
        if not math.isfinite(penalty) or not -2 <= penalty <= 2:
            raise ValueError('presence penalty must be finite and in [-2, 2]')
        if type(prompt_length) is not int or prompt_length < 0:
            raise ValueError('prompt length must be a nonnegative integer')
        self.penalty = float(penalty)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        import torch

        if (input_ids.ndim != 2 or scores.ndim != 2
                or input_ids.shape[0] != scores.shape[0]
                or input_ids.shape[1] < self.prompt_length):
            raise ValueError('invalid generation shapes or prompt boundary')
        emitted = input_ids[:, self.prompt_length:]
        if not emitted.numel() or self.penalty == 0:
            return scores
        present = torch.zeros_like(scores, dtype=torch.bool)
        present.scatter_(1, emitted, True)
        return scores - self.penalty * present.to(scores.dtype)
