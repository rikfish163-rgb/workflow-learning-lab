"""Synchronous saved-activation copies; model parameter/buffer storage stays put.

Wrap a forward pass after the model is placed on its final device. The backward
may run after the context exits. Plain non-reentrant checkpointing still
recomputes each block on its original device; this moves the saved boundary
inputs, not all temporary tensors created during that recomputation.

CPU execution clones activations to exercise the same packing/value semantics.
Byte counters count each save/load, not distinct storage or peak resident RAM.
No pinned-memory allocation, CUDA initialization, or asynchronous copy is used.
"""


def storage_key(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


class ActivationOffload:
    def __init__(self, model):
        self.model_storage = {storage_key(t) for t in (*model.parameters(), *model.buffers())
            if t.numel()}
        self.stats = dict(pack_calls=0, unpack_calls=0, copied_bytes=0,
            kept_model_bytes=0, gpu_to_cpu_bytes=0, cpu_to_gpu_bytes=0)
        self.context = None

    def pack(self, tensor):
        self.stats['pack_calls'] += 1
        size = tensor.numel() * tensor.element_size()
        if not size or storage_key(tensor) in self.model_storage:
            self.stats['kept_model_bytes'] += size
            # detach avoids retaining the original autograd graph through a saved output.
            return False, tensor.device, tensor.detach()
        saved = tensor.detach().to('cpu', copy=True, non_blocking=False)
        self.stats['copied_bytes'] += size
        if tensor.device.type == 'cuda':
            self.stats['gpu_to_cpu_bytes'] += size
        return True, tensor.device, saved

    def unpack(self, packed):
        copied, device, tensor = packed
        self.stats['unpack_calls'] += 1
        if copied and device.type == 'cuda':
            self.stats['cpu_to_gpu_bytes'] += tensor.numel() * tensor.element_size()
        return tensor.to(device, non_blocking=False) if copied else tensor

    def __enter__(self):
        import torch

        if self.context is not None:
            raise RuntimeError('offload context is already active')
        self.context = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.context.__exit__(*args)
        finally:
            self.context = None
