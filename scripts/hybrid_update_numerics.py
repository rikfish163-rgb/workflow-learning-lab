"""Same-state native/repeat/actual-SFT comparison for an explicit encoded schedule.

Caller owns model loading, placement, provenance and GPU admission. Every arm
uses the same streamed-delta and activation-offload contexts. This is an
isolated, single-threaded diagnostic; no model or adapter is loaded or saved.
"""
import copy
import hashlib
import json
import math
from contextlib import contextmanager
from threading import Lock

from scripts.activation_offload import ActivationOffload
from scripts.flat_training_numerics import streamed_delta_context
from scripts.hybrid_target_updates import train_groups, validate_groups
from scripts.probe_allocator import memory
from scripts.qwen38_training import target_batch
from scripts.training_numerics import tensor_digest, tensor_metrics

ARMS = ['reference', 'repeat_reference', 'candidate']
TOLERANCE = {'loss_abs': 1e-5, 'gradient_global_relative_l2': 1e-4,
             'gradient_max_abs': 1e-3, 'parameter_max_abs': 5e-6}
_ACTIVE = Lock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshot(values):
    return {name: value.detach().cpu().clone() for name, value in values.items()}


@contextmanager
def _capture_clipping(torch, params, pending):
    original = torch.nn.utils.clip_grad_norm_

    def capture(*args, **kwargs):
        _require(torch.nn.utils.clip_grad_norm_ is capture, 'native clipping function changed')
        _require(args and isinstance(args[0], (tuple, list))
                 and [id(p) for p in args[0]] == [id(p) for p in params.values()],
                 'clipping must cover exactly the diagnostic adapters')
        _require(len(args) == 2 and args[1] == 1.0 and kwargs == {'error_if_nonfinite': True},
                 'fixed native clipping settings required')
        _require(not pending, 'multiple clips before one optimizer update')
        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()),
                 'missing/nonfinite aggregate adapter gradient')
        gradients = _snapshot({name: p.grad for name, p in params.items()})
        # Delegate the original arguments unchanged. These are aggregate .grad
        # tensors after all microbatches, before the native clip mutates them.
        norm = original(*args, **kwargs)
        pending.update(gradients=gradients, norm=float(norm))
        return norm

    _require(torch.nn.utils.clip_grad_norm_ is original, 'native clipping identity changed')
    torch.nn.utils.clip_grad_norm_ = capture
    try:
        yield
        _require(torch.nn.utils.clip_grad_norm_ is capture, 'native clipping function replaced during arm')
    finally:
        torch.nn.utils.clip_grad_norm_ = original
        _require(torch.nn.utils.clip_grad_norm_ is original, 'native clipping function was not restored')


def _native_groups(model, groups, totals, params, initial, progress):
    """Independent native selected-target CE; no stable-loss helper is reused."""
    import torch

    optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0, foreach=False)
    device = next(iter(params.values())).device
    model.train()
    losses, norms = [], []
    try:
        for index, (group, total) in enumerate(zip(groups, totals, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for row_index, row in enumerate(group):
                detail = {'optimizer_step': index, 'microbatch': row_index, 'target_tokens': total}
                batch = target_batch({key: torch.tensor([value], dtype=torch.long, device=device)
                                      for key, value in row.items()})
                batch['num_items_in_batch'] = total
                progress('forward', detail)
                output = model(**batch)
                _require(tuple(output.logits.shape[:2]) == (1, len(batch['logits_to_keep']))
                         and bool(torch.isfinite(output.loss)), 'invalid native target projection/loss')
                value = float(output.loss.detach())
                progress('backward', {**detail, 'normalized_loss': value})
                output.loss.backward()
                group_loss += value
                del output, batch
            _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values())
                     and all(p.grad is None for p in model.parameters() if not p.requires_grad),
                     'invalid native adapter/frozen gradient')
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
            progress('update', {'optimizer_step': index, 'target_tokens': total,
                                'examples': len(group), 'normalized_loss': group_loss})
            optimizer.step()
            _require(all(bool(torch.isfinite(p).all()) for p in params.values()), 'nonfinite native update')
            losses.append(group_loss)
            norms.append(float(norm))
    finally:
        optimizer.zero_grad(set_to_none=True)
    changed = [name for name, parameter in params.items()
               if not torch.equal(initial[name], parameter.detach().cpu())]
    kinds = sorted({kind for name in changed for kind in ['linear_attn', 'self_attn'] if kind in name})
    _require(changed and kinds == ['linear_attn', 'self_attn'], 'both native text attention families must update')
    return {'optimizer_steps': len(groups), 'supervised_tokens': sum(totals),
            'target_tokens_per_update': totals, 'examples_per_update': [len(g) for g in groups],
            'losses': losses, 'gradient_norms_before_clipping': norms,
            'changed_adapter_tensors': len(changed), 'changed_attention_types': kinds,
            'initial_adapter_sha256': tensor_digest(initial), 'final_adapter_sha256': tensor_digest(params),
            'finite_parameters': True, 'finite_gradients': True, 'adapter_saved': False}


def compare_hybrid_updates(model, groups, progress, *, max_length, window_tokens=512):
    """Measure all three arms and restore adapters, buffers, RNG and temporary patches.

    Does not enforce tolerances: persist the returned evidence, then call
    assess_hybrid_comparison before any continuation. Input tokens are hashed,
    never returned; the schedule contains lengths and target counts only.
    """
    totals = validate_groups(groups, max_length=max_length)
    _require(callable(progress) and type(window_tokens) is int and window_tokens > 0
             and window_tokens % 64 == 0, 'callback and positive 64-token window required')
    _require(hasattr(model, 'peft_config')
             and type(model.get_base_model()).__name__ == 'Qwen3_5ForConditionalGeneration'
             and model.training, 'prepared training hybrid text LoRA required')
    import torch

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    _require(params and all('lora_' in name and 'visual' not in name for name in params)
             and len({str(p.device) for p in params.values()}) == 1
             and all(str(p.device) in ['cpu', 'cuda:0'] and p.dtype == torch.float32
                     and p.grad is None for p in params.values())
             and all(p.grad is None for p in model.parameters()), 'clean FP32 text adapters required')
    _require(all(token < model.get_input_embeddings().num_embeddings
                 for group in groups for row in group for token in row['input_ids']),
             'input token exceeds model vocabulary')
    _require(_ACTIVE.acquire(blocking=False), 'hybrid comparison is already active')
    try:
        return _compare(model, groups, totals, params, progress, max_length, window_tokens, torch)
    finally:
        _ACTIVE.release()


def _compare(model, groups, totals, params, progress, max_length, window_tokens, torch):
    device = next(iter(params.values())).device
    initial = _snapshot(params)
    buffer_objects = dict(model.named_buffers())
    buffers = _snapshot(buffer_objects)
    slots = [(module, dict(module._buffers), set(module._non_persistent_buffers_set), module.training)
             for module in model.modules()]
    frozen = [(p, p.data_ptr(), p.dtype, p.device, p._version)
              for p in model.parameters() if not p.requires_grad]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None

    def rng_values():
        return {'cpu': torch.get_rng_state(), **({'cuda': torch.cuda.get_rng_state(device)}
                                                if cuda_rng is not None else {})}

    adapter_sha, buffer_sha, rng_sha = tensor_digest(initial), tensor_digest(buffers), tensor_digest(rng_values())

    def restore():
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(initial[name])
            for module, original_buffers, persistence, training in slots:
                module._buffers.clear()
                module._buffers.update(original_buffers)
                module._non_persistent_buffers_set = set(persistence)
                module.training = training
            for name, value in buffers.items():
                buffer_objects[name].copy_(value)
            for parameter in model.parameters():
                parameter.grad = None
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

    outcomes, comparisons = [], {}
    reference_gradients, reference_parameters = None, None
    result = None
    try:
        for arm in ARMS:
            restore()
            _require(tensor_digest(params) == adapter_sha and tensor_digest(dict(model.named_buffers())) == buffer_sha
                     and tensor_digest(rng_values()) == rng_sha, 'arm initial state differs')
            pending, gradients, parameter_hashes, update_losses, update_norms = {}, [], [], [], []

            def measured(phase, detail, *, arm=arm, pending=pending, gradients=gradients,
                         parameter_hashes=parameter_hashes, update_losses=update_losses,
                         update_norms=update_norms):
                index = detail['optimizer_step']
                if phase == 'forward' and detail['microbatch'] == 0:
                    _require(index == len(gradients), 'forward update order differs')
                    if index:
                        parameter_hashes.append(tensor_digest(params))
                if phase == 'update':
                    _require(index == len(gradients) and pending
                             and detail['target_tokens'] == totals[index]
                             and detail['examples'] == len(groups[index]), 'missing aggregate update capture/counts')
                    gradients.append(pending.pop('gradients'))
                    update_norms.append(pending.pop('norm'))
                    update_losses.append(detail['normalized_loss'])
                progress(arm, phase, dict(detail))

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            offload = ActivationOffload(model)
            with _capture_clipping(torch, params, pending), \
                    streamed_delta_context(model, window_tokens=window_tokens) as streaming, offload:
                if arm == 'candidate':
                    outcome = train_groups(model, groups, max_length=max_length, progress=measured)
                else:
                    outcome = _native_groups(model, groups, totals, params, initial, measured)
            _require(not pending and len(gradients) == len(groups)
                     and len(parameter_hashes) == len(groups) - 1
                     and outcome['losses'] == update_losses
                     and outcome['gradient_norms_before_clipping'] == update_norms,
                     'incomplete or inconsistent per-update evidence')
            _require(all(p.data_ptr() == pointer and p.dtype == dtype and p.device == placement
                         and p._version == version and p.grad is None and not p.requires_grad
                         for p, pointer, dtype, placement, version in frozen),
                     'frozen storage/dtype/version/gradient changed')
            _require(dict(model.named_buffers()).keys() == buffer_objects.keys()
                     and all(dict(model.named_buffers())[name] is value for name, value in buffer_objects.items()),
                     'buffer identity/topology changed')
            final = _snapshot(params)
            final_sha = tensor_digest(final)
            parameter_hashes.append(final_sha)
            _require(outcome['final_adapter_sha256'] == final_sha, 'final adapter hash differs')
            outcome.update(arm=arm, initial_adapter_sha256=adapter_sha, initial_buffer_sha256=buffer_sha,
                initial_rng_sha256=rng_sha, final_buffer_sha256=tensor_digest(dict(model.named_buffers())),
                final_rng_sha256=tensor_digest(rng_values()), final_adapter_sha256=final_sha,
                gradient_capture='aggregate_preclip_per_optimizer_update',
                clip_calls_per_update=[1] * len(groups),
                gradient_sha256_by_step=[tensor_digest(values) for values in gradients],
                parameter_sha256_by_step=parameter_hashes,
                objective='fp32_token_ce_fp64_sum' if arm == 'candidate' else 'native_selected_target_ce',
                memory=memory(torch) if device.type == 'cuda' else None,
                offload=copy.deepcopy(offload.stats), streaming=copy.deepcopy(streaming))
            if arm == 'reference':
                reference_gradients, reference_parameters = gradients, final
            else:
                comparisons[arm] = {
                    'losses_bitwise_equal': outcomes[0]['losses'] == outcome['losses'],
                    'loss_abs_errors': [abs(a - b) for a, b in zip(outcomes[0]['losses'], outcome['losses'], strict=True)],
                    'gradients': [tensor_metrics(a, b) for a, b in zip(reference_gradients, gradients, strict=True)],
                    'parameters': tensor_metrics(reference_parameters, final)}
            outcomes.append(outcome)
        result = {'arms': outcomes, 'comparisons': comparisons, 'max_length': max_length,
                  'window_tokens': window_tokens, 'trainable_adapter_tensors': len(params),
                  'input_sha256': hashlib.sha256(json.dumps(groups, sort_keys=True,
                                                          separators=(',', ':')).encode()).hexdigest(),
                  'schedule': [[{'input_tokens': len(row['input_ids']),
                                 'target_tokens': sum(value != -100 for value in row['labels'][1:])}
                                for row in group] for group in groups], 'tolerance': dict(TOLERANCE)}
    finally:
        restore()
    _require(tensor_digest(params) == adapter_sha and tensor_digest(dict(model.named_buffers())) == buffer_sha
             and tensor_digest(rng_values()) == rng_sha and all(p.grad is None for p in model.parameters()),
             'final state restoration differs')
    result['state_restored'] = True
    return result


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _metric(metric, count, same):
    _require(isinstance(metric, dict) and metric.get('tensor_count') == count
             and metric.get('bitwise_equal') is same
             and all(_finite(metric.get(key)) for key in ['max_abs_error', 'global_relative_l2_error',
                                                          'reference_l2_norm', 'difference_l2_norm']),
             'invalid numerical metric or hash equality disagreement')
    rows = metric.get('largest_differences')
    _require(isinstance(rows, list) and len(rows) == min(5, count)
             and all(isinstance(row, dict) and isinstance(row.get('name'), str)
                     and type(row.get('bitwise_equal')) is bool and _finite(row.get('max_abs_error'))
                     for row in rows)
             and len({row['name'] for row in rows}) == len(rows)
             and max(row['max_abs_error'] for row in rows) == metric['max_abs_error']
             and metric['max_abs_error'] <= metric['difference_l2_norm'], 'inconsistent metric detail/norm')
    expected_relative = (metric['difference_l2_norm'] / metric['reference_l2_norm']
                         if metric['reference_l2_norm'] else 0.0)
    _require((metric['reference_l2_norm'] > 0 or metric['difference_l2_norm'] == 0)
             and math.isclose(metric['global_relative_l2_error'], expected_relative, rel_tol=1e-12, abs_tol=0),
             'inconsistent relative gradient/parameter norm')
    if same:
        _require(metric['max_abs_error'] == metric['global_relative_l2_error']
                 == metric['difference_l2_norm'] == 0, 'bitwise equality has nonzero error')


def assess_hybrid_comparison(result):
    """Reject incomplete evidence, non-repeatable references or fixed-gate violations."""
    _require(isinstance(result, dict) and result.get('state_restored') is True
             and result.get('tolerance') == TOLERANCE and _hash(result.get('input_sha256')),
             'incomplete/restoration/tolerance evidence')
    arms, comparisons = result.get('arms'), result.get('comparisons')
    _require(isinstance(arms, list) and [arm.get('arm') for arm in arms] == ARMS
             and isinstance(comparisons, dict) and set(comparisons) == {'repeat_reference', 'candidate'},
             'incomplete hybrid arm comparison')
    schedule, count = result.get('schedule'), result.get('trainable_adapter_tensors')
    _require(type(count) is int and count > 0 and isinstance(schedule, list) and schedule
             and type(result.get('max_length')) is int and result['max_length'] >= 2
             and type(result.get('window_tokens')) is int and result['window_tokens'] > 0
             and result['window_tokens'] % 64 == 0, 'invalid schedule/configuration evidence')
    totals, examples = [], []
    for group in schedule:
        _require(isinstance(group, list) and group, 'empty schedule group')
        for row in group:
            _require(isinstance(row, dict) and set(row) == {'input_tokens', 'target_tokens'}
                     and type(row['input_tokens']) is type(row['target_tokens']) is int
                     and 1 <= row['target_tokens'] < row['input_tokens'] <= result['max_length'],
                     'invalid schedule summary')
        totals.append(sum(row['target_tokens'] for row in group))
        examples.append(len(group))
    steps = len(schedule)
    for key in ['initial_adapter_sha256', 'initial_buffer_sha256', 'initial_rng_sha256']:
        _require(all(_hash(arm.get(key)) for arm in arms) and len({arm[key] for arm in arms}) == 1,
                 'different initial adapter/buffer/RNG state')
    for index, arm in enumerate(arms):
        _require(arm.get('optimizer_steps') == steps and arm.get('supervised_tokens') == sum(totals)
                 and arm.get('target_tokens_per_update') == totals and arm.get('examples_per_update') == examples
                 and arm.get('clip_calls_per_update') == [1] * steps
                 and arm.get('gradient_capture') == 'aggregate_preclip_per_optimizer_update'
                 and arm.get('finite_parameters') is arm.get('finite_gradients') is True
                 and arm.get('adapter_saved') is False
                 and type(arm.get('changed_adapter_tensors')) is int
                 and 0 < arm['changed_adapter_tensors'] <= count
                 and arm.get('changed_attention_types') == ['linear_attn', 'self_attn']
                 and arm.get('objective') == ('fp32_token_ce_fp64_sum' if index == 2 else 'native_selected_target_ce'),
                 'invalid arm update/count/objective evidence')
        for field in ['losses', 'gradient_norms_before_clipping']:
            _require(isinstance(arm.get(field), list) and len(arm[field]) == steps
                     and all(_finite(value) for value in arm[field]), 'invalid finite per-update values')
        for field in ['gradient_sha256_by_step', 'parameter_sha256_by_step']:
            _require(isinstance(arm.get(field), list) and len(arm[field]) == steps
                     and all(_hash(value) for value in arm[field]), 'incomplete per-update hashes')
        _require(all(_hash(arm.get(field)) for field in ['final_adapter_sha256', 'final_buffer_sha256', 'final_rng_sha256'])
                 and arm['parameter_sha256_by_step'][-1] == arm['final_adapter_sha256'], 'invalid final hashes')
        _require('memory' in arm and isinstance(arm.get('offload'), dict)
                 and type(arm['offload'].get('copied_bytes')) is int and arm['offload']['copied_bytes'] > 0
                 and isinstance(arm.get('streaming'), dict)
                 and arm['streaming'].get('window_tokens') == result['window_tokens'],
                 'missing common offload/streaming evidence')
    for index, name in enumerate(['repeat_reference', 'candidate'], 1):
        arm, comp = arms[index], comparisons[name]
        _require(isinstance(comp, dict) and comp.get('losses_bitwise_equal') is (arms[0]['losses'] == arm['losses'])
                 and comp.get('loss_abs_errors') == [abs(a - b) for a, b in zip(arms[0]['losses'], arm['losses'], strict=True)]
                 and isinstance(comp.get('gradients'), list) and len(comp['gradients']) == steps,
                 'loss/gradient evidence disagrees')
        _require(all(_finite(value) for value in comp['loss_abs_errors']), 'nonfinite loss error')
        _metric(comp.get('parameters'), count, arms[0]['final_adapter_sha256'] == arm['final_adapter_sha256'])
        for step, gradient in enumerate(comp['gradients']):
            _metric(gradient, count, arms[0]['gradient_sha256_by_step'][step] == arm['gradient_sha256_by_step'][step])
        _require(max(comp['loss_abs_errors']) <= TOLERANCE['loss_abs']
                 and comp['parameters']['max_abs_error'] <= TOLERANCE['parameter_max_abs']
                 and all(g['max_abs_error'] <= TOLERANCE['gradient_max_abs']
                         and g['global_relative_l2_error'] <= TOLERANCE['gradient_global_relative_l2']
                         for g in comp['gradients']), 'numerical difference exceeds fixed tolerance')
    _require(all(arms[0][field] == arms[1][field] for field in [
        'losses', 'gradient_norms_before_clipping', 'gradient_sha256_by_step', 'parameter_sha256_by_step',
        'final_adapter_sha256', 'final_buffer_sha256', 'final_rng_sha256']), 'native reference is not bitwise repeatable')
    return {'within_registered_tolerance': True, 'repeat_reference_bitwise_equal': True,
            'candidate_bitwise_equal': comparisons['candidate']['losses_bitwise_equal']
                and comparisons['candidate']['parameters']['bitwise_equal']
                and all(metric['bitwise_equal'] for metric in comparisons['candidate']['gradients']),
            'tolerance': dict(TOLERANCE), 'task_learning_demonstrated': False}
