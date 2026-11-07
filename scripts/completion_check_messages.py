"""Construct paired inspection prompts from actor-visible history only.

The caller must supply identical trusted tool instructions for both arms and
independently enforce any read-only tool policy. Prompt text is not an access
control, and this helper neither chooses tasks nor reads evaluation results.
"""

from wll.prompts import initial_messages
from wll.protocol import Task

INSPECTION_BOUNDARY = (
    'The immediately preceding assistant response is a pending action and has not been executed. '
    'There is no observation for that pending action. The next response is for inspection only: '
    'do not change state and do not call, request, or submit task completion. '
    'For this inspection response only, output exactly one JSON object and nothing else; '
    'this overrides the existing Python code-output rules. Do not include Markdown fences, '
    'Python comments, or extra text. Use the JSON format and permitted tool instructions below.'
)

ORDINARY_CHECK = (
    'Ordinary check: Review what you consider necessary in the visible task and history. '
    'State what you intend to check in the JSON "check" field. '
    'Use a permitted read-only inspection if needed; otherwise use a permitted no-op.'
)

EVIDENCE_GAP_CHECK = (
    'Evidence-gap check: Identify one task requirement that lacks visible supporting evidence. '
    'State the gap, or that none is apparent, in the JSON "check" field. '
    'If there is a gap, inspect it using a permitted read-only operation; otherwise use a permitted no-op.'
)

_ARMS = {'A': ORDINARY_CHECK, 'B': EVIDENCE_GAP_CHECK}


def build_check_messages(task, prefix_steps, pending_response, *, arm, tool_instructions,
                         prompt_version='appworld-v3'):
    """Return fresh messages; the last assistant action remains unexecuted.

    Each prefix element is an explicit ``action``/``observation`` dict with an
    optional ``raw_response`` only. The pending response is text, not a Step.
    No observation or result is manufactured for that pending response.
    """
    if type(task) is not Task or any(type(getattr(task, name)) is not str
            for name in ('task_id', 'scenario_id', 'split', 'instruction', 'supervisor')):
        raise ValueError('task must be an explicit text-only Task')
    if type(prefix_steps) is not list:
        raise ValueError('prefix must be a list of actor-visible steps')
    required = {'action', 'observation'}
    for step in prefix_steps:
        if (type(step) is not dict or set(step) not in (required, required | {'raw_response'})
                or any(type(value) is not str for value in step.values())):
            raise ValueError('prefix step must contain only actor-visible text fields')
    if type(pending_response) is not str:
        raise ValueError('pending response must be text')
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError('arm must be A or B')
    if type(tool_instructions) is not str or not tool_instructions.strip():
        raise ValueError('tool instructions must be nonempty text')
    if type(prompt_version) is not str:
        raise ValueError('prompt version must be text')

    result = initial_messages(task, prompt_version)
    for step in prefix_steps:
        result.append({'role': 'assistant', 'content': step.get('raw_response') or step['action']})
        result.append({'role': 'user', 'content': step['observation']})
    result.append({'role': 'assistant', 'content': pending_response})
    result.append({'role': 'user', 'content': f'{INSPECTION_BOUNDARY}\n\n'
                   f'Tool instructions:\n{tool_instructions}\n\n{_ARMS[arm]}'})
    return result
