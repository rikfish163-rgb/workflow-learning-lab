"""Independent task definitions for an authored AppWorld CPU validation panel."""

from dataclasses import dataclass
from hashlib import sha256

ORIGINS = ("Orchid", "Juniper", "Cobalt")


@dataclass(frozen=True)
class Task:
    scenario: str
    variant: int
    app: str
    field: str
    mode: str
    goal: bool | str
    initial: tuple[bool | str, ...]
    labels: tuple[str, ...]
    instruction: str

    @property
    def task_id(self):
        scenario = sha256(f"wll-authored-panel-001:{self.scenario}".encode()).hexdigest()[:7]
        return f"{scenario}_{self.variant}"

    @property
    def target_id(self):
        return self.variant

    @property
    def guard_id(self):
        return self.variant % len(self.labels) + 1

    @property
    def final(self):
        if self.mode == "append":
            return self.initial[self.target_id - 1] + "\n" + self.goal
        return self.goal


def _tasks(variant):
    origin = ORIGINS[variant - 1]
    labels = tuple(f"{origin} {name}" for name in ("Check-in", "Follow-up", "Review"))
    target = labels[variant - 1]
    values = tuple(f"{origin} draft {index}" for index in (71, 82, 93))
    completion = variant != 2
    switch = tuple((index != variant) and ((index + variant) % 2 == 0) for index in (1, 2, 3))
    inverse = tuple(not value for value in switch)
    content = f"{origin} approved"
    untouched = " Leave the other records unchanged."

    def row(scenario, app, field, mode, goal, initial, instruction):
        return Task(scenario, variant, app, field, mode, goal, initial, labels, instruction)

    return (
        row("pin_note", "simple_note", "pinned", "set", True, switch,
            f'Pin the Simple Note titled "{target}".' + untouched),
        row("unpin_note", "simple_note", "pinned", "set", False, inverse,
            f'Unpin the Simple Note titled "{target}" without changing its contents.' + untouched),
        row("complete_todo", "todoist", "is_completed", "set", completion,
            switch if completion else inverse,
            (f'Mark the Todoist Inbox task "{target}" as '
             f'{"completed" if completion else "not completed"}.' + untouched)),
        row("switch_alarm", "phone", "enabled", "set", completion,
            switch if completion else inverse,
            f'{"Enable" if completion else "Disable"} the phone alarm labeled "{target}".' + untouched),
        row("mark_email", "gmail", "read", "set", completion,
            switch if completion else inverse,
            (f'Mark the Gmail thread with subject "{target}" as '
             f'{"read" if completion else "unread"}.' + untouched)),
        row("append_note", "simple_note", "content", "append", content, values,
            (f'Append a new line containing exactly "{content}" to the Simple Note "{target}". '
             "Keep the existing text and add the line exactly once." + untouched)),
        row("keep_pinned", "simple_note", "pinned", "keep", True, inverse,
            (f'Make sure the Simple Note titled "{target}" is pinned. '
             "If it is already pinned, leave it alone." + untouched)),
        row("read_note", "simple_note", "content", "read", values[variant - 1], values,
            f'Return the exact content of the Simple Note "{target}" without modifying any notes.'),
        row("edit_description", "todoist", "description", "set", content, values,
            f'Change only the description of the Todoist Inbox task "{target}" to "{content}".'
            + untouched),
    )


PANEL = {task.task_id: task for variant in (1, 2, 3) for task in _tasks(variant)}
