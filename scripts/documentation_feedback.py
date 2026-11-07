"""Explicit feedback for a narrow observation gap; never rewrite or replay actions.

This is a manually designed scaffold component, not learned experience. A future
study must register its changed environment identity and compare it explicitly.
"""
from scripts.probe_observation_visibility import printable_documentation_call

FEEDBACK = (
    'No documentation was printed by this successful call. For this read-only documentation '
    'query, use print(...) to inspect its returned value before choosing the next action. '
    'Do not repeat mutating operations merely to make their return values visible.'
)


class DocumentationFeedbackWorld:
    def __init__(self, world):
        self.world = world
        self.feedback_events = 0

    def __getattr__(self, name):
        return getattr(self.world, name)

    def execute(self, action):
        observation = self.world.execute(action)
        if observation != 'Execution successful.':
            return observation
        try:
            printable_documentation_call(action)
        except ValueError:
            return observation
        self.feedback_events += 1
        return observation+'\n'+FEEDBACK

    def close(self):
        self.world.close()
