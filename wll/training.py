"""Distillation examples omit the lesson from the student's input."""
from .prompts import initial_messages
from .protocol import fingerprint


def export_validated_transfer_examples(episodes, report_paths):
    from .experience import load_validated_experience

    reports = {}
    for path in report_paths:
        candidate, evidence = load_validated_experience(path)
        if candidate.experience_id in reports:
            raise ValueError("duplicate experience validation")
        reports[candidate.experience_id] = candidate, evidence
    for episode in episodes:
        if episode.experience_id not in reports:
            raise ValueError("missing experience validation")
        candidate, evidence = reports[episode.experience_id]
        if episode.metadata.get("validation_report_hash") != evidence["report_hash"] or (
                episode.model_id != evidence["model_id"]):
            raise ValueError("episode validation provenance or model differs")
        if episode.metadata.get("backend") != evidence["backend"]:
            raise ValueError("episode inference backend differs from validation")
        if episode.task.scenario_id in set(candidate.source_scenarios + evidence["validation_scenarios"]):
            raise ValueError("distillation must use independent training scenarios")
    return export_first_attempt_examples(episodes)


def encode_target(tokenizer, messages, max_length):
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be assistant target")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=False, return_dict=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError("chat template prefix mismatch; cannot safely infer label boundary")
    if len(tokens) > max_length or len(tokens) <= len(prefix):
        raise ValueError("invalid sample length; do not silently truncate")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prefix) + tokens[len(prefix):]}


def export_first_attempt_examples(episodes):
    rows = []
    for e in episodes:
        if e.task.split != "train":
            raise ValueError("only train split may enter SFT")
        if e.branch != "first" or e.replayed_steps:
            raise ValueError("recovery suffix is not first-attempt supervision; re-solve from reset")
        if not e.success:
            continue
        history = initial_messages(e.task, e.metadata.get("prompt_version", "appworld-v1"))
        for i, step in enumerate(e.steps):
            target = {"role": "assistant", "content": step.action}
            # A successful final state does not make an earlier rejected action
            # a positive imitation target. Keep the error in the next history,
            # so the corrected action is still learned in its recovery context.
            observed_error = step.observation.startswith(("Execution failed", "Generation failed:"))
            if step.action.strip() and not observed_error:
                rows.append({
                    "messages": history + [target], "target_role": "assistant",
                    "task_id": e.task.task_id, "scenario_id": e.task.scenario_id,
                    "split": "train", "source_model": e.model_id,
                    "source_experience_id": e.experience_id,
                    "source_episode_hash": fingerprint(e.to_dict()), "step_index": i,
                    "target_selection": "exclude_observed_errors_v2",
                })
            history = history + [target, {"role": "user", "content": step.observation}]
    return rows
