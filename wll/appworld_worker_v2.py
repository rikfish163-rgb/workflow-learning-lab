"""Seeded private JSONL worker, only for the restricted AppWorld container.

Model output goes only to execute(), behind the unchanged code policy. Never
serialize evaluator reports or ground-truth assertions to the controller.
"""
import contextlib
import json
import sys
import traceback

from code_policy import violation

SEED_PROTOCOL = "wll-appworld-seed-v1"


def main():
    from appworld import AppWorld, load_task_ids
    from appworld.task import Task

    world = None
    evaluated = False
    try:
        for line in sys.stdin:
            req = None
            try:
                req = json.loads(line)
                if type(req) is not dict:
                    raise TypeError("controller request must be an object")
                with contextlib.redirect_stdout(sys.stderr):
                    op = req["op"]
                    if op == "list":
                        result = load_task_ids(req["split"])
                    elif op == "describe":
                        task = Task.load(task_id=req["task_id"], load_ground_truth=False)
                        result = {"instruction": task.instruction,
                                  "supervisor": str(task.supervisor)}
                    elif op == "start":
                        if world is not None:
                            raise ValueError("worker already has a world")
                        seed = req["environment_seed"]
                        if type(seed) is not int or seed != 100:
                            raise ValueError("environment seed must be the registered integer 100")
                        if type(req["task_id"]) is not str or not req["task_id"]:
                            raise ValueError("task_id must be a nonempty string")
                        if type(req["max_steps"]) is not int or req["max_steps"] < 1:
                            raise ValueError("max_steps must be a positive integer")
                        new_world = AppWorld(task_id=req["task_id"], experiment_name="isolated",
                                             max_interactions=req["max_steps"], random_seed=seed,
                                             load_ground_truth=False,
                                             timeout_seconds=30, raise_on_unsafe_syntax=True,
                                             raise_on_unsafe_execution=True)
                        try:
                            effective_seed = new_world.random_seed
                            if type(effective_seed) is not int or effective_seed != seed:
                                raise ValueError("effective environment seed differs")
                        except BaseException:
                            new_world.close()
                            raise
                        world = new_world
                        result = {"protocol": SEED_PROTOCOL, "started": True,
                                  "task_id": req["task_id"], "environment_seed": {
                                      "requested": seed, "effective": effective_seed,
                                      "verified": True}}
                    elif op == "execute":
                        if world is None or evaluated:
                            raise RuntimeError("execution requires an unevaluated seeded world")
                        reason = violation(req["code"])
                        result = ("Execution failed. Policy restriction: " + reason
                                  if reason else world.execute(req["code"]))
                    elif op == "completed":
                        if world is None:
                            raise RuntimeError("completion requires a seeded world")
                        result = world.task_completed()
                    elif op == "evaluate":
                        if world is None or evaluated:
                            raise RuntimeError("evaluation requires an unevaluated seeded world")
                        evaluated = True
                        from appworld.evaluator import evaluate_task

                        tracker = evaluate_task(task_id=world.task_id,
                                                experiment_name=world.experiment_name,
                                                suppress_errors=True)
                        if tracker.total_count != tracker.num_tests or tracker.num_tests == 0:
                            raise RuntimeError("incomplete official evaluation")
                        result = tracker.success
                        if type(result) is not bool:
                            raise TypeError("official success must be a boolean")
                    elif op == "close":
                        if world is not None:
                            closing, world = world, None
                            closing.close()
                        result = True
                    else:
                        raise ValueError("unknown controller operation")
                print(json.dumps({"ok": True, "result": result}), flush=True)
                if op == "close":
                    break
            except Exception as exc:
                if isinstance(req, dict) and req.get("op") in {"start", "list", "describe"}:
                    traceback.print_exc(file=sys.stderr)
                # Error type only; evaluator details must never reach the actor.
                print(json.dumps({"ok": False, "error": type(exc).__name__}), flush=True)
    finally:
        if world is not None:
            world.close()


if __name__ == "__main__":
    main()
