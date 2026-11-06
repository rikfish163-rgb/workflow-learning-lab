"""Private JSONL worker. Run only inside the restricted AppWorld container.

Protocol input is controller-owned; model output is only passed to execute().
Never serialize evaluator reports or ground-truth assertions to the controller.
"""
import contextlib
import json
import sys
import traceback

from code_policy import violation


def main():
    from appworld import AppWorld, load_task_ids
    from appworld.task import Task

    world = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
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
                    world = AppWorld(task_id=req["task_id"], experiment_name="isolated",
                                     max_interactions=req["max_steps"],
                                     load_ground_truth=False,
                                     timeout_seconds=30, raise_on_unsafe_syntax=True,
                                     raise_on_unsafe_execution=True)
                    result = True
                elif op == "execute":
                    reason = violation(req["code"])
                    result = ("Execution failed. Policy restriction: " + reason
                              if reason else world.execute(req["code"]))
                elif op == "completed":
                    result = world.task_completed()
                elif op == "evaluate":
                    from appworld.evaluator import evaluate_task

                    tracker = evaluate_task(task_id=world.task_id,
                                            experiment_name=world.experiment_name,
                                            suppress_errors=True)
                    if tracker.total_count != tracker.num_tests or tracker.num_tests == 0:
                        raise RuntimeError("incomplete official evaluation")
                    result = tracker.success
                elif op == "close":
                    if world is not None:
                        world.close()
                    result = True
                else:
                    raise ValueError("unknown controller operation")
            print(json.dumps({"ok": True, "result": result}), flush=True)
            if op == "close":
                break
        except Exception as exc:
            if req.get("op") in {"start", "list", "describe"}:
                traceback.print_exc(file=sys.stderr)
            # Error type alone avoids accidentally exposing ground truth on evaluation errors.
            print(json.dumps({"ok": False, "error": type(exc).__name__}), flush=True)


if __name__ == "__main__":
    main()
