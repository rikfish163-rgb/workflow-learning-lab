"""Single-task private stdio worker for the native AppWorld tool boundary.

Run in a separately constrained CPU process using a private controller pipe.
This worker does not supply kernel resource limits or host isolation, run a
model/controller/scorer, or equate its counters with CodeAct experiment steps.
Only ``dispatch.action`` is model-provided data; CLI options and RPC operations
belong to the trusted controller. Metadata, replies, and all files are private.

RPC lines are exactly {op: metadata}, {op: dispatch, action: JSON-string},
{op: save}, or {op: close}. Replies contain ok, result (or error_type/message),
and numeric cost counters. All recognized dispatch requests count as attempts,
including invalid actions, malformed dispatch envelopes, and limit rejections.
World/API output is redirected to the exclusive private runtime.log. No action
is retried. Only tool-boundary rejections and observed upstream HTTP 4xx errors
are recoverable. Other API errors and all save failures stop the worker with
unknown outcome/cost; raw diagnostics remain in the private journal.

Every recognized dispatch invalidates the saved snapshot, even on rejection.
Only a complete explicit save with durable success records restores freshness.
Scoring eligibility additionally requires successful cleanup; dirty close/EOF
returns nonzero and persists closed_dirty, never an eligible saved snapshot.
"""

import argparse
import contextlib
import json
import os
import re
import socket
import stat
import sys
from pathlib import Path
from typing import Any, BinaryIO, TextIO

if __package__:
    from .native_appworld_tools import (
        ToolDispatchError,
        _check_depth,
        _check_json_values,
        _reject_constant,
        _unique_object,
        build_appworld_dispatcher,
    )
else:
    from native_appworld_tools import (
        ToolDispatchError,
        _check_depth,
        _check_json_values,
        _reject_constant,
        _unique_object,
        build_appworld_dispatcher,
    )

PRIVATE_STATE_ROOT = Path("/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state")
ASSET_ROOT = PRIVATE_STATE_ROOT / "native-cpu-assets-001"
MAX_RPC_BYTES = 6 * 65536 + 4096
_TASK_ID = re.compile(r"[0-9a-f]{7}_[1-9][0-9]*", re.ASCII)
_EXPERIMENT = re.compile(r"[A-Za-z0-9_-]{1,128}", re.ASCII)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class WorkerError(RuntimeError):
    """Trusted worker configuration or lifecycle validation failed."""


class FatalWorkerError(WorkerError):
    """The process must stop and clean up without further API calls or saves."""


class ObservedAPIError(Exception):
    """The trusted requester observed an HTTP 4xx and its original raiser failed."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _open_directory(path: str | Path) -> int:
    raw = os.fspath(path)
    parsed = Path(raw)
    if not parsed.is_absolute() or raw != str(parsed) or ".." in parsed.parts:
        raise WorkerError("Directory paths must be absolute and canonical.")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parsed.parts[1:]:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError:
        os.close(fd)
        raise WorkerError("A required directory is missing or contains a symlink.") from None


def _check_private_directory(fd: int) -> None:
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise WorkerError("Private directories must be owned by this UID with mode 0700.")


def _child_directory(parent_fd: int, name: str, *, private: bool = False) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError:
        raise WorkerError("Output directories must not contain symlinks.") from None
    if private:
        try:
            _check_private_directory(fd)
        except BaseException:
            os.close(fd)
            raise
    return fd


def _exclusive_file(directory_fd: int, name: str) -> TextIO:
    try:
        fd = os.open(
            name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600, dir_fd=directory_fd,
        )
    except OSError:
        raise WorkerError("Private state files and reservations must be new.") from None
    try:
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "w+", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


def _flush(file: TextIO) -> None:
    file.flush()
    os.fsync(file.fileno())


def _check_new_output(experiment_fd: int, task_id: str) -> None:
    try:
        tasks_fd = os.open("tasks", _DIR_FLAGS, dir_fd=experiment_fd)
    except FileNotFoundError:
        return
    except OSError:
        raise WorkerError("The task output parent must not be a symlink.") from None
    try:
        try:
            os.stat(task_id, dir_fd=tasks_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise WorkerError("Task output already exists; reinitialization is forbidden.")
    finally:
        os.close(tasks_fd)


def _reserve_output(task_id: str, experiment_name: str, run_dir: Path) -> Path:
    fd = _open_directory(ASSET_ROOT)
    try:
        for name in ("experiments", "outputs", experiment_name):
            child = _child_directory(fd, name, private=name == experiment_name)
            os.close(fd)
            fd = child
        _check_new_output(fd, task_id)
        reservations_fd = _child_directory(fd, ".native-reservations", private=True)
        try:
            with _exclusive_file(reservations_fd, task_id + ".json") as marker:
                marker.write(_json_text({"task_id": task_id, "experiment_name": experiment_name, "run_dir": str(run_dir)}) + "\n")
                _flush(marker)
            os.fsync(reservations_fd)
        finally:
            os.close(reservations_fd)
        _check_new_output(fd, task_id)
    finally:
        os.close(fd)
    return ASSET_ROOT / "experiments" / "outputs" / experiment_name / "tasks" / task_id


def _create_world(**kwargs: Any) -> Any:
    """Import only the fixed existing CPU assets; never import model code."""
    deps = ASSET_ROOT / "deps"
    os.close(_open_directory(deps))
    if str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    from appworld import AppWorld
    from appworld.common.path_store import path_store

    path_store.update_root(str(ASSET_ROOT))
    try:
        return AppWorld(**kwargs)
    except BaseException:
        AppWorld.close_all()
        raise


def _parse_rpc(message: str | bytes) -> dict[str, Any] | None:
    try:
        if type(message) not in (str, bytes) or len(message) > MAX_RPC_BYTES:
            return None
        text = message.decode("utf-8") if type(message) is bytes else message
        if len(text.encode("utf-8")) > MAX_RPC_BYTES:
            return None
        _check_depth(text, 32)
        request = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        _check_json_values(request)
        if type(request) is not dict or type(request.get("op")) is not str:
            return None
        return request
    except (ValueError, UnicodeError, RecursionError):
        return None


class NativeAppWorldWorker:
    def __init__(
        self,
        task_id: str,
        experiment_name: str,
        run_dir: str | Path,
        *,
        max_actions: int = 40,
        max_api_requests: int = 128,
        _world_factory: Any = None,
    ) -> None:
        if not _TASK_ID.fullmatch(task_id) or not _EXPERIMENT.fullmatch(experiment_name):
            raise WorkerError("Invalid task or experiment identifier.")
        for limit in (max_actions, max_api_requests):
            if type(limit) is not int or limit < 1:
                raise WorkerError("Worker limits must be positive integers.")
        if socket.gethostname() != "amax" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise WorkerError("The CPU worker requires host amax and explicitly empty CUDA visibility.")
        parsed_run_dir = Path(run_dir)
        if PRIVATE_STATE_ROOT not in parsed_run_dir.parents:
            raise WorkerError("The run directory must be inside the fixed private-state root.")
        run_fd = _open_directory(run_dir)
        try:
            _check_private_directory(run_fd)
        except BaseException:
            os.close(run_fd)
            raise
        self._task_id, self._experiment_name = task_id, experiment_name
        self._run_dir, self._max_actions = parsed_run_dir, max_actions
        self._max_api_requests = max_api_requests
        self._world = self._dispatcher = None
        self._status_file = self._journal_file = self._runtime_file = None
        self._attempts = self._upstream_requests = 0
        self._fatal = self._closed = False
        self._current_state_saved = False
        self._last_successful_save_through_attempt = None
        self._outcome_unknown = self._unknown_cost = False
        self._original_raise_if_failure = None
        try:
            try:
                self._status_file = _exclusive_file(run_fd, "status.json")
                self._journal_file = _exclusive_file(run_fd, "actions.jsonl")
                self._runtime_file = _exclusive_file(run_fd, "runtime.log")
                os.fsync(run_fd)
            finally:
                os.close(run_fd)
            self._set_status("initializing")
            self._output_directory = _reserve_output(task_id, experiment_name, parsed_run_dir)
            os.environ["APPWORLD_ROOT"] = str(ASSET_ROOT)
            self._world = self._private_call(
                _world_factory or _create_world,
                task_id=task_id, experiment_name=experiment_name,
                remote_apis_url=None, remote_environment_url=None, remote_mcp_url=None,
                remote_docker=False, load_ground_truth=False,
                raise_on_unsafe_syntax=True, raise_on_unsafe_execution=True,
                import_utils=False, parse_datetimes=False, allow_datetime_change=False,
                add_login_shortcut=False, include_direct_functions=False,
                raise_on_failure=True, raise_on_extra_parameters=True,
                munchify_response=False, wrap_response=False, unwrap_response=False,
                random_seed=100, max_api_calls_per_interaction=max_api_requests,
            )
            if self._world.task.ground_truth is not None:
                raise WorkerError("Ground truth must not be loaded in the task worker.")
            if (
                Path(self._world.output_directory) != self._output_directory
                or Path(self._world.output_db_home_path_on_disk) != self._output_directory / "dbs"
            ):
                raise WorkerError("The world output path differs from the reserved output.")
            self._install_http_failure_classifier()
            self._dispatcher = self._private_call(build_appworld_dispatcher, self._world.apis)
            self._set_status("ready")
        except BaseException:
            self._mark_unknown()
            try:
                self.close()
            except BaseException:
                pass
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def costs(self) -> dict[str, int]:
        if self._world is not None and not self._closed:
            self._upstream_requests = max(self._upstream_requests, len(self._world.requester.requests))
        return {
            "action_attempts": self._attempts,
            "api_invocations": self._dispatcher.api_invocations if self._dispatcher is not None else 0,
            "upstream_requests": self._upstream_requests,
        }

    @property
    def freshness(self) -> dict[str, Any]:
        return {
            "last_successful_save_through_attempt": self._last_successful_save_through_attempt,
            "current_state_saved": self._current_state_saved,
            "score_eligible": (
                self._closed and self._current_state_saved and not self._fatal
                and not self._outcome_unknown and not self._unknown_cost
                and self._last_successful_save_through_attempt == self._attempts
            ),
            "outcome_unknown": self._outcome_unknown,
            "unknown_cost": self._unknown_cost,
        }

    def _install_http_failure_classifier(self) -> None:
        requester = self._world.requester
        original = requester.raise_if_failure
        self._original_raise_if_failure = original

        def observed_raise_if_failure(response: Any, raise_on_failure: bool | None = None) -> None:
            try:
                return original(response, raise_on_failure=raise_on_failure)
            except Exception as error:
                status = getattr(response, "status_code", None)
                # The source-bound upstream raiser's deliberate HTTP failure is
                # exactly Exception. Unexpected parser/internal subclasses stay fatal.
                if type(status) is int and 400 <= status < 500 and type(error) is Exception:
                    raise ObservedAPIError(status, str(error)) from None
                raise

        requester.raise_if_failure = observed_raise_if_failure

    def _mark_unknown(self) -> None:
        self._fatal = self._outcome_unknown = self._unknown_cost = True
        self._current_state_saved = False

    def _unknown_failure(self, event: str, error: Exception) -> None:
        self._mark_unknown()
        self._record({
            "event": event, "attempt": self._attempts,
            "error_type": type(error).__name__, "message": str(error),
            "outcome_unknown": True, "unknown_cost": True, "cost": self.costs,
        })
        self._set_status("failed")
        raise FatalWorkerError("Worker stopped after an unknown outcome.") from None

    def _private_call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            with contextlib.redirect_stdout(self._runtime_file), contextlib.redirect_stderr(self._runtime_file):
                return function(*args, **kwargs)
        finally:
            _flush(self._runtime_file)

    def _append_record(self, record: dict[str, Any]) -> None:
        self._journal_file.write(_json_text(record) + "\n")
        _flush(self._journal_file)

    def _record(self, record: dict[str, Any]) -> None:
        try:
            self._append_record(record)
        except Exception:
            self._mark_unknown()
            raise FatalWorkerError("Private action journaling failed.") from None

    def _set_status(self, state: str) -> None:
        try:
            self._status_file.seek(0)
            self._status_file.write(_json_text({
                "state": state, "task_id": self._task_id,
                "experiment_name": self._experiment_name,
                "max_actions": self._max_actions, "max_api_requests": self._max_api_requests,
                "cost": self.costs, **self.freshness,
            }) + "\n")
            self._status_file.truncate()
            _flush(self._status_file)
        except Exception:
            self._mark_unknown()
            raise FatalWorkerError("Private worker status logging failed.") from None

    def _success(self, result: Any) -> dict[str, Any]:
        return {"ok": True, "result": result, "cost": self.costs, "freshness": self.freshness}

    def _error(self, error_type: str, message: str) -> dict[str, Any]:
        return {"ok": False, "error_type": error_type, "message": message, "cost": self.costs, "freshness": self.freshness}

    def handle_request(self, message: str | bytes) -> dict[str, Any]:
        if self._closed:
            return self._error("WorkerClosed", "Worker is closed.")
        if self._fatal:
            raise FatalWorkerError("Worker stopped after a private infrastructure failure.")
        request = _parse_rpc(message)
        if request is None:
            return self._error("InvalidRPC", "Invalid RPC request.")
        op = request["op"]
        if op == "dispatch":
            self._attempts += 1
            self._current_state_saved = False
            self._set_status("ready")  # invalidate previous disk freshness before any API effect
            action = request.get("action")
            self._record({"event": "attempt", "attempt": self._attempts, "action": action if type(action) is str else None})
            if set(request) != {"op", "action"} or type(action) is not str:
                reply = self._error("InvalidRPC", "Invalid RPC request.")
            elif self._attempts > self._max_actions:
                reply = self._error("ActionLimitError", "The dispatch attempt limit has been reached.")
            else:
                try:
                    result = self._private_call(self._dispatcher.dispatch, action)
                    reply = self._success(result)
                except (ToolDispatchError, ObservedAPIError) as error:
                    reply = self._error(type(error).__name__, str(error))
                    if isinstance(error, ObservedAPIError):
                        reply["status_code"] = error.status_code
                except Exception as error:
                    self._unknown_failure("result", error)
            self._record({"event": "result", "attempt": self._attempts, "reply": reply})
            self._set_status("ready")
            return reply
        if set(request) != {"op"} or op not in ("metadata", "save", "close"):
            return self._error("InvalidRPC", "Invalid RPC request.")
        if op == "metadata":
            task = self._world.task
            return self._success({
                "instruction": task.instruction,
                "supervisor": {key: task.supervisor[key] for key in ("first_name", "last_name", "email", "phone_number") if key in task.supervisor},
                "app_descriptions": task.app_descriptions,
            })
        if op == "save":
            self._current_state_saved = False  # save_state may delete the previous snapshot first
            self._set_status("saving")
            self._record({"event": "save_attempt"})
            try:
                self._private_call(self._world.save_state)
            except Exception as error:
                self._unknown_failure("save_result", error)
            self._record({"event": "save_result", "success": True, "saved_through_attempt": self._attempts})
            self._last_successful_save_through_attempt = self._attempts
            self._current_state_saved = True
            self._set_status("ready")
            return self._success({"db_path": str(self._output_directory / "dbs")})
        self.close()
        return self._success(None)

    def close(self) -> None:
        if self._closed:
            return
        _ = self.costs  # retain upstream request counts before Requester.close resets them
        self._closed = True
        failed = False
        if self._world is not None:
            for cleanup in (self._world.close, type(self._world).close_all):
                try:
                    self._private_call(cleanup)
                except BaseException:
                    failed = True
                    self._mark_unknown()
        for file in (self._journal_file, self._runtime_file):
            if file is not None:
                try:
                    file.close()
                except Exception:
                    failed = True
                    self._mark_unknown()
        try:
            if self._status_file is not None:
                state = "failed" if self._fatal else ("closed" if self._current_state_saved else "closed_dirty")
                self._set_status(state)
        except Exception:
            failed = True
        finally:
            if self._status_file is not None:
                try:
                    self._status_file.close()
                except Exception:
                    failed = True
                    self._mark_unknown()
        if failed:
            raise FatalWorkerError("Worker cleanup or private logging failed.")


def serve_stdio(worker: NativeAppWorldWorker, incoming: BinaryIO, outgoing: TextIO) -> int:
    """Serve the trusted private pipe, closing the world on EOF or any failure."""
    exit_code = 0
    try:
        while not worker.closed:
            line = incoming.readline(MAX_RPC_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_RPC_BYTES or not line.endswith(b"\n"):
                worker._fatal = True
                worker._current_state_saved = False
                reply = worker._error("InvalidRPC", "Invalid RPC request.")
                exit_code = 1
            else:
                reply = worker.handle_request(line)
            outgoing.write(_json_text(reply) + "\n")
            outgoing.flush()
            if exit_code:
                break
    except Exception:
        worker._mark_unknown()
        exit_code = 1
        try:
            outgoing.write(_json_text(worker._error("WorkerStopped", "Worker stopped after a private infrastructure failure.")) + "\n")
            outgoing.flush()
        except Exception:
            pass
    finally:
        try:
            worker.close()
        except BaseException:
            exit_code = 1
        if not worker.freshness["score_eligible"]:
            exit_code = 1
    return exit_code


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-actions", type=_positive_integer, required=True)
    parser.add_argument("--max-api-requests", type=_positive_integer, required=True)
    args = parser.parse_args(argv)
    try:
        worker = NativeAppWorldWorker(
            args.task_id, args.experiment_name, args.run_dir,
            max_actions=args.max_actions, max_api_requests=args.max_api_requests,
        )
    except Exception:
        sys.stderr.write("Native AppWorld worker initialization failed. Inspect private state.\n")
        return 1
    return serve_stdio(worker, sys.stdin.buffer, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
