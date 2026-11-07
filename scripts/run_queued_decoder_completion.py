"""CPU-side client for the registered two-task completion; waits for remote idle dispatch."""
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/home/hts/workflow-learning-lab"
JOB_ID = "decoder-sft-greedy-completion-001"
PORT = 18969
MODEL = "qwen3-4b-teacher-init-001"
RESUME = ROOT / "artifacts/decoder-sft-greedy-resume-plan-001.json"
STATE = ROOT / "artifacts/decoder-resume-controller-status.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
       "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", "4090"]


def remote_python(source, payload=None):
    result = subprocess.run([*SSH, "python3 -c " + shlex.quote(source)],
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True, text=True, timeout=20, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def preflight():
    resume = json.loads(RESUME.read_text())
    plan = json.loads((ROOT / "artifacts/decoder-factorial-plan-v1.json").read_text())
    if resume["job_id"] != JOB_ID or Path(resume["completion_output"]).exists():
        raise ValueError("job mismatch or an attempt already exists; inspect rather than retry")
    for name, digest in resume["original_files"].items():
        path = ROOT / "artifacts/decoder-sft-greedy-001" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("original evidence changed")
    for name, digest in resume["inference_code_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen inference code changed")
    if Path(resume["task_ids_file"]).read_text().splitlines() != resume["pending_ids"]:
        raise ValueError("pending task IDs changed")
    for key in ("data_dir", "deps_dir"):
        if not Path(plan[key]).is_dir():
            raise ValueError("local benchmark runtime is unavailable")
    image = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", plan["image"]],
                                    text=True, timeout=15).strip()
    if image != plan["image"]:
        raise ValueError("benchmark image changed")
    remote = remote_python('''import hashlib,json,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');os.chdir(root)
old=json.loads(Path('artifacts/decoder-sft-server-001.json').read_text())
p=Path(f"/proc/{old['process_group']}")
assert not p.exists() or (p/'stat').read_text().split()[21]!=old['start_time']
weights=Path('artifacts/teacher-warmstart-sft-001/adapter/adapter_model.safetensors')
print(json.dumps({'adapter_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
 'policy_sha256':hashlib.sha256(Path('config/gpu-policy.json').read_bytes()).hexdigest(),
 'server_sha256':hashlib.sha256(Path('scripts/serve_transformers.py').read_bytes()).hexdigest()}))
''')
    if (remote["adapter_sha256"] != plan["adapter_weights_sha256"]
            or remote["policy_sha256"] != resume["allowed_gpu_policies"][0]["policy_sha256"]
            or remote["server_sha256"] != resume["inference_code_sha256"]["scripts/serve_transformers.py"]):
        raise ValueError("remote model, server or policy mismatch")
    return plan, resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan, resume = preflight()
    if args.check:
        print(json.dumps({"preflight": "passed", "pending_ids": resume["pending_ids"], "uses_gpu": False}))
        return

    def status(phase, **extra):
        atomic_json(STATE, {"phase": phase, "pid": os.getpid(), "epoch": time.time(),
                           "job_id": JOB_ID, "pending_ids": resume["pending_ids"], **extra})
        print(json.dumps({"phase": phase, **extra}), flush=True)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    stopped = threading.Event()
    remote_state = {}

    def heartbeat():
        while not stopped.is_set():
            try:
                snapshot = remote_python('''import json,time
from pathlib import Path
p=Path('/home/hts/workflow-learning-lab/artifacts')
(p/'gpu-idle-heartbeat-decoder-sft-greedy-completion-001').touch()
def read(name):
 f=p/name
 return json.loads(f.read_text()) if f.exists() else None
print(json.dumps({'watch':read('gpu-idle-watch-status.json'),
 'job':read('gpu-idle-job-decoder-sft-greedy-completion-001.json'),'epoch':time.time()}))
''')
                remote_state.update(snapshot)
                atomic_json(ROOT / "artifacts/gpu-idle-remote-status.json", snapshot)
            except Exception as exc:
                remote_state["last_connection_error"] = type(exc).__name__
            stopped.wait(15)

    client = tunnel = thread = None
    with (ROOT / "artifacts/decoder-resume-controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            argv = ["/home/hts/budgeted-reward-verification/.venv/bin/python", "scripts/serve_transformers.py",
                "--model-path", plan["model_path"], "--adapter-path", plan["adapter_path"],
                "--model-id", MODEL, "--port", str(PORT), "--top-p", "0.8", "--top-k", "20",
                "--min-p", "0", "--max-context", "32768", "--max-output-tokens", "1024"]
            job = {"id": JOB_ID, "argv": argv, "max_runtime_seconds": 2700,
                   "resume_plan_sha256": hashlib.sha256(RESUME.read_bytes()).hexdigest(),
                   "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            remote_python('''import json,sys
from pathlib import Path
sys.path.insert(0,'/home/hts/workflow-learning-lab')
from wll.storage import atomic_json
p=Path('/home/hts/workflow-learning-lab/artifacts/gpu-idle-queue.json');job=json.load(sys.stdin)
assert not p.exists(), 'Queue already exists; inspect rather than replace'
atomic_json(p,job)
''', job)
            atomic_json(ROOT / "artifacts/gpu-idle-queue.json", job)
            with (ROOT / "artifacts/decoder-resume-tunnel.log").open("x") as log:
                tunnel = subprocess.Popen([*SSH[:-1], "-N", "-o", "ExitOnForwardFailure=yes",
                    "-L", f"{PORT}:127.0.0.1:{PORT}", "4090"], stdout=log, stderr=subprocess.STDOUT)
                thread = threading.Thread(target=heartbeat, daemon=True)
                thread.start()
                status("waiting_for_gpu8_or_gpu9_idle_600_seconds")
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                while True:
                    if tunnel.poll() is not None:
                        raise RuntimeError("SSH tunnel closed; retain evidence and stop heartbeat")
                    job_state = remote_state.get("job") or {}
                    if job_state.get("status") == "finished":
                        raise RuntimeError("remote job exited before healthy startup")
                    if job_state.get("status") == "running":
                        try:
                            with opener.open(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
                                health = json.load(response)
                            break
                        except OSError:
                            if time.time() - job_state["started_epoch"] > 300:
                                raise RuntimeError("model health timeout") from None
                    time.sleep(5)
                prior = json.loads((ROOT / "artifacts/decoder-sft-health-001.json").read_text())
                actual, expected = health["backend"], prior["backend"]
                if (actual["gpu_policy"] not in resume["allowed_gpu_policies"]
                        or {k: v for k, v in actual.items() if k != "gpu_policy"}
                        != {k: v for k, v in expected.items() if k != "gpu_policy"}):
                    raise ValueError("inference backend changed beyond the resource amendment")
                atomic_json(ROOT / "artifacts/decoder-resume-health.json", health)
                status("running_two_incomplete_tasks", gpu=actual["gpu_policy"])
                command = [sys.executable, "-u", "-m", "wll.cli", "run",
                    "--data-dir", plan["data_dir"], "--deps-dir", plan["deps_dir"], "--image", plan["image"],
                    "--split", "train", "--task-ids-file", resume["task_ids_file"],
                    "--seed", str(plan["seed"]), "--steps", str(plan["registered_steps_per_episode"]),
                    "--max-tokens", str(plan["generation"]["max_tokens"]), "--temperature", "0",
                    "--prompt-version", plan["prompt_version"], "--endpoint", f"http://127.0.0.1:{PORT}/v1",
                    "--model", MODEL, "--output", resume["completion_output"]]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT),
                           TMPDIR="/media/hetaisheng/044A81D94A81C83E/wll-runtime/tmp")
                client = subprocess.Popen(command, env=env)
                if client.wait(timeout=2400) != 0:
                    raise RuntimeError("completion has infrastructure errors; no automatic resampling")
                subprocess.run([sys.executable, "scripts/report_decoder_factorial.py", "--root", "artifacts",
                    "--output", "artifacts/decoder-factorial-report-001.json", "--resume-plan", str(RESUME)],
                    env=env, check=True, timeout=60)
                report = json.loads((ROOT / "artifacts/decoder-factorial-report-001.json").read_text())
                status("complete", decision=report["decision"])
        except BaseException as exc:
            status("failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if client is not None and client.poll() is None:
                client.terminate()
                try:
                    client.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    client.kill()
                    client.wait()
            stopped.set()
            if thread is not None:
                thread.join(timeout=25)
            try:
                remote_python('''import json,os,signal
from pathlib import Path
p=Path('/home/hts/workflow-learning-lab/artifacts')
(p/'gpu-idle-heartbeat-decoder-sft-greedy-completion-001').unlink(missing_ok=True)
f=p/'gpu-idle-job-decoder-sft-greedy-completion-001.json'
if f.exists():
 info=json.loads(f.read_text());pid=info.get('process_group');proc=Path(f'/proc/{pid}')
 if pid and proc.exists() and (proc/'stat').read_text().split()[21]==info.get('start_ticks'):
  if b'scripts/serve_transformers.py' in (proc/'cmdline').read_bytes() and os.getpgid(pid)==pid:
   os.killpg(pid,signal.SIGTERM)
''')
            except Exception:
                print("Cleanup unconfirmed; remote heartbeat expiry and runtime cap remain active.", flush=True)
            if tunnel is not None and tunnel.poll() is None:
                tunnel.terminate()
                tunnel.wait(timeout=10)


if __name__ == "__main__":
    main()
