"""
jobs.py -- the Train tab's detached-job registry.

Training launches as an independent OS subprocess (subprocess.Popen with
Windows' CREATE_NEW_PROCESS_GROUP/DETACHED_PROCESS, not a QThread/
in-process call) so it survives the hub closing, crashing, or a remote-
desktop disconnect -- the whole point of the "reattachable monitor"
decision. State lives on disk (this module's jobs.json + each job's own
progress/log files, all under data_engine/data/gui_jobs/), not in the GUI
process, which is what makes "reattach after restarting the hub" work at
all: list_jobs() just re-derives current status from disk + psutil, same
as it would on first launch.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil

from data_engine.registry.registry import REPO_ROOT

JOBS_DIR = REPO_ROOT / "data_engine" / "data" / "gui_jobs"
JOBS_FILE = JOBS_DIR / "jobs.json"


def _load_jobs() -> list:
    if not JOBS_FILE.exists():
        return []
    try:
        return json.loads(JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _save_jobs(jobs: list) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def read_progress(job: dict):
    path = job.get("progress_file")
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def tail_log(job: dict, n_lines: int = 200) -> str:
    path = job.get("log_file")
    if not path or not Path(path).exists():
        return ""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def is_alive(job: dict) -> bool:
    """True if job's PID is running AND still looks like this specific
    training process -- guards against PID reuse after the original
    process already exited (matching on this job's own --progress-file
    path is more precise than trying to reconstruct/guess a run_id, and
    it's a value the GUI itself chose, so it's known up front)."""
    pid = job.get("pid")
    if pid is None or not psutil.pid_exists(pid):
        return False
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return "data_engine.stages.train.train" in cmdline and job.get("progress_file", "\0") in cmdline


def list_jobs() -> list:
    """Refresh each 'running' job's status against reality before
    returning. This is the entire reattach mechanism -- a freshly-started
    hub calls this and gets back accurate status with no extra state."""
    jobs = _load_jobs()
    changed = False
    for job in jobs:
        if job["status"] == "running" and not is_alive(job):
            progress = read_progress(job)
            save_dir = progress.get("save_dir") if progress else None
            if save_dir and (Path(save_dir) / "run_manifest.json").exists():
                job["status"] = "completed"
                job["save_dir"] = save_dir
            else:
                job["status"] = "failed"
                job["save_dir"] = save_dir
            changed = True
    if changed:
        _save_jobs(jobs)
    return jobs


def launch(argv: list) -> dict:
    """argv: train.py CLI flags (NOT including the interpreter/module
    prefix, and NOT including --progress-file -- this appends that itself
    so the job's identity/progress path is decided here, not guessed from
    Ultralytics' project=/name= path nesting later)."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    progress_file = JOBS_DIR / f"{job_id}_progress.json"
    log_file = JOBS_DIR / f"{job_id}.log"

    cmd = [sys.executable, "-m", "data_engine.stages.train.train"] + argv + [
        "--progress-file", str(progress_file),
    ]

    creationflags = 0
    if sys.platform == "win32":
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: DETACHED_PROCESS only hides
        # a console for THIS process -- confirmed empirically to still let
        # Windows pop open a new visible console for each of train.py's
        # PyTorch DataLoader worker subprocesses (workers=2), since they have
        # no parent console to inherit either. CREATE_NO_WINDOW suppresses
        # console allocation for the whole process tree, not just the top one.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with open(log_file, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), creationflags=creationflags,
        )

    job = {
        "job_id": job_id,
        "pid": proc.pid,
        "command": cmd,
        "progress_file": str(progress_file),
        "log_file": str(log_file),
        "save_dir": None,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    return job


def stop(job: dict) -> None:
    pid = job.get("pid")
    if pid and psutil.pid_exists(pid) and is_alive(job):
        try:
            psutil.Process(pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def remove(job_id: str) -> None:
    """Drops a job from the registry and deletes its progress/log files --
    for dismissing stopped/crashed/completed jobs from the Train tab's
    list. Does NOT touch the actual training output (weights,
    run_manifest.json under data_engine/models/) -- only this GUI's own
    job-tracking bookkeeping. Never call this on a still-running job (the
    Train tab only enables its "Remove" button once status != "running");
    it doesn't stop anything itself."""
    all_jobs = _load_jobs()
    job = next((j for j in all_jobs if j["job_id"] == job_id), None)
    _save_jobs([j for j in all_jobs if j["job_id"] != job_id])
    if job:
        for key in ("progress_file", "log_file"):
            path = job.get(key)
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except OSError:
                    pass
