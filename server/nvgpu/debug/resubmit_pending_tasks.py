#!/usr/bin/env python3
"""
Temporary helper to re-submit NVGPU tasks that remained pending after a server reset.

This mirrors the earlier script you ran from the translation workspace: we parse the
gpt_oss_20b_xpiler_extended run log, detect task submissions that never left the
`pending` state, and re-submit their `check_triton.py` scripts via NVGPU.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure the repository root (/workspace) is on sys.path so we can import NVGPU modules.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.nvgpu.client import NVGPUClient, TaskResult  # noqa: E402


@dataclass
class TaskInfo:
    """Metadata extracted from the log for a task submission."""

    task_id: str
    attempt_dir: Optional[Path]
    statuses: set[str]
    line_number: int
    submission_line: str

    @property
    def case_name(self) -> str:
        if not self.attempt_dir:
            return "<unknown>"
        try:
            return self.attempt_dir.parent.name
        except Exception:  # pragma: no cover - defensive
            return str(self.attempt_dir)

    @property
    def attempt_label(self) -> str:
        if not self.attempt_dir:
            return "<unknown>"
        return f"{self.case_name}/{self.attempt_dir.name}"


def parse_pending_tasks(log_path: Path) -> List[TaskInfo]:
    """Parse the run log and return tasks that never left the pending state."""
    copy_pattern = re.compile(r"Copy .+ to (.+?/attempt_\d+)")
    conv_pattern = re.compile(r"Conversation history saved to (.+?/attempt_\d+)/logs/")
    submit_pattern = re.compile(r"Task submitted: ([0-9a-f\-]+)")
    status_pattern = re.compile(r"Task ([0-9a-f\-]+): ([a-z_]+)")

    current_attempt: Optional[Path] = None
    order: list[str] = []
    records: dict[str, TaskInfo] = {}

    with log_path.open(encoding="utf-8") as log_file:
        for lineno, raw_line in enumerate(log_file, start=1):
            line = raw_line.strip()

            copy_match = copy_pattern.search(line)
            if copy_match:
                current_attempt = Path(copy_match.group(1))

            conv_match = conv_pattern.search(line)
            if conv_match:
                current_attempt = Path(conv_match.group(1))

            submit_match = submit_pattern.search(line)
            if submit_match:
                task_id = submit_match.group(1)
                order.append(task_id)
                records[task_id] = TaskInfo(
                    task_id=task_id,
                    attempt_dir=current_attempt,
                    statuses=set(),
                    line_number=lineno,
                    submission_line=line,
                )

            status_match = status_pattern.search(line)
            if status_match:
                task_id, status = status_match.groups()
                status = status.lower()
                if task_id in records:
                    records[task_id].statuses.add(status)

    pending: list[TaskInfo] = []
    for task_id in order:
        info = records[task_id]
        if not info.statuses or info.statuses <= {"pending"}:
            pending.append(info)

    return pending


def add_priority_task(pending_tasks: List[TaskInfo]) -> List[TaskInfo]:
    """Add the specific CUDA error triggering task as the first priority."""
    # The specific task that triggered the CUDA error (now under debug examples)
    priority_path = (
        PROJECT_ROOT
        / "server"
        / "nvgpu"
        / "debug"
        / "gpt_oss_20b_xpiler_extended"
        / "20251018_223123"
        / "batchnorm_128_32_32_32"
        / "attempt_01"
    )
    
    # Check if this task is already in the pending list
    for task in pending_tasks:
        if task.attempt_dir and task.attempt_dir == priority_path:
            # Move it to the front
            pending_tasks.remove(task)
            pending_tasks.insert(0, task)
            print(f"[priority] Moved {task.attempt_label} to front of queue")
            return pending_tasks
    
    # If not found in pending tasks, create a new TaskInfo for it
    priority_task = TaskInfo(
        task_id="priority-cuda-error-task",
        attempt_dir=priority_path,
        statuses={"pending"},  # Assume it was pending
        line_number=0,
        submission_line="Priority task: CUDA error trigger"
    )
    
    # Check if the script exists
    try:
        locate_script(priority_path)
        pending_tasks.insert(0, priority_task)
        print(f"[priority] Added {priority_task.attempt_label} as first priority task")
    except FileNotFoundError as exc:
        print(f"[warn] Priority task script not found: {exc}")
    
    return pending_tasks


def locate_script(attempt_dir: Path, preferred_name: Optional[str] = None) -> Path:
    """Return the `check_triton` script path for the attempt."""
    candidates: list[str] = []
    if preferred_name:
        candidates.append(preferred_name)
    candidates.extend(
        [
            "check_triton.py",
            "check_triton_dynamic.py",
            "check_triton_perf.py",
            "check_triton_dynamic_perf.py",
        ]
    )

    for name in candidates:
        candidate = attempt_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No check_triton script found under {attempt_dir}")


def describe_statuses(statuses: Iterable[str]) -> str:
    """Readable summary of statuses seen in the log."""
    if not statuses:
        return "no-status"
    return ",".join(sorted(statuses))


def resubmit_tasks(
    pending_tasks: List[TaskInfo],
    *,
    server: str,
    gpu_id: Optional[int],
    task_type: str,
    task_mode: Optional[str],
    no_perf: bool,
    dry_run: bool,
    limit: Optional[int],
    script_name: Optional[str],
    force: bool,
) -> None:
    """Resubmit pending tasks using NVGPU client."""
    client = NVGPUClient(server)

    if not dry_run and not client.health_check():
        print(f"[warn] NVGPU server {server} did not respond to health check, continuing anyway.")

    args = ["--no-perf"] if no_perf else []

    processed = 0
    for info in pending_tasks:
        if limit is not None and processed >= limit:
            break

        attempt_dir = info.attempt_dir
        if attempt_dir is None:
            print(f"[skip] {info.task_id}: missing attempt directory context (line {info.line_number})")
            continue

        try:
            script_path = locate_script(attempt_dir, script_name)
        except FileNotFoundError as exc:
            print(f"[skip] {info.task_id}: {exc}")
            continue

        human_label = info.attempt_label
        status_summary = describe_statuses(info.statuses)

        should_submit = True
        if not force:
            try:
                result: TaskResult = client.get_task(info.task_id)
                if result.status.lower() in {"pending", "queued", "running"}:
                    print(
                        f"[skip] {human_label} ({info.task_id}) still {result.status}, "
                        "use --force to resubmit anyway."
                    )
                    should_submit = False
                else:
                    print(
                        f"[skip] {human_label} ({info.task_id}) already {result.status}, "
                        "no resubmission needed."
                    )
                    should_submit = False
            except RuntimeError:
                # Task no longer exists -> proceed with resubmission.
                should_submit = True

        if not should_submit:
            continue

        if dry_run:
            print(f"[dry-run] Would resubmit {human_label} ({info.task_id}) -> {script_path}")
        else:
            new_task_id = client.submit_task_in_script_dir(
                script_path=str(script_path),
                task_mode=task_mode,
                task_type=task_type,
                args=args,
                gpu_id=gpu_id,
            )
            print(
                f"[submitted] {human_label}: {script_path} "
                f"(old task {info.task_id}, statuses={status_summary}) -> new task {new_task_id}"
            )

        processed += 1

    if processed == 0:
        print("[info] No tasks were resubmitted.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-submit NVGPU tasks that stayed pending after a server issue."
    )
    # Default to the same translation log path used previously.
    default_log = (
        PROJECT_ROOT
        / "server"
        / "nvgpu"
        / "debug"
        / "gpt_oss_20b_xpiler_extended"
        / "20251018_223123"
        / "gpt_oss_20b_xpiler_extended.log"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help=f"Path to the run log (default: {default_log})",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080",
        help="NVGPU server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Explicit GPU id to use; omit for auto assignment.",
    )
    parser.add_argument(
        "--task-type",
        default="functional",
        help="Task type to assign when resubmitting (default: functional).",
    )
    parser.add_argument(
        "--task-mode",
        choices=["shared", "exclusive"],
        default=None,
        help="Override task mode; defaults to NVGPU server behaviour.",
    )
    parser.add_argument(
        "--with-perf",
        dest="no_perf",
        action="store_false",
        help="Run without passing --no-perf to check_triton.py.",
    )
    parser.set_defaults(no_perf=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken without submitting jobs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Submit at most N tasks (useful for staged recovery).",
    )
    parser.add_argument(
        "--script-name",
        help="Explicit check_triton script name if it differs from the defaults.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resubmit even if the old task still exists on the NVGPU server.",
    )
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="List pending tasks (case/attempt) from the log and exit.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"Log file not found: {args.log}")

    pending_tasks = parse_pending_tasks(args.log)

    # Add the priority CUDA error task as the first task
    pending_tasks = add_priority_task(pending_tasks)
    
    if not pending_tasks:
        print("[info] No pending tasks detected in the log.")
        return

    if args.list_pending:
        print(f"[info] Listing {len(pending_tasks)} pending task(s) from {args.log}")
        shown_cases: set[str] = set()
        for info in pending_tasks:
            attempt_dir = info.attempt_dir or Path("<unknown>")
            case_label = info.case_name
            attempt_label = info.attempt_label
            status_summary = describe_statuses(info.statuses)
            print(
                f"- {attempt_label} | task_id={info.task_id} | statuses={status_summary} | path={attempt_dir}"
            )
            shown_cases.add(case_label)
        print("[cases] " + ", ".join(sorted(shown_cases)))
        return

    print(f"[info] Detected {len(pending_tasks)} pending task(s) in {args.log}")
    resubmit_tasks(
        pending_tasks,
        server=args.server,
        gpu_id=args.gpu,
        task_type=args.task_type,
        task_mode=args.task_mode,
        no_perf=args.no_perf,
        dry_run=args.dry_run,
        limit=args.limit,
        script_name=args.script_name,
        force=args.force,
    )


if __name__ == "__main__":
    main()
