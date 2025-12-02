from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Sequence

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT))

from ..clients.nvgpu import NVGPU_AVAILABLE, NVGPUClient  # type: ignore[attr-defined]


@dataclass(frozen=True)
class AttemptPlan:
    case_type: str
    case_name: str
    attempt_number: int
    metadata: dict
    script_path: Path

    @property
    def attempt_dir(self) -> Path:
        return self.script_path.parent

    @property
    def task_label(self) -> str:
        return f"perf/{self.case_type}/{self.case_name}/attempt_{self.attempt_number:02d}"


@dataclass(frozen=True)
class AttemptResult:
    plan: AttemptPlan
    task_id: str | None
    status: str
    success: bool
    log_path: Path
    latest_log_path: Path
    error_message: str | None = None
    nv_log: str | None = None


@dataclass(frozen=True)
class LogPaths:
    directory: Path
    session_log: Path
    latest_log: Path


class SessionLogger:
    def __init__(self, session_log_path: Path, latest_log_path: Path) -> None:
        self._lock = Lock()
        self._session_log_path = session_log_path
        self._latest_log_path = latest_log_path
        ensure_directory(session_log_path.parent)
        session_log_path.write_text("", encoding="utf-8")
        latest_log_path.write_text("", encoding="utf-8")

    def log(self, message: str = "") -> None:
        with self._lock:
            print(message)
            with self._session_log_path.open("a", encoding="utf-8") as fp:
                fp.write(message + "\n")
            copy_to_latest_log(self._session_log_path, self._latest_log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit NVGPU performance tests for successful Triton kernels."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
        help="Root directory containing run artifacts.",
    )
    parser.add_argument(
        "--stats-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "stats",
        help="Root directory containing stats outputs.",
    )
    parser.add_argument(
        "--project",
        default="cu2tri",
        help="Project name (first path segment under runs/stats).",
    )
    parser.add_argument(
        "--pipeline",
        default="xpiler",
        help="Pipeline name (second path segment under runs/stats).",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Model tag directory under pipeline (e.g. deepseek_v3_2_exp).",
    )
    parser.add_argument(
        "--all-model-tags",
        action="store_true",
        help="Process every model tag under the chosen pipeline.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Run timestamp directory (e.g. 20251024_075645).",
    )
    parser.add_argument(
        "--all-run-timestamps",
        action="store_true",
        help="Process every timestamp shared between runs/stats for each model tag.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=None,
        help="Optional explicit path to case_success.json. Overrides stats-root resolution.",
    )
    parser.add_argument(
        "--case-types",
        nargs="+",
        default=None,
        help="Restrict to selected case types (e.g. add conv2d).",
    )
    parser.add_argument(
        "--case-names",
        nargs="+",
        default=None,
        help="Restrict to exact case names.",
    )
    parser.add_argument(
        "--case-name-patterns",
        nargs="+",
        default=None,
        help="Restrict to fnmatch patterns (e.g. add_*_64).",
    )
    parser.add_argument(
        "--case-name-substrings",
        nargs="+",
        default=None,
        help="Restrict to case names containing any provided substring.",
    )
    parser.add_argument(
        "--attempts",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to the given attempt numbers (1-based).",
    )
    parser.add_argument(
        "--min-attempt",
        type=int,
        default=None,
        help="Minimum attempt number (inclusive).",
    )
    parser.add_argument(
        "--max-attempt",
        type=int,
        default=None,
        help="Maximum attempt number (inclusive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of attempts to execute after filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned attempts without contacting NVGPU.",
    )
    parser.add_argument(
        "--nvgpu-server",
        default="http://localhost:8080",
        help="NVGPU server base URL.",
    )
    parser.add_argument(
        "--nvgpu-gpu",
        type=int,
        default=None,
        help="Optional explicit GPU id for NVGPU submissions.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["shared", "exclusive"],
        default="shared",
        help="NVGPU task mode for submissions.",
    )
    parser.add_argument(
        "--task-type",
        choices=["functional", "performance", "both"],
        default="performance",
        help="NVGPU task type metadata.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between NVGPU task status polls.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help="Maximum seconds to wait for each NVGPU task (None waits indefinitely).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Directory for aggregated runner logs (perf_<ts>.log + perf.log).",
    )
    parser.add_argument(
        "--attempt-log-dir",
        type=Path,
        default=None,
        help="Optional directory to mirror per-attempt NVGPU logs instead of attempt_xx/logs/perf/.",
    )
    parser.add_argument(
        "--script-arg",
        action="append",
        default=None,
        help="Extra arguments forwarded to each check_triton_gpu_all.py invocation.",
    )
    parser.add_argument(
        "--session-tag",
        default=None,
        help="Optional custom label appended to log filenames.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=25,
        help="Maximum number of NVGPU tasks to run in parallel.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per attempt when submission or monitoring fails.",
    )
    args = parser.parse_args()
    return args


def detect_single_directory(base: Path) -> str:
    candidates = sorted(d.name for d in base.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No directories found under {base}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple directories found under {base}; specify one explicitly.")
    return candidates[0]


def detect_latest_timestamp(run_bucket: Path, stats_bucket: Path) -> str:
    run_candidates = {d.name for d in run_bucket.iterdir() if d.is_dir()}
    stats_candidates = {d.name for d in stats_bucket.iterdir() if d.is_dir()}
    common = sorted(run_candidates & stats_candidates)
    if not common:
        raise FileNotFoundError(
            f"No common timestamp directories between {run_bucket} and {stats_bucket}"
        )
    return common[-1]


def list_common_timestamps(run_bucket: Path, stats_bucket: Path) -> list[str]:
    run_candidates = {d.name for d in run_bucket.iterdir() if d.is_dir()}
    stats_candidates = {d.name for d in stats_bucket.iterdir() if d.is_dir()}
    common = sorted(run_candidates & stats_candidates)
    if not common:
        raise FileNotFoundError(
            f"No common timestamp directories between {run_bucket} and {stats_bucket}"
        )
    return common


def list_subdirectories(path: Path) -> list[str]:
    return sorted(d.name for d in path.iterdir() if d.is_dir())


def resolve_model_tags(
    model_bucket: Path, requested_tag: str | None, process_all: bool
) -> list[str]:
    if requested_tag:
        target_dir = model_bucket / requested_tag
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Model tag '{requested_tag}' not found under {model_bucket}")
        return [requested_tag]

    candidates = list_subdirectories(model_bucket)
    if not candidates:
        raise FileNotFoundError(f"No model tags found under {model_bucket}")

    if process_all:
        return candidates

    if len(candidates) == 1:
        return candidates

    raise ValueError(
        f"Multiple model tags found under {model_bucket}; specify --model-tag or enable --all-model-tags."
    )


def resolve_run_timestamps(
    run_bucket: Path,
    stats_bucket: Path,
    requested_timestamp: str | None,
    process_all: bool,
) -> list[str]:
    if requested_timestamp:
        run_dir = run_bucket / requested_timestamp
        stats_dir = stats_bucket / requested_timestamp
        if not run_dir.is_dir() or not stats_dir.is_dir():
            raise FileNotFoundError(
                f"Timestamp '{requested_timestamp}' missing under runs or stats for bucket {run_bucket}"
            )
        return [requested_timestamp]

    timestamps = list_common_timestamps(run_bucket, stats_bucket)
    if process_all:
        return timestamps
    return [timestamps[-1]]


def load_case_success(stats_file: Path) -> dict:
    with stats_file.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def match_case_name(
    case_name: str,
    case_names: set[str] | None,
    patterns: Sequence[str] | None,
    substrings: Sequence[str] | None,
) -> bool:
    if case_names and case_name not in case_names:
        return False
    if patterns and not any(fnmatch.fnmatch(case_name, pattern) for pattern in patterns):
        return False
    if substrings and not any(sub in case_name for sub in substrings):
        return False
    return True


def match_attempt_number(
    attempt_number: int,
    attempts: set[int] | None,
    min_attempt: int | None,
    max_attempt: int | None,
) -> bool:
    if attempts and attempt_number not in attempts:
        return False
    if min_attempt is not None and attempt_number < min_attempt:
        return False
    if max_attempt is not None and attempt_number > max_attempt:
        return False
    return True


def build_attempt_plan(
    stats: dict,
    run_dir: Path,
    case_types: set[str] | None,
    case_names: set[str] | None,
    case_patterns: Sequence[str] | None,
    case_substrings: Sequence[str] | None,
    attempts: set[int] | None,
    min_attempt: int | None,
    max_attempt: int | None,
) -> list[AttemptPlan]:
    plans: list[AttemptPlan] = []
    for case_type, cases in sorted(stats.items()):
        if case_types and case_type not in case_types:
            continue
        if not isinstance(cases, dict):
            continue
        for case_name, metadata in sorted(cases.items()):
            if not match_case_name(case_name, case_names, case_patterns, case_substrings):
                continue
            attempt_entries = metadata.get("attempts") or []
            for attempt_entry in attempt_entries:
                attempt_number = attempt_entry.get("attempt_number")
                if not isinstance(attempt_number, int):
                    continue
                if not attempt_entry.get("success", False):
                    continue
                if not match_attempt_number(
                    attempt_number,
                    attempts,
                    min_attempt,
                    max_attempt,
                ):
                    continue
                script_path = (
                    run_dir / case_name / f"attempt_{attempt_number:02d}" / "check_triton_gpu_all.py"
                )
                plans.append(
                    AttemptPlan(
                        case_type=case_type,
                        case_name=case_name,
                        attempt_number=attempt_number,
                        metadata=attempt_entry,
                        script_path=script_path,
                    )
                )
    return plans


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_perf_summary(stdout_text: str) -> list[str]:
    lines = stdout_text.splitlines()
    capture = False
    summary: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not capture:
            if "Performance Comparison" in stripped or stripped.startswith("Performance Test"):
                capture = True
                summary.append(stripped)
            continue
        if not stripped or stripped.startswith("===") or stripped.startswith("Summary"):
            break
        summary.append(stripped)
    return summary


def copy_to_latest_log(source: Path, latest_path: Path) -> None:
    shutil.copyfile(source, latest_path)


def build_log_paths(
    plan: AttemptPlan,
    session_stamp: str,
    override_dir: Path | None,
) -> LogPaths:
    if override_dir is None:
        directory = plan.attempt_dir / "logs" / "perf"
        ensure_directory(directory)
        return LogPaths(
            directory=directory,
            session_log=directory / f"perf_{session_stamp}.log",
            latest_log=directory / "perf.log",
        )

    directory = override_dir
    ensure_directory(directory)
    base = f"{plan.case_type}_{plan.case_name}_attempt_{plan.attempt_number:02d}"
    return LogPaths(
        directory=directory,
        session_log=directory / f"{base}_perf_{session_stamp}.log",
        latest_log=directory / f"{base}_perf.log",
    )


def run_single_attempt(
    plan: AttemptPlan,
    plan_index: int,
    total_plans: int,
    args: argparse.Namespace,
    session_stamp: str,
    script_args: list[str],
    session_logger: SessionLogger,
    attempt_log_override: Path | None,
) -> AttemptResult:
    client = NVGPUClient(args.nvgpu_server)
    log_paths = build_log_paths(plan, session_stamp, attempt_log_override)
    session_logger.log(
        f"--- [{plan_index}/{total_plans}] {plan.case_type}/{plan.case_name} "
        f"attempt {plan.attempt_number:02d} ---"
    )

    if not plan.script_path.exists():
        session_logger.log(f"Skipping: script not found at {plan.script_path}")
        with log_paths.session_log.open("w", encoding="utf-8") as log_fp:
            log_fp.write(f"Script not found at {plan.script_path}\n")
        copy_to_latest_log(log_paths.session_log, log_paths.latest_log)
        return AttemptResult(
            plan=plan,
            task_id=None,
            status="missing_script",
            success=False,
            log_path=log_paths.session_log,
            latest_log_path=log_paths.latest_log,
            error_message="script not found",
        )

    last_exception: Exception | None = None
    task_id: str | None = None
    result: TaskResult | None = None

    for retry_idx in range(1, args.max_retries + 1):
        prefix = f"[attempt {retry_idx}/{args.max_retries}] "
        try:
            task_id = client.submit_task_in_script_dir(
                script_path=str(plan.script_path),
                task_type=args.task_type,
                task_mode=args.task_mode,
                task_label=plan.task_label,
                args=script_args,
                gpu_id=args.nvgpu_gpu,
            )
            session_logger.log(prefix + f"Submitted task: {task_id} ({plan.task_label})")
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            session_logger.log(prefix + f"Submission failed: {exc}")
            if retry_idx == args.max_retries:
                with log_paths.session_log.open("w", encoding="utf-8") as log_fp:
                    log_fp.write(f"Submission failed: {exc}\n")
                copy_to_latest_log(log_paths.session_log, log_paths.latest_log)
                return AttemptResult(
                    plan=plan,
                    task_id=None,
                    status="submission_failed",
                    success=False,
                    log_path=log_paths.session_log,
                    latest_log_path=log_paths.latest_log,
                    error_message=str(exc),
                )
            continue

        try:
            result = client.wait_for_task(
                task_id,
                timeout=args.task_timeout,
                poll_interval=args.poll_interval,
            )
            session_logger.log(
                prefix + f"Task completed with status={result.status} exit_code={result.exit_code}"
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            session_logger.log(prefix + f"Task monitoring failed: {exc}")
            if retry_idx == args.max_retries:
                with log_paths.session_log.open("w", encoding="utf-8") as log_fp:
                    log_fp.write(f"Task monitoring failed: {exc}\n")
                copy_to_latest_log(log_paths.session_log, log_paths.latest_log)
                return AttemptResult(
                    plan=plan,
                    task_id=task_id,
                    status="monitor_failed",
                    success=False,
                    log_path=log_paths.session_log,
                    latest_log_path=log_paths.latest_log,
                    error_message=str(exc),
                )
            continue

    timings = []
    for label, value in (
        ("pending_ms", getattr(result, "pending_duration_ms", None)),
        ("queue_ms", getattr(result, "queue_duration_ms", None)),
        ("waiting_ms", getattr(result, "waiting_duration_ms", None)),
        ("running_ms", getattr(result, "running_duration_ms", None)),
        ("total_ms", getattr(result, "total_duration_ms", None)),
    ):
        if value is not None:
            timings.append(f"{label}={value:.3f}")
    if timings:
        session_logger.log("Timings: " + ", ".join(timings))

    nv_log = getattr(result, "log_file", None)
    if nv_log:
        session_logger.log(f"NVGPU log: {nv_log}")
    else:
        session_logger.log("NVGPU log path unavailable.")

    stdout_text = ""
    stderr_text = ""
    try:
        stdout_text = client.get_full_task_log(task_id, log_type="stdout")
    except Exception as exc:  # noqa: BLE001
        session_logger.log(f"Failed to fetch stdout: {exc}")
    try:
        stderr_text = client.get_full_task_log(task_id, log_type="stderr")
    except Exception as exc:  # noqa: BLE001
        session_logger.log(f"Failed to fetch stderr: {exc}")

    perf_summary = extract_perf_summary(stdout_text)
    if perf_summary:
        session_logger.log("Performance Summary:")
        for line in perf_summary:
            session_logger.log(f"  {line}")
    else:
        session_logger.log("Performance summary was not detected in stdout.")

    nv_log_copied = False
    if nv_log:
        nv_log_path = Path(nv_log)
        if nv_log_path.is_file():
            try:
                shutil.copyfile(nv_log_path, log_paths.session_log)
                nv_log_copied = True
            except OSError as exc:
                session_logger.log(f"Failed to copy NVGPU log: {exc}")
        else:
            session_logger.log(f"NVGPU log path not found: {nv_log_path}")

    if not nv_log_copied:
        with log_paths.session_log.open("w", encoding="utf-8") as log_fp:
            log_fp.write(stdout_text or "NVGPU log unavailable.\n")
            if stderr_text.strip():
                log_fp.write("\n=== STDERR ===\n")
                log_fp.write(stderr_text)

    copy_to_latest_log(log_paths.session_log, log_paths.latest_log)

    success = result.status == "completed" and result.exit_code == 0
    session_logger.log("Result: " + ("SUCCESS" if success else "FAILED"))

    return AttemptResult(
        plan=plan,
        task_id=result.task_id if "result" in locals() else None,
        status=getattr(result, "status", "unknown"),
        success=success,
        log_path=log_paths.session_log,
        latest_log_path=log_paths.latest_log,
        error_message=None if success else "task failed",
        nv_log=nv_log,
    )


def process_single_run(
    *,
    args: argparse.Namespace,
    session_stamp: str,
    session_logger: SessionLogger,
    script_args: list[str],
    case_types: set[str] | None,
    case_names: set[str] | None,
    attempts: set[int] | None,
    run_dir: Path,
    stats_file: Path,
    model_tag: str,
    run_timestamp: str,
    attempt_log_override: Path | None,
) -> tuple[int, int]:
    session_logger.log("")
    session_logger.log(f"=== Model: {model_tag} | Run: {run_timestamp} ===")
    session_logger.log(f"Runs dir     : {run_dir}")
    session_logger.log(f"Stats file   : {stats_file}")

    try:
        stats_data = load_case_success(stats_file)
    except FileNotFoundError as exc:
        session_logger.log(f"Stats file not found: {exc}")
        return 1, 0

    plans = build_attempt_plan(
        stats=stats_data,
        run_dir=run_dir,
        case_types=case_types,
        case_names=case_names,
        case_patterns=args.case_name_patterns,
        case_substrings=args.case_name_substrings,
        attempts=attempts,
        min_attempt=args.min_attempt,
        max_attempt=args.max_attempt,
    )

    plans.sort(key=lambda p: (p.case_type, p.case_name, p.attempt_number))
    if args.limit is not None:
        plans = plans[: args.limit]

    if not plans:
        session_logger.log("No attempts match the provided filters for this run.")
        return 0, 0

    session_logger.log(f"Planned attempts (run): {len(plans)}")
    if args.dry_run:
        for plan in plans:
            session_logger.log(
                f"[DRY-RUN] {plan.case_type}/{plan.case_name} attempt "
                f"{plan.attempt_number:02d} -> {plan.script_path}"
            )
        return 0, 0

    total_attempts = len(plans)
    results: list[AttemptResult] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_map = {
            executor.submit(
                run_single_attempt,
                plan,
                idx,
                total_attempts,
                args,
                session_stamp,
                script_args,
                session_logger,
                attempt_log_override,
            ): plan
            for idx, plan in enumerate(plans, start=1)
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)

    results.sort(key=lambda r: (r.plan.case_type, r.plan.case_name, r.plan.attempt_number))
    success_count = sum(1 for result in results if result.success)
    failure_count = len(results) - success_count

    session_logger.log("--- Run Summary ---")
    session_logger.log(f"Attempts run  : {len(results)}")
    session_logger.log(f"Successful    : {success_count}")
    session_logger.log(f"Failed        : {failure_count}")

    for result in results:
        status = "SUCCESS" if result.success else "FAIL"
        session_logger.log(
            f"{status:7} | {result.plan.case_type}/{result.plan.case_name} "
            f"attempt {result.plan.attempt_number:02d} | {result.log_path}"
        )

    return (0 if failure_count == 0 else 1, len(results))


def main() -> int:
    args = parse_args()

    if args.max_concurrency < 1:
        print("--max-concurrency must be >= 1")
        return 1

    if not NVGPU_AVAILABLE:
        print("NVGPU client dependency is unavailable. Ensure PROJECT_ROOT is set correctly.")
        return 1

    runs_root = args.runs_root.expanduser().resolve()
    stats_root = args.stats_root.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    attempt_log_override = (
        args.attempt_log_dir.expanduser().resolve() if args.attempt_log_dir else None
    )
    ensure_directory(log_dir)

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.session_tag:
        session_stamp = f"{session_stamp}_{args.session_tag}"

    session_log_path = log_dir / f"perf_{session_stamp}.log"
    latest_session_log_path = log_dir / "perf.log"
    session_logger = SessionLogger(session_log_path, latest_session_log_path)

    def finalize(exit_code: int) -> int:
        return exit_code

    if args.stats_file and args.all_model_tags:
        session_logger.log("Cannot combine --stats-file with --all-model-tags.")
        return finalize(1)
    if args.stats_file and args.all_run_timestamps:
        session_logger.log("Cannot combine --stats-file with --all-run-timestamps.")
        return finalize(1)

    case_types = set(args.case_types) if args.case_types else None
    case_names = set(args.case_names) if args.case_names else None
    attempts = set(args.attempts) if args.attempts else None

    script_args = args.script_arg or []
    project_label = f"{args.project}/{args.pipeline}"

    header_lines = [
        f"=== Performance Session {session_stamp} ===",
        f"Project/Pipe : {project_label}",
        f"NVGPU server : {args.nvgpu_server}",
        f"Task type    : {args.task_type}",
        f"Task mode    : {args.task_mode}",
        f"GPU override : {args.nvgpu_gpu if args.nvgpu_gpu is not None else 'auto'}",
        "Filters      : "
        f"case_types={case_types or '*'}, "
        f"case_names={case_names or '*'}, "
        f"patterns={args.case_name_patterns or '*'}, "
        f"substr={args.case_name_substrings or '*'}, "
        f"attempts={attempts or '*'}, "
        f"min_attempt={args.min_attempt or '-'}, "
        f"max_attempt={args.max_attempt or '-'}",
        f"Dry run      : {args.dry_run}",
        f"all_model_tags={args.all_model_tags}, all_run_timestamps={args.all_run_timestamps}",
        "=========================================",
    ]
    for line in header_lines:
        session_logger.log(line)

    model_bucket_root = runs_root / args.project / args.pipeline
    stats_bucket_root = stats_root / args.project / args.pipeline

    try:
        model_tags = resolve_model_tags(model_bucket_root, args.model_tag, args.all_model_tags)
    except (FileNotFoundError, ValueError) as exc:
        session_logger.log(str(exc))
        return finalize(1)

    health_client = NVGPUClient(args.nvgpu_server)
    if not health_client.health_check():
        session_logger.log(f"NVGPU server {args.nvgpu_server} is unreachable.")
        return finalize(1)

    stats_file_override = args.stats_file.expanduser().resolve() if args.stats_file else None

    overall_exit = 0
    run_combinations = 0
    attempt_counter = 0

    for model_tag in model_tags:
        run_bucket_model = model_bucket_root / model_tag
        stats_bucket_model = stats_bucket_root / model_tag

        try:
            run_timestamps = resolve_run_timestamps(
                run_bucket_model,
                stats_bucket_model,
                args.run_timestamp,
                args.all_run_timestamps,
            )
        except (FileNotFoundError, ValueError) as exc:
            session_logger.log(str(exc))
            overall_exit = 1
            continue

        for run_timestamp in run_timestamps:
            run_dir = run_bucket_model / run_timestamp
            if not run_dir.is_dir():
                session_logger.log(f"Run directory not found: {run_dir}")
                overall_exit = 1
                continue

            stats_dir = stats_bucket_model / run_timestamp
            if not stats_file_override and not stats_dir.is_dir():
                session_logger.log(f"Stats directory not found: {stats_dir}")
                overall_exit = 1
                continue

            stats_file = stats_file_override or (stats_dir / "case_success.json")

            exit_code, attempts_ran = process_single_run(
                args=args,
                session_stamp=session_stamp,
                session_logger=session_logger,
                script_args=script_args,
                case_types=case_types,
                case_names=case_names,
                attempts=attempts,
                run_dir=run_dir,
                stats_file=stats_file,
                model_tag=model_tag,
                run_timestamp=run_timestamp,
                attempt_log_override=attempt_log_override,
            )

            run_combinations += 1
            attempt_counter += attempts_ran
            if exit_code != 0:
                overall_exit = 1

    session_logger.log("")
    session_logger.log("=== Overall Summary ===")
    session_logger.log(f"Model tags processed : {len(model_tags)}")
    session_logger.log(f"Run combinations     : {run_combinations}")
    session_logger.log(f"Attempts executed    : {attempt_counter}")
    session_logger.log(f"Overall status       : {'SUCCESS' if overall_exit == 0 else 'FAIL'}")

    if run_combinations == 0:
        session_logger.log("No run combinations were executed.")

    return finalize(overall_exit)


if __name__ == "__main__":
    sys.exit(main())

