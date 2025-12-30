from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..clients import NVGPU_AVAILABLE, NVGPUClient
from ..core.runtime import RuntimeContext
from ..io.jsonl import write_jsonl_log
from ..utils.formatting import format_ms
import shutil
from utils.task_refs import format_task_ref
from utils.timezone import ensure_timezone, now_timestamp


def _perf_json_path(test_work_dir: Path) -> Path:
    return test_work_dir / "perf.json"


def _perf_config_path() -> Path:
    """Location of the perf configuration YAML (direction/testset → model_tag/timestamps)."""
    # services/perf.py → llm_trans/ → config/perf_runs.yaml
    return Path(__file__).resolve().parents[1] / "config" / "perf_runs.yaml"


def load_perf_config(path: Path | None = None) -> Mapping[str, object]:
    """Load performance configuration from YAML.

    Expected structure::

        cu2tri:
          xpiler:
            model_tag:
              - 20250101_000000
              - 20250102_000000

    Returns an empty mapping if the file is missing or invalid.
    """
    cfg_path = path or _perf_config_path()
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        if not isinstance(data, Mapping):
            return {}
        return data
    except Exception:
        return {}


@dataclass(frozen=True)
class PerfRunRecord:
    """Single perf run record resolved from config/perf_runs.yaml."""

    direction: str
    testset: str
    model_tag: str
    run_timestamp: str


def resolve_perf_records_from_config(
    *,
    direction: str,
    testset: str,
    config: Mapping[str, object] | None = None,
) -> list[PerfRunRecord]:
    """Resolve all configured perf run records for the given direction/testset."""
    data = config or load_perf_config()
    dir_entry = data.get(direction)
    if not isinstance(dir_entry, Mapping):
        return []
    testset_entry = dir_entry.get(testset)
    if not isinstance(testset_entry, Mapping):
        return []

    records: list[PerfRunRecord] = []
    for model_tag, timestamps in testset_entry.items():
        if not isinstance(model_tag, str):
            continue
        if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes)):
            for ts in timestamps:
                # 支持字符串和整数时间戳；如果是纯数字且长度为14，则自动补上日期/时间之间的下划线。
                if isinstance(ts, (str, int)):
                    ts_str = str(ts)
                    if "_" not in ts_str and len(ts_str) == 14:
                        ts_str = f"{ts_str[:8]}_{ts_str[8:]}"
                    records.append(
                        PerfRunRecord(
                            direction=direction,
                            testset=testset,
                            model_tag=model_tag,
                            run_timestamp=ts_str,
                        )
                    )
    return records


def _format_task_progress(result: Any, ms_format: bool = False) -> str:
    parts: list[str] = []
    if getattr(result, "total_duration_ms", None) is not None:
        parts.append(f"total={format_ms(result.total_duration_ms, ms_format)}")
    else:
        if getattr(result, "pending_duration_ms", None) is not None:
            parts.append(f"pending={format_ms(result.pending_duration_ms, ms_format)}")
        if getattr(result, "queue_duration_ms", None) is not None:
            parts.append(f"queue={format_ms(result.queue_duration_ms, ms_format)}")
        if getattr(result, "waiting_duration_ms", None) is not None:
            parts.append(f"waiting={format_ms(result.waiting_duration_ms, ms_format)}")
        if getattr(result, "running_duration_ms", None) is not None:
            parts.append(f"running={format_ms(result.running_duration_ms, ms_format)}")
    return ", ".join(parts)


def _extract_server_times(result: Any) -> dict[str, Any]:
    return {
        "pending_ms": getattr(result, "pending_duration_ms", None),
        "queue_ms": getattr(result, "queue_duration_ms", None),
        "waiting_ms": getattr(result, "waiting_duration_ms", None),
        "running_ms": getattr(result, "running_duration_ms", None),
        "total_ms": getattr(result, "total_duration_ms", None),
        "assigned_gpu": getattr(result, "gpu_id", getattr(result, "assigned_gpu", None)),
        "status": getattr(result, "status", None),
        "exit_code": getattr(result, "exit_code", None),
        "task_id": getattr(result, "task_id", None),
        "submit_time": getattr(result, "submit_timestamp", getattr(result, "submit_time", None)),
        "start_time": getattr(result, "start_timestamp", getattr(result, "start_time", None)),
        "end_time": getattr(result, "end_timestamp", getattr(result, "end_time", None)),
    }


def _parse_perf_results(perf_file: Path, stdout_fallback: str) -> dict[str, Any]:
    """Parse performance results from perf.json file or stdout fallback."""
    if perf_file.exists():
        try:
            return json.loads(perf_file.read_text(encoding="utf-8"))
        except Exception:
            # Fallback to parsing stdout if file is not valid JSON
            pass

    lines = [ln for ln in stdout_fallback.splitlines() if ln.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass

    return {}


async def _run_nvgpu_task(
    context: RuntimeContext,
    *,
    client: NVGPUClient,
    script_path: Path,
    work_dir: Path,
    log_file: Path,
    task_label: str,
    task_args: list[str],
    gpu_id: int | None,
) -> tuple[dict[str, Any], bool, str, str, str]:
    """Submit a single NVGPU task and wait for completion."""
    logger = context.logger
    args = context.args

    logger.info("Submitting NVGPU perf task '%s' using script %s with args %s", task_label, script_path, task_args)

    task_id = client.submit_task_in_script_dir(
        script_path=str(script_path),
        task_type="performance",
        task_label=task_label,
        args=task_args,
        gpu_id=gpu_id,
    )
    logger.info("Perf task submitted: %s", format_task_ref(task_id, include_label=False))

    last_status: str | None = None
    while True:
        result = client.get_task(task_id)
        status = getattr(result, "status", None)
        if status != last_status:
            progress = _format_task_progress(result, getattr(args, "ms_format", False))
            ref = format_task_ref(result, include_label=True)
            if progress:
                logger.info("[perf]::%s: %s (%s)", ref, status, progress)
            else:
                logger.info("[perf]::%s: %s", ref, status)
            last_status = status

        if status in {"completed", "failed", "cancelled"}:
            break

        await asyncio.sleep(2)

    server_times = _extract_server_times(result)

    try:
        stdout_text = client.get_full_task_log(task_id, log_type="stdout")
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Failed to fetch stdout for perf task %s: %s", task_id, exc)
        stdout_text = ""

    try:
        stderr_text = client.get_full_task_log(task_id, log_type="stderr")
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Failed to fetch stderr for perf task %s: %s", task_id, exc)
        stderr_text = ""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as fp:
        fp.write(f"=== NVGPU Perf Task ===\n")
        fp.write(f"Task: {format_task_ref(result, include_label=True)}\n")
        fp.write(f"Status: {server_times.get('status')}\n")
        fp.write(f"Exit code: {server_times.get('exit_code')}\n")
        fp.write("\n=== Timing (ms) ===\n")
        for key in ["pending_ms", "queue_ms", "waiting_ms", "running_ms", "total_ms"]:
            value = server_times.get(key)
            if value is not None:
                fp.write(f"{key}: {value}\n")
        fp.write("\n=== STDOUT ===\n")
        fp.write(stdout_text)
        fp.write("\n=== STDERR ===\n")
        fp.write(stderr_text)
        fp.write("\n=== END OF LOG ===\n")

    success = (
        server_times.get("status") == "completed"
        and (server_times.get("exit_code") is None or server_times.get("exit_code") == 0)
    )

    if not success:
        logger.error(
            "NVGPU perf task failed: status=%s exit=%s task=%s",
            server_times.get("status"),
            server_times.get("exit_code"),
            format_task_ref(result, include_label=False),
        )

    return server_times, success, stdout_text, stderr_text, task_id


async def run_perf_nvgpu(
    context: RuntimeContext,
    test_work_dir: Path,
    *,
    gpu_id: int | None = None,
    case_tag: str | None = None,
    shape_tag: str | None = None,
    warmup: int | None = None,
    iters: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run performance test using NVGPU for a single translated case."""
    logger = context.logger
    args = context.args

    if not (context.settings.use_nvgpu and context.nvgpu_available and NVGPU_AVAILABLE):
        raise RuntimeError("NVGPU is not available for performance testing")

    direction = context.settings.direction
    if direction == "tri2cute":
        script_name = "check_cute.py"
    else:
        script_name = f"check_triton{context.settings.check_suffix}.py"

    script_path = test_work_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Perf script not found: {script_path}")

    client = NVGPUClient(args.nvgpu_server)
    if not client.health_check():
        raise RuntimeError(f"NVGPU server at {args.nvgpu_server} is not responding")

    perf_out = _perf_json_path(test_work_dir)
    task_args: list[str] = [str(script_path.absolute())]

    logs_dir = test_work_dir / "logs" / "perf"
    log_file = logs_dir / "perf.log"
    resolved_gpu_id = gpu_id if gpu_id is not None else getattr(args, "nvgpu_gpu", None)
    label = f"[perf]::[{case_tag}]"

    server_times, success, stdout_text, _, _ = await _run_nvgpu_task(
        context,
        client=client,
        script_path=script_path,
        work_dir=test_work_dir,
        log_file=log_file,
        task_label=label,
        task_args=task_args,
        gpu_id=resolved_gpu_id,
    )

    if not success:
        raise RuntimeError(f"NVGPU perf task failed with status {server_times.get('status')}")

    perf_data = _parse_perf_results(perf_out, stdout_text)

    exec_ms = server_times.get("running_ms") or server_times.get("total_ms") or 0.0
    logger.info(
        "NVGPU perf completed for %s: %s",
        case_tag or "perf",
        format_ms(exec_ms, getattr(args, "ms_format", False)),
    )

    if shape_tag:
        perf_data["shape_tag"] = shape_tag
    if case_tag:
        perf_data["case_tag"] = case_tag

    return perf_data, server_times


async def record_perf_result(
    context: RuntimeContext,
    test_work_dir: Path,
    perf_data: dict[str, Any],
    server_times: dict[str, Any] | None = None,
) -> None:
    """Record performance results to perf.json and JSONL log."""
    logger = context.logger

    if not perf_data:
        logger.warning("No performance data to record")
        return

    out_path = _perf_json_path(test_work_dir)
    out_path.write_text(
        json.dumps(perf_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.debug("Performance results saved to %s", out_path)

    record: dict[str, Any] = {
        "_type": "perf",
        "timestamp": ensure_timezone(now_timestamp()).isoformat(),
        "temperature": getattr(context.args, "temperature", None),
        **perf_data,
    }

    if server_times:
        record["server_times_ms"] = server_times
        exec_time = (
            server_times.get("running_ms")
            or server_times.get("total_ms")
            or server_times.get("execution_ms")
        )
        if exec_time:
            logger.info(
                "Performance test completed in %s",
                format_ms(exec_time, getattr(context.args, "ms_format", False)),
            )

    await write_jsonl_log(context, record)


async def run(context: RuntimeContext) -> dict[str, Any]:
    """Run config-driven NVGPU perf session using config/perf_runs.yaml for xpiler runs."""
    logger = context.logger
    args = context.args

    if not NVGPU_AVAILABLE:
        raise RuntimeError("NVGPU client dependency is unavailable")

    nvgpu_server = getattr(args, "nvgpu_server", None)
    if not nvgpu_server:
        raise RuntimeError("NVGPU server is not configured (missing --nvgpu-server).")

    direction = context.settings.direction
    testset = args.testset

    cfg = load_perf_config()
    records: list[PerfRunRecord] = resolve_perf_records_from_config(
        direction=direction,
        testset=testset,
        config=cfg,
    )
    if not records:
        logger.warning("No perf records found in config/perf_runs.yaml for %s/%s", direction, testset)
        return {"success": False, "attempts": 0, "successful_attempts": 0, "runs": 0}

    client = NVGPUClient(nvgpu_server)
    if not client.health_check():
        raise RuntimeError(f"NVGPU server {nvgpu_server} is unreachable")

    project_root = context.settings.project_root
    runs_root = project_root / "cu2til" / "llm_trans" / "runs"
    stats_root = project_root / "cu2til" / "llm_trans" / "stats"

    # 收集所有需要跑 perf 的 attempt，后面用 asyncio 并发执行
    jobs: list[dict[str, Any]] = []

    for record in records:
        run_dir = runs_root / record.direction / record.testset / record.model_tag / record.run_timestamp
        stats_file = (
            stats_root
            / record.direction
            / record.testset
            / record.model_tag
            / record.run_timestamp
            / "case_success.json"
        )

        if not stats_file.exists():
            logger.warning("Stats file not found for %s: %s", record.model_tag, stats_file)
            continue

        try:
            stats_data = json.loads(stats_file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("Failed to load stats file %s: %s", stats_file, exc)
            continue

        for case_type, cases in sorted(stats_data.items()):
            if not isinstance(cases, dict):
                continue
            for case_name, meta in sorted(cases.items()):
                attempts = meta.get("attempts") or []
                for attempt_entry in attempts:
                    if not attempt_entry.get("success", False):
                        continue
                    attempt_number = attempt_entry.get("attempt_number")
                    if not isinstance(attempt_number, int):
                        continue

                    attempt_dir = run_dir / case_name / f"attempt_{attempt_number:02d}"
                    script_path = attempt_dir / "check_triton_gpu_all.py"
                    if not script_path.exists():
                        logger.warning("Perf script missing for %s/%s attempt %d", case_type, case_name, attempt_number)
                        continue

                    logs_dir = attempt_dir / "logs" / "perf"
                    logs_dir.mkdir(parents=True, exist_ok=True)

                    jobs.append(
                        {
                            "case_type": case_type,
                            "case_name": case_name,
                            "attempt_number": attempt_number,
                            "attempt_dir": attempt_dir,
                            "script_path": script_path,
                            "logs_dir": logs_dir,
                            "model_tag": record.model_tag,
                        }
                    )

    if not jobs:
        logger.warning("No successful attempts found for perf in the configured runs")
        return {"success": False, "attempts": 0, "successful_attempts": 0, "runs": len(records)}

    max_concurrency = getattr(args, "perf_concurrency", 4)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single_job(job: dict[str, Any]) -> bool:
        case_type = job["case_type"]
        case_name = job["case_name"]
        attempt_number = job["attempt_number"]
        attempt_dir: Path = job["attempt_dir"]
        script_path: Path = job["script_path"]
        logs_dir: Path = job["logs_dir"]

        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log = logs_dir / f"perf_{session_stamp}.log"
        latest_log = logs_dir / "perf.log"
        label = f"{case_name}|at@{attempt_number:02d}"
        gpu_override = getattr(args, "nvgpu_gpu", None)

        async with semaphore:
            server_times, success, _, _, _ = await _run_nvgpu_task(
                context,
                client=client,
                script_path=script_path,
                work_dir=attempt_dir,
                log_file=session_log,
                task_label=label,
                task_args=[],
                gpu_id=gpu_override,
            )

        # 维护一份无时间戳的最新日志
        try:
            shutil.copyfile(session_log, latest_log)
        except OSError:
            pass

        if not success:
            logger.warning(
                "Perf attempt failed for %s/%s attempt %d: status=%s exit=%s",
                case_type,
                case_name,
                attempt_number,
                server_times.get("status"),
                server_times.get("exit_code"),
            )
        return success

    tasks = [asyncio.create_task(run_single_job(job)) for job in jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_attempts = len(jobs)
    success_attempts = 0
    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            logger.error(
                "Perf attempt raised exception for %s/%s attempt %d: %s",
                job["case_type"],
                job["case_name"],
                job["attempt_number"],
                result,
            )
        elif result:
            success_attempts += 1

    logger.info(
        "Perf session finished: %d/%d attempts succeeded across %d runs",
        success_attempts,
        total_attempts,
        len(records),
    )

    return {
        "success": total_attempts > 0 and success_attempts == total_attempts,
        "attempts": total_attempts,
        "successful_attempts": success_attempts,
        "runs": len(records),
    }


__all__ = [
    "run_perf_nvgpu",
    "record_perf_result",
    "run",
]
