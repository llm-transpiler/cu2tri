from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..clients import NVGPU_AVAILABLE, NVGPUClient
from ..core.runtime import RuntimeContext
from ..io.jsonl import write_jsonl_log
from utils.task_refs import format_task_ref
from utils.timezone import now_timestamp, parse_timestamp
from ..utils.formatting import format_ms


def _perf_json_path(test_work_dir: Path) -> Path:
    return test_work_dir / "perf.json"


async def run_perf_local(
    context: RuntimeContext,
    test_work_dir: Path,
    *,
    case_tag: Optional[str] = None,
    shape_tag: Optional[str] = None,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
) -> Dict[str, Any]:
    """Run performance test locally using check_triton script."""
    logger = context.logger
    if context.settings.direction == "tri2cute":
        script = "check_cute.py"
    else:
        script = f"check_triton{context.settings.check_suffix}.py"
    script_path = test_work_dir / script
    if not script_path.exists():
        raise FileNotFoundError(f"Perf local: script not found: {script_path}")

    perf_out = _perf_json_path(test_work_dir)
    cmd = [
        sys.executable,
        script,
        "--perf-only",
        "--perf-json-out",
        str(perf_out),
    ]
    if context.settings.direction == "tri2cute":
        cmd += ["--target-gpu", context.settings.target_gpu]
    if case_tag:
        cmd += ["--case-tag", str(case_tag)]
    if shape_tag:
        cmd += ["--shape-tag", str(shape_tag)]
    if warmup is not None:
        cmd += ["--perf-warmup", str(warmup)]
    if iters is not None:
        cmd += ["--perf-iters", str(iters)]

    logger.info("Running local perf: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(test_work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_data, stderr_data = await proc.communicate()
    if proc.returncode != 0:
        logger.error("Perf local failed: %s", stderr_data.decode("utf-8", errors="ignore"))
        raise RuntimeError("Local perf failed")

    # Parse performance results
    data = _parse_perf_results(perf_out, stdout_data.decode("utf-8", errors="ignore"))
    return data


async def run_perf_nvgpu(
    context: RuntimeContext,
    test_work_dir: Path,
    *,
    gpu_id: Optional[int] = None,
    case_tag: Optional[str] = None,
    shape_tag: Optional[str] = None,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run performance test using NVGPU server with proper task handling."""
    logger = context.logger

    # Check NVGPU availability
    if not NVGPU_AVAILABLE or NVGPUClient is None:
        logger.warning("NVGPU client unavailable, falling back to local perf")
        data = await run_perf_local(context, test_work_dir, case_tag=case_tag, shape_tag=shape_tag, warmup=warmup, iters=iters)
        return data, {}

    # Initialize client
    client = NVGPUClient(context.args.nvgpu_server)
    if not client.health_check():
        logger.warning("NVGPU server unhealthy, using local perf")
        data = await run_perf_local(context, test_work_dir, case_tag=case_tag, shape_tag=shape_tag, warmup=warmup, iters=iters)
        return data, {}

    # Determine script name based on direction
    if context.settings.direction == "tri2cute":
        script_name = "check_cute.py"
    else:
        script_name = f"check_triton{context.settings.check_suffix}.py"

    script_path = test_work_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Perf NVGPU: script not found: {script_path}")

    script = str(script_path.absolute())
    perf_out = _perf_json_path(test_work_dir)

    # Build task arguments for performance testing
    task_args = [
        "--perf-only",
        "--perf-json-out", str(perf_out.absolute())
    ]

    if context.settings.direction == "tri2cute":
        task_args += ["--target-gpu", context.settings.target_gpu]
    if case_tag:
        task_args += ["--case-tag", str(case_tag)]
    if shape_tag:
        task_args += ["--shape-tag", str(shape_tag)]
    if warmup is not None:
        task_args += ["--perf-warmup", str(warmup)]
    if iters is not None:
        task_args += ["--perf-iters", str(iters)]

    # Submit performance task to NVGPU server
    logger.info("Submitting NVGPU perf task (exclusive mode) for %s", case_tag or "performance test")

    # Use the correct submit_task method with work_dir parameter
    task_id = client.submit_task(
        script_path=script,
        task_mode="exclusive",  # Performance tasks need exclusive GPU access
        task_type="performance",
        task_label=f"{case_tag or 'perf'}_performance",
        work_dir=str(test_work_dir.absolute()),
        args=task_args,
        gpu_id=gpu_id if gpu_id is not None else context.args.nvgpu_gpu,
    )

    logger.info("Perf task submitted: %s", format_task_ref(task_id, include_label=False))

    # Monitor task progress
    last_status = None
    start_time = now_timestamp()

    while True:
        result = client.get_task(task_id)

        # Log status changes
        if result.status != last_status:
            progress_info = _format_task_progress(result, context.args.ms_format)
            task_ref = format_task_ref(result, include_label=True)
            logger.info("[perf] %s: %s%s", task_ref, result.status, f" ({progress_info})" if progress_info else "")
            last_status = result.status

        # Check if task is complete
        if result.status in ["completed", "failed", "cancelled"]:
            break

        await asyncio.sleep(2)  # Poll every 2 seconds

    # Handle task failure
    if result.status != "completed" or (result.exit_code is not None and result.exit_code != 0):
        stdout = client.get_full_task_log(task_id, log_type="stdout")
        stderr = client.get_full_task_log(task_id, log_type="stderr")
        logger.error(
            "Perf NVGPU task failed: status=%s, exit=%s, task=%s\nSTDERR:%s\nSTDOUT:%s",
            result.status,
            result.exit_code,
            format_task_ref(result),
            stderr[-1000:] if stderr else "",
            stdout[-1000:] if stdout else "",
        )
        raise RuntimeError(f"NVGPU perf task failed with status {result.status}")

    # Parse performance results
    perf_data = _parse_perf_results(perf_out, client.get_full_task_log(task_id, log_type="stdout"))

    # Extract server timing information
    server_times = _extract_server_times(result, start_time)

    logger.info("NVGPU perf completed successfully in %s", format_ms(server_times.get("execution_ms", 0), context.args.ms_format))

    return perf_data, server_times


def _parse_perf_results(perf_file: Path, stdout_fallback: str) -> Dict[str, Any]:
    """Parse performance results from perf.json file or stdout fallback."""
    if perf_file.exists():
        try:
            return json.loads(perf_file.read_text(encoding="utf-8"))
        except Exception as e:
            # Fallback to parsing stdout
            pass

    # Parse JSON from stdout (last line)
    lines = [ln for ln in stdout_fallback.splitlines() if ln.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass

    # Return empty performance data if parsing fails
    return {}


def _format_task_progress(result: Any, ms_format: bool = False) -> str:
    """Format task progress information."""
    parts = []

    if result.total_duration_ms is not None:
        parts.append(f"total={format_ms(result.total_duration_ms, ms_format)}")
    else:
        if result.pending_duration_ms is not None:
            parts.append(f"pending={format_ms(result.pending_duration_ms, ms_format)}")
        if result.queue_duration_ms is not None:
            parts.append(f"queue={format_ms(result.queue_duration_ms, ms_format)}")
        if result.waiting_duration_ms is not None:
            parts.append(f"waiting={format_ms(result.waiting_duration_ms, ms_format)}")
        if result.running_duration_ms is not None:
            parts.append(f"running={format_ms(result.running_duration_ms, ms_format)}")

    return ", ".join(parts)


def _extract_server_times(result: Any, start_time: Any) -> Dict[str, Any]:
    """Extract timing information from NVGPU server result."""
    server_times = {
        "pending_ms": result.pending_duration_ms,
        "queue_ms": result.queue_duration_ms,
        "waiting_ms": result.waiting_duration_ms,
        "running_ms": result.running_duration_ms,
        "total_ms": result.total_duration_ms,
        "execution_ms": result.execution_duration_ms or result.running_duration_ms,
        "assigned_gpu": result.gpu_id,
        "status": result.status,
        "exit_code": result.exit_code,
        "task_id": result.task_id,
    }

    # Add timestamp information
    if hasattr(result, 'submit_timestamp'):
        server_times["submit_time"] = result.submit_timestamp
    if hasattr(result, 'start_timestamp'):
        server_times["start_time"] = result.start_timestamp
    if hasattr(result, 'end_timestamp'):
        server_times["end_time"] = result.end_timestamp

    return server_times


async def record_perf_result(
    context: RuntimeContext,
    test_work_dir: Path,
    perf_data: Dict[str, Any],
    server_times: Dict[str, Any] | None = None,
) -> None:
    """Record performance results to JSON file and JSONL log."""
    logger = context.logger

    # Validate performance data
    if not perf_data:
        logger.warning("No performance data to record")
        return

    # Persist perf.json (overwrite to keep latest)
    out_path = _perf_json_path(test_work_dir)
    out_path.write_text(
        json.dumps(perf_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    logger.debug("Performance results saved to %s", out_path)

    # Create comprehensive record for JSONL logging
    record: Dict[str, Any] = {
        "_type": "perf",
        "timestamp": now_timestamp().isoformat(),
        "temperature": getattr(context.args, 'temperature', None),
        **perf_data,
    }

    if server_times:
        record["server_times_ms"] = server_times

        # Log performance summary
        exec_time = server_times.get("execution_ms")
        if exec_time:
            logger.info("Performance test completed in %s", format_ms(exec_time, context.args.ms_format))

    # Write to JSONL log
    await write_jsonl_log(context, record)
