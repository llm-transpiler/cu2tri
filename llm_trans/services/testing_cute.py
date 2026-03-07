"""
CUTE (CUTLASS) code compilation and testing support.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from ..utils.trans_timer import TransTimer
from server.common.timezone import ensure_timezone, now_timestamp

from ..core.runtime import RuntimeContext
from ..data.models import TestRoundRecord, AttemptTimingStats
from ..utils.formatting import format_ms


async def run_test_round_cute(
    context: RuntimeContext,
    round_num: int,
    test_work_dir: Path,
    timing_stats: AttemptTimingStats | None = None,
):
    """
    Run a test round for CUTE (C++) code.
    
    Args:
        context: Runtime context
        round_num: Current round number
        test_work_dir: Working directory for this test
        timing_stats: Optional timing statistics to update
    
    Returns:
        Tuple of (success, stdout, stderr, log_file)
    """
    logger = context.logger
    settings = context.settings
    
    logger.debug(f"Running CUTE test round {round_num}...")
    
    # Paths
    kernel_path = test_work_dir / settings.dir_cute / "kernel.cu"
    backup_path = test_work_dir / settings.dir_cute / f"kernel_v{round_num}.cu"
    
    # Backup current kernel
    if kernel_path.exists():
        shutil.copy(kernel_path, backup_path)
        logger.debug(f"Backed up kernel to kernel_v{round_num}.cu")
    
    # Create logs directory
    logs_dir = test_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"cute_test_round_{round_num}.log"
    
    # Run test based on execution mode
    if settings.use_nvgpu and context.nvgpu_available:
        return await run_test_round_cute_nvgpu(
            context, round_num, test_work_dir, log_file, timing_stats
        )
    return await run_test_round_cute_local(
        context, round_num, test_work_dir, log_file, timing_stats
    )


async def run_test_round_cute_local(
    context: RuntimeContext,
    round_num: int,
    test_work_dir: Path,
    log_file: Path,
    timing_stats: AttemptTimingStats | None = None,
):
    """Run CUTE test locally."""
    import sys
    args = context.args
    
    # Command to run the check_cute.py script
    cmd = [sys.executable, "check_cute.py", "--target-gpu", context.settings.target_gpu]
    if args.no_perf:
        cmd.append("--no-perf")
    
    test_start_wall = now_timestamp()
    trans_timer = TransTimer()
    trans_timer.start("test_cute_local_round")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(test_work_dir),
        )
        
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=300  # 5 minutes timeout for compilation
            )
            stdout = stdout_data.decode("utf-8")
            stderr = stderr_data.decode("utf-8")
            returncode = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            error_msg = f"CUTE test round {round_num} timed out"
            with open(log_file, "w") as fp:
                fp.write(f"=== CUTE Test Round {round_num} ===\n")
                fp.write(f"ERROR: {error_msg}\n")
            return False, "", error_msg, log_file
        
        test_end_wall = now_timestamp()
        test_duration_ms = trans_timer.stop("test_cute_local_round") or 0.0
        
        # Update timing stats
        if timing_stats is not None:
            record = TestRoundRecord(
                round=round_num,
                success=returncode == 0 and ("PASSED" in stdout or "successfully" in stdout.lower()),
                execution_mode="local_cute",
                start_time=ensure_timezone(test_start_wall).isoformat(),
                end_time=ensure_timezone(test_end_wall).isoformat(),
                duration_ms=round(test_duration_ms, 3),
            )
            timing_stats.add_test_round(record)
            timing_stats.add_test_time(test_duration_ms)
            
            # Add timer info
            try:
                timers = trans_timer.as_dict()
                if timers:
                    if timing_stats.aggregated_timers is None:
                        timing_stats.aggregated_timers = {}
                    for label, value in timers.items():
                        timing_stats.aggregated_timers[label] = (
                            timing_stats.aggregated_timers.get(label, 0.0) + float(value)
                        )
            except Exception:
                pass
        
        # Write log file
        with open(log_file, "w") as fp:
            fp.write(f"=== CUTE Test Round {round_num} (Local) ===\n")
            fp.write(f"Command: {' '.join(cmd)}\n")
            fp.write(f"Exit code: {returncode}\n")
            fp.write(f"Execution time: {format_ms(test_duration_ms, context.args.ms_format)}\n\n")
            fp.write("=== STDOUT ===\n")
            fp.write(stdout)
            fp.write("\n=== STDERR ===\n")
            fp.write(stderr)
            fp.write("\n=== END OF LOG ===\n")
        
        context.logger.debug(f"CUTE test output saved to {log_file}")
        
        # Determine success
        success = returncode == 0 and ("PASSED" in stdout or "successfully" in stdout.lower())
        return success, stdout, stderr, log_file
    
    except Exception as exc:
        error_msg = f"CUTE test round {round_num} failed: {exc}"
        with open(log_file, "w") as fp:
            fp.write(f"=== CUTE Test Round {round_num} ===\n")
            fp.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file


async def run_test_round_cute_nvgpu(
    context: RuntimeContext,
    round_num: int,
    test_work_dir: Path,
    log_file: Path,
    timing_stats: AttemptTimingStats | None = None,
):
    """Run CUTE test on NVGPU server."""
    # TODO: Implement NVGPU support for CUTE testing
    # For now, fall back to local execution
    context.logger.warning("NVGPU support for CUTE not yet implemented, using local execution")
    return await run_test_round_cute_local(
        context, round_num, test_work_dir, log_file, timing_stats
    )
