#!/usr/bin/env python3
"""
Performance Testing System for Successful Triton Kernels

This module provides functionality to run performance tests on successful Triton kernels
from specific run directories, using NVGPU server in exclusive mode.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..clients import NVGPUClient
from ..core.runtime import RuntimeContext
from ..data.models import AttemptTimingStats
from ..utils.formatting import format_ms
from ..utils.timezone import now_timestamp

console = Console()


class SuccessfulKernelFinder:
    """Find successful Triton kernels from run directories."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def find_successful_kernels(
        self,
        model: Optional[str] = None,
        case_types: Optional[List[str]] = None,
        attempt_numbers: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find successful Triton kernels from run directories.

        Args:
            model: Model name filter (e.g., "gpt_5_mini", "gpt_oss_120b")
            case_types: List of case types to include (e.g., ["add", "avgpool"])
            attempt_numbers: List of attempt numbers to include (e.g., [1, 2])

        Returns:
            List of dictionaries containing kernel information
        """
        successful_kernels = []

        # Iterate through model directories
        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir():
                continue

            if model and model_dir.name != model:
                continue

            # Iterate through timestamp directories
            for timestamp_dir in model_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                # Iterate through case directories
                for case_dir in timestamp_dir.iterdir():
                    if not case_dir.is_dir():
                        continue

                    case_name = case_dir.name

                    # Filter by case types
                    if case_types:
                        case_type = case_name.split('_')[0] if '_' in case_name else case_name
                        if case_type not in case_types:
                            continue

                    # Find successful attempts
                    successful_attempts = self._find_successful_attempts(
                        case_dir, attempt_numbers
                    )

                    for attempt_info in successful_attempts:
                        kernel_info = {
                            "model": model_dir.name,
                            "timestamp": timestamp_dir.name,
                            "case_name": case_name,
                            "case_type": case_name.split('_')[0] if '_' in case_name else case_name,
                            "attempt": attempt_info["attempt"],
                            "round": attempt_info["round"],
                            "kernel_path": attempt_info["kernel_path"],
                            "log_path": attempt_info["log_path"],
                            "work_dir": attempt_info["work_dir"],
                        }
                        successful_kernels.append(kernel_info)

        return successful_kernels

    def _find_successful_attempts(
        self,
        case_dir: Path,
        attempt_numbers: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Find successful attempts in a case directory."""
        successful_attempts = []

        for attempt_dir in case_dir.iterdir():
            if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                continue

            try:
                attempt_num = int(attempt_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            if attempt_numbers and attempt_num not in attempt_numbers:
                continue

            # Find the last successful test round
            success_info = self._find_last_successful_round(attempt_dir)
            if success_info:
                success_info["attempt"] = attempt_num
                success_info["work_dir"] = attempt_dir
                successful_attempts.append(success_info)

        return successful_attempts

    def _find_last_successful_round(self, attempt_dir: Path) -> Optional[Dict[str, Any]]:
        """Find the last successful test round in an attempt directory."""
        logs_dir = attempt_dir / "logs"
        if not logs_dir.exists():
            return None

        # Find all test round logs
        round_logs = list(logs_dir.glob("triton_test_round_*.log"))
        if not round_logs:
            return None

        # Sort by round number and find the last successful one
        round_logs.sort(key=lambda x: int(x.stem.split("_")[-1]))

        successful_round = None
        for log_file in reversed(round_logs):
            if self._is_log_successful(log_file):
                round_num = int(log_file.stem.split("_")[-1])
                kernel_path = attempt_dir / "triton_/kernel.py"

                if kernel_path.exists():
                    return {
                        "round": round_num,
                        "kernel_path": kernel_path,
                        "log_path": log_file,
                    }

        return None

    def _is_log_successful(self, log_file: Path) -> bool:
        """Check if a log file indicates a successful test."""
        try:
            content = log_file.read_text()

            # Check for exit code 0 (success)
            if "Exit code: 0" in content:
                return True

            # Check for PASSED status
            if "STATUS: PASSED" in content or "PASSED" in content:
                return True

            return False
        except Exception:
            return False


class PerformanceTestRunner:
    """Run performance tests on successful kernels."""

    def __init__(self, nvgpu_server: str, nvgpu_gpu: Optional[int] = None):
        self.nvgpu_server = nvgpu_server
        self.nvgpu_gpu = nvgpu_gpu
        self.client = None

    async def _ensure_client(self):
        """Ensure NVGPU client is initialized and healthy."""
        if self.client is None:
            self.client = NVGPUClient(self.nvgpu_server)

        if not self.client.health_check():
            raise RuntimeError(f"NVGPU server at {self.nvgpu_server} is not responding")

    async def run_performance_test(
        self,
        kernel_info: Dict[str, Any],
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Run performance test on a single kernel.

        Args:
            kernel_info: Information about the kernel to test
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations

        Returns:
            Performance test results
        """
        await self._ensure_client()

        work_dir = kernel_info["work_dir"]
        kernel_path = kernel_info["kernel_path"]

        # Create performance test script
        perf_script_path = work_dir / "perf_test.py"
        self._create_performance_script(
            perf_script_path,
            kernel_path,
            warmup_iterations,
            benchmark_iterations
        )

        # Submit performance test task to NVGPU server in exclusive mode
        task_label = (
            f"perf_test/{kernel_info['model']}/"
            f"{kernel_info['case_name']}/"
            f"attempt_{kernel_info['attempt']}/"
            f"round_{kernel_info['round']}"
        )

        task_id = self.client.submit_task_in_script_dir(
            script_path=str(perf_script_path.absolute()),
            task_type="performance",  # This automatically uses exclusive mode
            task_label=task_label,
            gpu_id=self.nvgpu_gpu,
        )

        # Wait for completion
        result = await self._wait_for_task(task_id)

        # Collect performance data
        perf_data = await self._extract_performance_data(task_id)

        # Save performance log to original logs folder
        await self._save_performance_log(
            kernel_info, task_id, result, perf_data
        )

        return {
            "kernel_info": kernel_info,
            "task_id": task_id,
            "result": result,
            "performance_data": perf_data,
        }

    def _create_performance_script(
        self,
        script_path: Path,
        kernel_path: Path,
        warmup_iterations: int,
        benchmark_iterations: int,
    ):
        """Create a performance test script for the kernel."""

        # Read the original kernel to extract necessary information
        kernel_code = kernel_path.read_text()

        script_content = f'''#!/usr/bin/env python3
"""
Performance test script for Triton kernel
Generated automatically by performance testing system
"""

import sys
import time
import torch
import triton
from pathlib import Path

# Add the path to access checker utilities
sys.path.insert(0, "{kernel_path.parent.parent}")

# Import the kernel function
exec(open("{kernel_path}").read())

def get_performance_inputs():
    """Get test inputs for performance testing."""
    # Import the original get_data function if available
    try:
        from check_triton import get_cuda_torch_inputs
        return get_cuda_torch_inputs()
    except ImportError:
        # Fallback: create dummy inputs based on kernel name
        case_name = "{kernel_path.parent.parent.name}"
        if "add" in case_name.lower():
            size = int(case_name.split("_")[-1]) if case_name.split("_")[-1].isdigit() else 1024
            a = torch.randn(size, device='cuda', dtype=torch.float32)
            b = torch.randn(size, device='cuda', dtype=torch.float32)
            c = torch.empty_like(a)
            return [a, b, c], {{}}
        else:
            # Generic fallback
            a = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
            b = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
            c = torch.empty_like(a)
            return [a, b, c], {{}}

def run_performance_test():
    """Run the actual performance test."""
    print("🚀 Starting performance test...")
    print(f"📊 Warmup iterations: {warmup_iterations}")
    print(f"📊 Benchmark iterations: {benchmark_iterations}")

    # Get inputs
    inputs, params = get_performance_inputs()
    print(f"📋 Created test inputs")

    # Move to GPU and sync
    torch.cuda.synchronize()
    print("⚡ Running warmup...")

    # Warmup
    for i in range({warmup_iterations}):
        triton_kernel(*inputs)
        torch.cuda.synchronize()

    print("🏃 Running benchmark...")

    # Benchmark
    start_time = time.time()
    for i in range({benchmark_iterations}):
        triton_kernel(*inputs)
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_ms = (total_time / {benchmark_iterations}) * 1000
    throughput_ops_per_sec = {benchmark_iterations} / total_time

    print(f"✅ Performance test completed!")
    print(f"📊 Total time: {{total_time:.4f}}s")
    print(f"📊 Average time per iteration: {{avg_time_ms:.4f}}ms")
    print(f"📊 Throughput: {{throughput_ops_per_sec:.2f}} ops/sec")

    # Output results in JSON format for easy parsing
    results = {{
        "total_time_s": total_time,
        "avg_time_ms": avg_time_ms,
        "throughput_ops_per_sec": throughput_ops_per_sec,
        "warmup_iterations": {warmup_iterations},
        "benchmark_iterations": {benchmark_iterations},
        "success": True
    }}

    print("=== PERFORMANCE RESULTS ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import json
    run_performance_test()
'''

        script_path.write_text(script_content)
        script_path.chmod(0o755)

    async def _wait_for_task(self, task_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for task completion with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.client.get_task(task_id)

            if result.status in ["completed", "failed", "cancelled"]:
                return result

            await asyncio.sleep(2)

        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")

    async def _extract_performance_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Extract performance data from task output."""
        try:
            stdout = self.client.get_full_task_log(task_id, "stdout")

            # Look for JSON results in stdout
            for line in stdout.split('\n'):
                if line.startswith('{') and '"success"' in line:
                    return json.loads(line)

            return None
        except Exception as exc:
            console.print(f"[red]Failed to extract performance data: {exc}[/red]")
            return None

    async def _save_performance_log(
        self,
        kernel_info: Dict[str, Any],
        task_id: str,
        result: Dict[str, Any],
        perf_data: Optional[Dict[str, Any]],
    ):
        """Save performance test log to the original logs folder."""
        logs_dir = kernel_info["work_dir"] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        perf_log_path = logs_dir / f"performance_test_attempt_{kernel_info['attempt']}.log"

        try:
            stdout = self.client.get_full_task_log(task_id, "stdout")
            stderr = self.client.get_full_task_log(task_id, "stderr")

            with open(perf_log_path, "w") as f:
                f.write(f"=== Performance Test Results ===\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Kernel: {kernel_info['case_name']}\n")
                f.write(f"Model: {kernel_info['model']}\n")
                f.write(f"Attempt: {kernel_info['attempt']}\n")
                f.write(f"Round: {kernel_info['round']}\n")
                f.write(f"Status: {result.status}\n")
                f.write(f"Exit Code: {result.exit_code}\n")
                f.write(f"GPU: {getattr(result, 'gpu_id', 'auto')}\n\n")

                if perf_data:
                    f.write("=== Performance Data ===\n")
                    f.write(json.dumps(perf_data, indent=2))
                    f.write("\n\n")

                f.write("=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
                f.write("\n=== END OF LOG ===\n")

            console.print(f"[green]Performance log saved to: {perf_log_path}[/green]")

        except Exception as exc:
            console.print(f"[red]Failed to save performance log: {exc}[/red]")


# CLI Interface
@click.command()
@click.option(
    "--base-dir",
    default=Path("/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler"),
    type=click.Path(exists=True, path_type=Path),
    help="Base directory containing run results",
)
@click.option(
    "--model",
    help="Filter by model name (e.g., gpt_5_mini, gpt_oss_120b)",
)
@click.option(
    "--case-types",
    multiple=True,
    help="Filter by case types (e.g., add, avgpool). Can be used multiple times.",
)
@click.option(
    "--attempts",
    multiple=True,
    type=int,
    help="Specific attempt numbers to test (e.g., 1, 2). Can be used multiple times.",
)
@click.option(
    "--nvgpu-server",
    default="http://localhost:8080",
    help="NVGPU server URL",
)
@click.option(
    "--nvgpu-gpu",
    type=int,
    help="Specific GPU ID to use (optional)",
)
@click.option(
    "--warmup",
    default=10,
    type=int,
    help="Number of warmup iterations",
)
@click.option(
    "--iters",
    default=100,
    type=int,
    help="Number of benchmark iterations",
)
@click.option(
    "--concurrency",
    default=1,
    type=int,
    help="Number of concurrent performance tests to run",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be tested without running tests",
)
async def main(
    base_dir: Path,
    model: Optional[str],
    case_types: Tuple[str, ...],
    attempts: Tuple[int, ...],
    nvgpu_server: str,
    nvgpu_gpu: Optional[int],
    warmup: int,
    iters: int,
    concurrency: int,
    dry_run: bool,
):
    """Run performance tests on successful Triton kernels."""

    console.print("[bold blue]🚀 Performance Testing System for Successful Triton Kernels[/bold blue]")
    console.print(f"[dim]Base directory: {base_dir}[/dim]")

    # Find successful kernels
    finder = SuccessfulKernelFinder(base_dir)

    case_types_list = list(case_types) if case_types else None
    attempts_list = list(attempts) if attempts else None

    console.print("🔍 Searching for successful kernels...")
    successful_kernels = finder.find_successful_kernels(
        model=model,
        case_types=case_types_list,
        attempt_numbers=attempts_list,
    )

    if not successful_kernels:
        console.print("[yellow]No successful kernels found matching the criteria.[/yellow]")
        return

    console.print(f"[green]Found {len(successful_kernels)} successful kernels[/green]")

    # Display found kernels
    table = Table(title="Successful Kernels Found")
    table.add_column("Model", style="cyan")
    table.add_column("Case", style="magenta")
    table.add_column("Attempt", style="blue")
    table.add_column("Round", style="blue")
    table.add_column("Kernel Path", style="dim")

    for kernel in successful_kernels:
        table.add_row(
            kernel["model"],
            kernel["case_name"],
            str(kernel["attempt"]),
            str(kernel["round"]),
            str(kernel["kernel_path"].relative_to(base_dir)),
        )

    console.print(table)

    if dry_run:
        console.print("\n[yellow]🔍 Dry run mode - not executing performance tests[/yellow]")
        return

    # Run performance tests
    console.print(f"\n🏃 Running performance tests (concurrency: {concurrency})...")

    runner = PerformanceTestRunner(nvgpu_server, nvgpu_gpu)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(kernel_info):
        async with semaphore:
            console.print(f"🧪 Testing {kernel_info['case_name']} (attempt {kernel_info['attempt']})")
            try:
                result = await runner.run_performance_test(
                    kernel_info, warmup, iters
                )
                return result
            except Exception as exc:
                console.print(f"[red]❌ Failed to test {kernel_info['case_name']}: {exc}[/red]")
                return None

    # Run tests with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            f"Running {len(successful_kernels)} performance tests...",
            total=len(successful_kernels)
        )

        tasks = [run_single_test(kernel) for kernel in successful_kernels]
        results = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.advance(task_progress)

    # Summary
    successful_tests = [r for r in results if r is not None and r["result"].status == "completed"]
    failed_tests = [r for r in results if r is None or r["result"].status != "completed"]

    console.print(f"\n[bold green]✅ Performance testing completed![/bold green]")
    console.print(f"[green]Successful tests: {len(successful_tests)}[/green]")
    console.print(f"[red]Failed tests: {len(failed_tests)}[/red]")

    if successful_tests:
        console.print("\n[bold]📊 Performance Summary:[/bold]")
        summary_table = Table()
        summary_table.add_column("Case", style="magenta")
        summary_table.add_column("Avg Time (ms)", style="blue")
        summary_table.add_column("Throughput (ops/s)", style="green")

        for test_result in successful_tests:
            kernel_info = test_result["kernel_info"]
            perf_data = test_result["performance_data"]

            if perf_data:
                summary_table.add_row(
                    kernel_info["case_name"],
                    f"{perf_data.get('avg_time_ms', 0):.2f}",
                    f"{perf_data.get('throughput_ops_per_sec', 0):.2f}",
                )

        console.print(summary_table)


if __name__ == "__main__":
    asyncio.run(main())