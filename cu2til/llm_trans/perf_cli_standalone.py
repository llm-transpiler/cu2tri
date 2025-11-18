#!/usr/bin/env python3
"""
Standalone Performance Testing CLI for cu2tri

This module provides command-line interface for running performance tests
on successful Triton kernels from previous runs, with minimal dependencies.
"""

from __future__ import annotations

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

        console.print(f"[dim]Searching in: {self.base_dir}[/dim]")
        if model:
            console.print(f"[dim]Filtering by model: {model}[/dim]")

        # Determine the directory structure we're dealing with
        if self.base_dir.name == "xpiler":
            # Base directory: xpiler/ -> model/ -> timestamp/ -> case/ -> attempt/
            search_dirs = self._find_from_base_dir(model)
        elif self.base_dir.name.startswith("gpt_"):
            # Model directory: gpt_X/ -> timestamp/ -> case/ -> attempt/
            search_dirs = self._find_from_model_dir(model)
        else:
            # Timestamp directory: timestamp/ -> case/ -> attempt/ (no model subdirectory)
            search_dirs = self._find_from_timestamp_dir(model)

        for case_info in search_dirs:
            case_name = case_info["case_name"]
            case_dir = case_info["case_dir"]
            model_name = case_info["model"]
            timestamp = case_info["timestamp"]

            # Filter by case types
            if case_types:
                case_type = case_name.split('_')[0] if '_' in case_name else case_name
                if case_type not in case_types:
                    continue

            # Find successful attempts
            successful_attempts = self._find_successful_attempts(case_dir, attempt_numbers)

            for attempt_info in successful_attempts:
                kernel_info = {
                    "model": model_name,
                    "timestamp": timestamp,
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

    def _find_from_base_dir(self, model_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Search from base xpiler directory."""
        search_dirs = []

        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith("gpt_"):
                continue

            if model_filter and model_dir.name != model_filter:
                continue

            console.print(f"[dim]Processing model: {model_dir.name}[/dim]")
            for timestamp_dir in model_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                console.print(f"[dim]  Processing timestamp: {timestamp_dir.name}[/dim]")
                for case_dir in timestamp_dir.iterdir():
                    if not case_dir.is_dir():
                        continue

                    search_dirs.append({
                        "case_name": case_dir.name,
                        "case_dir": case_dir,
                        "model": model_dir.name,
                        "timestamp": timestamp_dir.name,
                    })

        return search_dirs

    def _find_from_model_dir(self, model_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Search from model directory."""
        search_dirs = []

        model_name = self.base_dir.name
        if model_filter and model_name != model_filter:
            console.print(f"[red]Model mismatch: expected {model_filter}, found {model_name}[/red]")
            return []

        console.print(f"[dim]Processing model: {model_name}[/dim]")
        for timestamp_dir in self.base_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            console.print(f"[dim]  Processing timestamp: {timestamp_dir.name}[/dim]")
            for case_dir in timestamp_dir.iterdir():
                if not case_dir.is_dir():
                    continue

                search_dirs.append({
                    "case_name": case_dir.name,
                    "case_dir": case_dir,
                    "model": model_name,
                    "timestamp": timestamp_dir.name,
                })

        return search_dirs

    def _find_from_timestamp_dir(self, model_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Search from timestamp directory (no model subdirectory)."""
        search_dirs = []

        # We need to infer the model name from the parent directory
        parent = self.base_dir.parent
        if parent.name.startswith("gpt_"):
            model_name = parent.name
        else:
            # Use a default or look for clues in the directory contents
            model_name = "unknown"

        if model_filter and model_name != model_filter:
            console.print(f"[red]Model mismatch: expected {model_filter}, found {model_name}[/red]")
            return []

        console.print(f"[dim]Processing timestamp: {self.base_dir.name} (model: {model_name})[/dim]")
        for case_dir in self.base_dir.iterdir():
            if not case_dir.is_dir():
                continue

            search_dirs.append({
                "case_name": case_dir.name,
                "case_dir": case_dir,
                "model": model_name,
                "timestamp": self.base_dir.name,
            })

        return search_dirs

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


class SimpleNVGPUClient:
    """Simplified NVGPU client for performance testing."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None

    def __enter__(self):
        import requests
        self.session = requests.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    def health_check(self) -> bool:
        """Check if the NVGPU server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def submit_task(self, script_path: str, task_type: str = "performance",
                   task_label: str = None, gpu_id: int = None) -> str:
        """Submit a task to the NVGPU server."""
        data = {
            "script_path": str(script_path),
            "task_type": task_type,
            "task_label": task_label,
            "gpu_id": gpu_id
        }

        response = self.session.post(f"{self.server_url}/tasks", json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["task_id"]

    def get_task(self, task_id: str):
        """Get task status."""
        response = self.session.get(f"{self.server_url}/tasks/{task_id}", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_full_task_log(self, task_id: str, log_type: str = "stdout") -> str:
        """Get full task log."""
        response = self.session.get(f"{self.server_url}/tasks/{task_id}/log",
                                  params={"log_type": log_type}, timeout=30)
        response.raise_for_status()
        return response.get("content", "")


async def run_performance_test(
    kernel_info: Dict[str, Any],
    nvgpu_server: str = "http://localhost:8080",
    nvgpu_gpu: Optional[int] = None,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
) -> Dict[str, Any]:
    """Run performance test on a single kernel."""

    work_dir = kernel_info["work_dir"]
    kernel_path = kernel_info["kernel_path"]

    # Create performance test script
    perf_script_path = work_dir / "perf_test.py"
    create_performance_script(
        perf_script_path,
        kernel_path,
        warmup_iterations,
        benchmark_iterations
    )

    # Use NVGPU client
    with SimpleNVGPUClient(nvgpu_server) as client:
        if not client.health_check():
            raise RuntimeError(f"NVGPU server at {nvgpu_server} is not responding")

        task_label = (
            f"perf_test/{kernel_info['model']}/"
            f"{kernel_info['case_name']}/"
            f"attempt_{kernel_info['attempt']}/"
            f"round_{kernel_info['round']}"
        )

        task_id = client.submit_task(
            script_path=str(perf_script_path.absolute()),
            task_type="performance",
            task_label=task_label,
            gpu_id=nvgpu_gpu,
        )

        console.print(f"[green]Submitted task: {task_id[:8]}[/green]")

        # Wait for completion
        result = await wait_for_task(client, task_id)

        # Collect performance data
        perf_data = extract_performance_data(client, task_id)

        # Save performance log
        save_performance_log(kernel_info, task_id, result, perf_data, client)

        return {
            "kernel_info": kernel_info,
            "task_id": task_id,
            "result": result,
            "performance_data": perf_data,
        }


def create_performance_script(
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
import json
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
            parts = case_name.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                size = int(parts[-1])
            else:
                size = 1024
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
    run_performance_test()
'''

    script_path.write_text(script_content)
    script_path.chmod(0o755)


async def wait_for_task(client, task_id: str, timeout: int = 600) -> Dict[str, Any]:
    """Wait for task completion with timeout."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = client.get_task(task_id)

        if result.get("status") in ["completed", "failed", "cancelled"]:
            return result

        await asyncio.sleep(2)

    raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")


def extract_performance_data(client, task_id: str) -> Optional[Dict[str, Any]]:
    """Extract performance data from task output."""
    try:
        stdout = client.get_full_task_log(task_id, "stdout")

        # Look for JSON results in stdout
        for line in stdout.split('\n'):
            if line.startswith('{') and '"success"' in line:
                return json.loads(line)

        return None
    except Exception as exc:
        console.print(f"[red]Failed to extract performance data: {exc}[/red]")
        return None


def save_performance_log(
    kernel_info: Dict[str, Any],
    task_id: str,
    result: Dict[str, Any],
    perf_data: Optional[Dict[str, Any]],
    client,
):
    """Save performance test log to the original logs folder."""
    logs_dir = kernel_info["work_dir"] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    perf_log_path = logs_dir / f"performance_test_attempt_{kernel_info['attempt']}.log"

    try:
        stdout = client.get_full_task_log(task_id, "stdout")
        stderr = client.get_full_task_log(task_id, "stderr")

        with open(perf_log_path, "w") as f:
            f.write(f"=== Performance Test Results ===\n")
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Kernel: {kernel_info['case_name']}\n")
            f.write(f"Model: {kernel_info['model']}\n")
            f.write(f"Attempt: {kernel_info['attempt']}\n")
            f.write(f"Round: {kernel_info['round']}\n")
            f.write(f"Status: {result.get('status', 'unknown')}\n")
            f.write(f"Exit Code: {result.get('exit_code', 'unknown')}\n")
            f.write(f"GPU: {result.get('assigned_gpu', 'auto')}\n\n")

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


async def run_performance_tests_async(
    successful_kernels: List[Dict[str, Any]],
    nvgpu_server: str,
    nvgpu_gpu: Optional[int],
    warmup: int,
    iters: int,
    concurrency: int,
) -> List[Dict[str, Any]]:
    """Run performance tests asynchronously."""

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(kernel_info):
        async with semaphore:
            console.print(f"🧪 Testing {kernel_info['case_name']} (attempt {kernel_info['attempt']})")
            try:
                result = await run_performance_test(
                    kernel_info, nvgpu_server, nvgpu_gpu, warmup, iters
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

    return results


# CLI Interface
@click.group()
def cli():
    """Performance testing CLI for cu2tri."""
    pass


@cli.command()
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
def test(
    base_dir: Path,
    model: str | None,
    case_types: tuple[str, ...],
    attempts: tuple[int, ...],
    nvgpu_server: str,
    nvgpu_gpu: int | None,
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

    # Run async tests
    results = asyncio.run(run_performance_tests_async(
        successful_kernels, nvgpu_server, nvgpu_gpu, warmup, iters, concurrency
    ))

    # Summary
    successful_tests = [r for r in results if r is not None and r["result"].get("status") == "completed"]
    failed_tests = [r for r in results if r is None or r["result"].get("status") != "completed"]

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


@cli.command()
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
    help="Specific attempt numbers to list (e.g., 1, 2). Can be used multiple times.",
)
def list_kernels(base_dir: Path, model: str | None, case_types: tuple[str, ...], attempts: tuple[int, ...]):
    """List all successful kernels available for performance testing."""

    finder = SuccessfulKernelFinder(base_dir)
    kernels = finder.find_successful_kernels(
        model=model,
        case_types=list(case_types) if case_types else None,
        attempt_numbers=list(attempts) if attempts else None,
    )

    if not kernels:
        console.print("No successful kernels found.")
        return

    console.print(f"Found {len(kernels)} successful kernels:\n")

    # Group by model
    models = {}
    for kernel in kernels:
        model_name = kernel["model"]
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(kernel)

    for model_name, model_kernels in models.items():
        console.print(f"📁 Model: {model_name}")

        # Group by case type
        case_types_found = {}
        for kernel in model_kernels:
            case_type = kernel["case_type"]
            if case_type not in case_types_found:
                case_types_found[case_type] = []
            case_types_found[case_type].append(kernel)

        for case_type, case_kernels in case_types_found.items():
            console.print(f"  📂 Case Type: {case_type}")
            for kernel in case_kernels:
                console.print(
                    f"    • {kernel['case_name']} (attempt {kernel['attempt']}, "
                    f"round {kernel['round']})"
                )
        console.print()


@cli.command()
@click.option(
    "--base-dir",
    default=Path("/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler"),
    type=click.Path(exists=True, path_type=Path),
    help="Base directory containing run results",
)
@click.option(
    "--model",
    help="Filter by model name",
)
@click.option(
    "--case-types",
    multiple=True,
    help="Filter by case types",
)
def stats(base_dir: Path, model: str | None, case_types: tuple[str, ...]):
    """Show statistics about successful kernels."""

    finder = SuccessfulKernelFinder(base_dir)
    kernels = finder.find_successful_kernels(
        model=model,
        case_types=list(case_types) if case_types else None,
    )

    if not kernels:
        console.print("No successful kernels found matching the criteria.")
        return

    # Statistics
    total_kernels = len(kernels)
    models = set(k["model"] for k in kernels)
    case_types_found = set(k["case_type"] for k in kernels)
    attempts_found = set(k["attempt"] for k in kernels)

    console.print(f"📊 Statistics for successful kernels:")
    console.print(f"  Total kernels: {total_kernels}")
    console.print(f"  Models: {', '.join(sorted(models))}")
    console.print(f"  Case types: {', '.join(sorted(case_types_found))}")
    console.print(f"  Attempt numbers: {', '.join(map(str, sorted(attempts_found)))}")

    # Success rate by model
    console.print(f"\n📈 Success breakdown by model:")
    for model_name in sorted(models):
        model_kernels = [k for k in kernels if k["model"] == model_name]
        console.print(f"  {model_name}: {len(model_kernels)} kernels")


if __name__ == "__main__":
    cli()