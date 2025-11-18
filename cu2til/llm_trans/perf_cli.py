#!/usr/bin/env python3
"""
Performance Testing CLI for cu2tri

This module provides command-line interface for running performance tests
on successful Triton kernels from previous runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Sequence

import click

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from cu2til.llm_trans.services.performance_testing import main as perf_main


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

    # Convert to the format expected by the main function
    asyncio.run(perf_main(
        base_dir=base_dir,
        model=model,
        case_types=case_types,
        attempts=attempts,
        nvgpu_server=nvgpu_server,
        nvgpu_gpu=nvgpu_gpu,
        warmup=warmup,
        iters=iters,
        concurrency=concurrency,
        dry_run=dry_run,
    ))


@cli.command()
@click.option(
    "--base-dir",
    default=Path("/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler"),
    type=click.Path(exists=True, path_type=Path),
    help="Base directory containing run results",
)
def list_kernels(base_dir: Path):
    """List all successful kernels available for performance testing."""

    from cu2til.llm_trans.services.performance_testing import SuccessfulKernelFinder

    finder = SuccessfulKernelFinder(base_dir)
    kernels = finder.find_successful_kernels()

    if not kernels:
        click.echo("No successful kernels found.")
        return

    click.echo(f"Found {len(kernels)} successful kernels:\n")

    # Group by model
    models = {}
    for kernel in kernels:
        model = kernel["model"]
        if model not in models:
            models[model] = []
        models[model].append(kernel)

    for model, model_kernels in models.items():
        click.echo(f"📁 Model: {model}")

        # Group by case type
        case_types = {}
        for kernel in model_kernels:
            case_type = kernel["case_type"]
            if case_type not in case_types:
                case_types[case_type] = []
            case_types[case_type].append(kernel)

        for case_type, case_kernels in case_types.items():
            click.echo(f"  📂 Case Type: {case_type}")
            for kernel in case_kernels:
                click.echo(
                    f"    • {kernel['case_name']} (attempt {kernel['attempt']}, "
                    f"round {kernel['round']})"
                )
        click.echo()


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

    from cu2til.llm_trans.services.performance_testing import SuccessfulKernelFinder

    finder = SuccessfulKernelFinder(base_dir)
    kernels = finder.find_successful_kernels(
        model=model,
        case_types=list(case_types) if case_types else None,
    )

    if not kernels:
        click.echo("No successful kernels found matching the criteria.")
        return

    # Statistics
    total_kernels = len(kernels)
    models = set(k["model"] for k in kernels)
    case_types_found = set(k["case_type"] for k in kernels)
    attempts = set(k["attempt"] for k in kernels)

    click.echo(f"📊 Statistics for successful kernels:")
    click.echo(f"  Total kernels: {total_kernels}")
    click.echo(f"  Models: {', '.join(sorted(models))}")
    click.echo(f"  Case types: {', '.join(sorted(case_types_found))}")
    click.echo(f"  Attempt numbers: {', '.join(map(str, sorted(attempts)))}")

    # Success rate by model
    click.echo(f"\n📈 Success breakdown by model:")
    for model_name in sorted(models):
        model_kernels = [k for k in kernels if k["model"] == model_name]
        click.echo(f"  {model_name}: {len(model_kernels)} kernels")


if __name__ == "__main__":
    cli()