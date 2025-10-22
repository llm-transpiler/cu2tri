from __future__ import annotations

import argparse
from typing import Sequence


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUDA to Triton translation with iterative fixing"
    )
    parser.add_argument(
        "--model",
        default="gpt_oss_20b",
        help="Model key or fully qualified model name defined in model_clients.yaml",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        default=True,
        help="Output logs to console (default: True)",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        default=False,
        help="Disable console output",
    )
    parser.add_argument(
        "--case-types",
        nargs="*",
        help="Specific case types to test (default: all)",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        default=False,
        help="Test only the first case from each case type",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        default=True,
        help="Skip performance testing (default: True)",
    )
    parser.add_argument(
        "--testset",
        choices=[
            "xpiler",
            "xpiler_extended",
            "leetcuda_dynamic",
            "leetcuda_dynamic_2",
            "hard",
        ],
        default="xpiler",
        help="Test set to use",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=60,
        help="Wait time in seconds when encountering API overload errors",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum number of retries for API overload errors",
    )
    parser.add_argument(
        "--use-nvgpu",
        action="store_true",
        default=True,
        help="Use NVGPU server for task execution",
    )
    parser.add_argument(
        "--no-nvgpu",
        action="store_true",
        default=False,
        help="Disable NVGPU server and use local execution",
    )
    parser.add_argument(
        "--nvgpu-server",
        default="http://localhost:8080",
        help="NVGPU server URL",
    )
    parser.add_argument(
        "--nvgpu-gpu",
        type=int,
        default=None,
        help="Specific GPU ID to use",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of fix-and-test rounds per attempt",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent tasks",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum number of independent attempts to run per case",
    )
    parser.add_argument(
        "--attempt-policy",
        choices=["first_success", "exhaustive"],
        default="exhaustive",
        help="Attempt execution policy",
    )
    parser.add_argument(
        "--ms-format",
        choices=["comma", "plain"],
        default="comma",
        help="Milliseconds formatting style for logs",
    )
    parser.add_argument(
        "--resume-conversation",
        action="store_true",
        default=False,
        help="Resume conversation from existing history if available",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        help=(
            "Override provider key for the selected model (must exist in model_clients.yaml)."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help=(
            "Root directory for outputs; overrides env LLM_TRANS_OUTPUTS_ROOT. "
            "Default: <project>/cu2til/llm_trans/runs/cu2tri"
        ),
    )
    return parser.parse_args(argv)


__all__ = ["parse_cli_args"]
