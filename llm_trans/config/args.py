from __future__ import annotations

import argparse
from typing import Sequence

from ..utils.gpu_targets import available_gpu_targets


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run code translation with iterative fixing (supports CUDA->Triton, Triton->CUTE, etc.)"
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
        "--skip-case-types",
        nargs="*",
        help="Case types to skip (exclude from run)",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        default=False,
        help="Test only the first case from each case type",
    )
    parser.add_argument(
        "--enable-perf",
        action="store_true",
        default=False,
        help="Enable performance testing (default: False)",
    )
    parser.add_argument(
        "--perf-warmup",
        type=int,
        default=None,
        help="Override warmup iterations for perf benchmark",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=None,
        help="Override timing iterations for perf benchmark",
    )
    parser.add_argument(
        "--direction",
        choices=["cu2tri", "tri2cute"],
        default="cu2tri",
        help="Translation direction: cu2tri (CUDA→Triton) or tri2cute (Triton→CUTE)",
    )
    parser.add_argument(
        "--testset",
        choices=[
            "xpiler",
            "xpiler_extended",
            "leetcuda_dynamic",
            "leetcuda_dynamic_2",
            "hard",
            "triton_tutorial",
            "flaggems_ops",
            "unsloth_kernels",
            "ligerkernel_ops",
        ],
        default="xpiler",
        help="Test set to use",
    )
    parser.add_argument(
        "--target-gpu",
        choices=available_gpu_targets(),
        default="h800_sxm",
        help="Target GPU profile used for prompts, nvcc flags, and validation",
    )
    parser.add_argument(
        "--test-gpu",
        type=int,
        default=None,
        help="Local CUDA device index for check scripts (default derived from --target-gpu)",
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
        "--rotate-endpoints",
        action="store_true",
        default=False,
        help="Enable rotating to next endpoint on retry when a provider pool is configured (default: False)",
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
        "--nvgpu-perf-gpu",
        type=int,
        default=None,
        help="Preferred GPU ID for performance tasks (fallback to --nvgpu-gpu)",
    )
    parser.add_argument(
        "--use-npu",
        action="store_true",
        default=True,
        help="Use NPU server for task execution",
    )
    parser.add_argument(
        "--no-npu",
        action="store_true",
        default=False,
        help="Disable NPU server and use local execution",
    )
    parser.add_argument(
        "--npu-server",
        default="http://localhost:8080",
        help="NPU server URL",
    )
    parser.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help="Specific NPU ID to use",
    )
    parser.add_argument(
        "--npu-perf-id",
        type=int,
        default=None,
        help="Preferred NPU ID for performance tasks (fallback to --npu-id)",
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
        "--perf-concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent NVGPU perf tasks (default: 4)",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for model generation (default: 0.3)",
    )
    return parser.parse_args(argv)


__all__ = ["parse_cli_args"]
