import os
import sys
import argparse
import json
from datetime import datetime, timezone

import torch

from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes  # type: ignore
try:
    from get_data import cuda_output_tensor_transform  # type: ignore
except ImportError:
    cuda_output_tensor_transform = lambda x: x
try:
    from get_data import get_all_cuda_torch_inputs  # type: ignore
except ImportError:
    get_all_cuda_torch_inputs = None

from torch_.ref import torch_kernel  # type: ignore
from triton_.kernel import triton_kernel  # type: ignore
from llm_trans.tools.checker import (
    check_all_kernels,
    run_performance_all,
    check_triton_vs_torch_dynamic,
    check_cuda_vs_torch_dynamic,
)

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive kernel test: CUDA, Triton vs PyTorch')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the CUDA kernel without running tests (default: False)')
    parser.add_argument('--no-perf', action='store_true',
                        help='Disable performance testing (default: False, performance testing enabled)')
    # perf-only and JSON output options
    parser.add_argument('--perf-only', action='store_true',
                        help='Only run performance benchmark and output JSON (skip correctness prints)')
    parser.add_argument('--perf-json-out', default=None,
                        help='Path to write performance JSON; if not set, prints one-line JSON to stdout')
    parser.add_argument('--perf-warmup', type=int, default=None,
                        help='Warmup iterations override for benchmark (if supported)')
    parser.add_argument('--perf-iters', type=int, default=None,
                        help='Timing iterations override for benchmark (if supported)')
    parser.add_argument('--case-tag', default=None, help='Case label for aggregation')
    parser.add_argument('--shape-tag', default=None, help='Shape label for aggregation')
    parser.add_argument('--gpu-index', type=int, default=None,
                        help='CUDA device index to run on (default: current device)')
    parser.add_argument('--dynamic', action='store_true',
                        help='If set and get_all_cuda_torch_inputs is available, run multi-shape suite')
    parser.add_argument('--max-shapes', type=int, default=None,
                        help='Limit number of shapes in dynamic mode')
    return parser.parse_args()


def _write_perf_json(data: dict, out_path: str | None) -> None:
    text = json.dumps(data, ensure_ascii=False)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text + "\n")
    else:
        # single line to stdout to simplify parsing
        print(text)


if __name__ == "__main__":
    args = parse_args()
    if args.gpu_index is not None:
        torch.cuda.set_device(args.gpu_index)
    # Use explicit flags when set
    compile_only = args.compile_only
    enable_perf = not args.no_perf

    if args.dynamic:
        if get_all_cuda_torch_inputs is None:
            sys.stderr.write("Dynamic mode requested but get_all_cuda_torch_inputs is not defined in get_data.py\n")
            sys.exit(2)
        params = Params()
        if args.max_shapes is not None and hasattr(params, "test_shapes"):
            params.test_shapes = params.test_shapes[: args.max_shapes]  # type: ignore[attr-defined]
        success_cuda = check_cuda_vs_torch_dynamic(
            TESTCASE_ROOT_DIR,
            get_all_cuda_torch_inputs,
            params,
            torch_kernel,
            get_cuda_argtypes,
            output_tensor_transform=cuda_output_tensor_transform,
            enable_perf=enable_perf,
            compile_only=compile_only,
        )
        success_triton = check_triton_vs_torch_dynamic(
            get_all_cuda_torch_inputs,
            params,
            torch_kernel,
            triton_kernel,
            output_tensor_transform=cuda_output_tensor_transform,
            enable_perf=enable_perf,
        )
        sys.exit(0 if (success_cuda and success_triton) else 1)

    # Fast path: perf-only JSON mode
    if args.perf_only:
        # Build inputs once, run perf across CUDA/Triton/Torch
        params = Params()
        triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)

        # Load CUDA kernel by triggering inside check_all_kernels Phase 1 logic minimally
        # We reuse get_cuda_argtypes() + cu2til.tools.builder inside check_all when needed
        # For perf-only, we still need the compiled CUDA kernel callable and PyTorch ref
        # Use check_all_kernels to compile and prepare; then reconstruct inputs for perf
        try:
            # We will call check_all_kernels with enable_perf=False and compile_only=False
            # to ensure CUDA kernel is compiled and Triton path is valid; correctness prints allowed
            _ = check_all_kernels(
                testcase_root_dir=TESTCASE_ROOT_DIR,
                get_cuda_torch_inputs=get_cuda_torch_inputs,
                params=params,
                torch_kernel=torch_kernel,
                triton_kernel=triton_kernel,
                get_cuda_argtypes=get_cuda_argtypes,
                output_tensor_transform=cuda_output_tensor_transform,
                enable_perf=False,
                compile_only=False,
            )
        except Exception as e:
            # Fall back to continuing; run_performance_all below will raise if missing
            sys.stderr.write(f"[perf-only] Warm compile/check failed: {e}\n")

        # Prepare per-kernel inputs again to get the CUDA pointer list
        cuda_all_inputs, torch_all_inputs2, _ = get_cuda_torch_inputs(params)
        # run_performance_all expects: (triton_all_inputs, cuda_all_inputs, torch_all_inputs, ...)
        try:
            triton_ms, cuda_ms, torch_ms = run_performance_all(
                triton_all_inputs,
                [t.data_ptr() if hasattr(getattr(t, 'data_ptr', None), '__call__') else t for t in cuda_all_inputs],
                torch_all_inputs,
                triton_kernel,
                None,  # not used by run_performance_all; we will ignore and recalc via checker if needed
                torch_kernel,
            )
        except Exception:
            # Fallback to checker.run_performance_all signature using loaded kernels from check_all_kernels path
            # Re-import to fetch function then call with kernels directly
            from llm_trans.tools.builder import load_cuda_kernel
            try:
                argtypes = get_cuda_argtypes()
                cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=False)
                # Rebuild inputs
                triton_all_inputs, torch_all_inputs, _ = get_cuda_torch_inputs(params)
                cuda_all_inputs, torch_all_inputs2, _ = get_cuda_torch_inputs(params)
                cuda_ptr_inputs = [t.data_ptr() if hasattr(getattr(t, 'data_ptr', None), '__call__') else t for t in cuda_all_inputs]
                triton_ms, cuda_ms, torch_ms = run_performance_all(
                    triton_all_inputs,
                    cuda_ptr_inputs,
                    torch_all_inputs,
                    triton_kernel,
                    cuda_kernel,
                    torch_kernel,
                )
            except Exception as e:
                sys.stderr.write(f"[perf-only] Performance run failed: {e}\n")
                sys.exit(2)

        # Compose JSON
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        except Exception:
            gpu_name = "unknown"

        out = {
            "type": "perf",
            "ts": _iso_now(),
            "case": args.case_tag,
            "shape": args.shape_tag,
            "gpu": gpu_name,
            "results_ms": {
                "triton": float(triton_ms),
                "cuda": float(cuda_ms),
                "torch": float(torch_ms),
            },
            "speedup": {
                "cuda_vs_torch": (float(torch_ms) / float(cuda_ms)) if cuda_ms else None,
                "triton_vs_cuda": (float(cuda_ms) / float(triton_ms)) if triton_ms else None,
                "triton_vs_torch": (float(torch_ms) / float(triton_ms)) if triton_ms else None,
            },
        }
        _write_perf_json(out, args.perf_json_out)
        sys.exit(0)

    # Default comprehensive flow
    params = Params()

    success = check_all_kernels(
        testcase_root_dir=TESTCASE_ROOT_DIR,
        get_cuda_torch_inputs=get_cuda_torch_inputs,
        params=params,
        torch_kernel=torch_kernel,
        triton_kernel=triton_kernel,
        get_cuda_argtypes=get_cuda_argtypes,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=enable_perf,
        compile_only=compile_only
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)
