"""
通用 CUTE kernel 测试脚本
参考 checker.py 的设计，提供通用的测试框架
"""
import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path

# Resolve project root so we can import shared tooling
PROJECT_ROOT = None
_current_path = Path(__file__).resolve()
for candidate in _current_path.parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add testcase root to path
TESTCASE_ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTCASE_ROOT_DIR))

from cu2til.llm_trans.utils.gpu_targets import get_gpu_target, available_gpu_targets
from triton_.kernel import triton_kernel

TARGET_GPU_CHOICES = ["auto"] + sorted(available_gpu_targets())

# Import test case specific modules
try:
    from get_data import (
        get_triton_torch_inputs,
        Params,
        triton_output_tensor_transform,
    )
    try:
        # optional multi-shape provider
        from get_data import get_all_triton_torch_inputs  # type: ignore
    except Exception:
        get_all_triton_torch_inputs = None
except ImportError as e:
    print(f"ERROR: Failed to import test case modules: {e}")
    sys.exit(1)

# Import common comparison function
try:
    from cu2til.tools.checker import compare_results, run_performance_test
except ImportError:
    print("Warning: Could not import checker utilities, using basic comparison")
    compare_results = None
    run_performance_test = None

# GPU Architecture detection / selection
def _autodetect_gpu_arch():
    """Detect GPU compute capability."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            major, minor = props.major, props.minor
            compute_cap = major * 10 + minor

            arch_map = {
                90: ("compute_90", "sm_90", "Hopper SM90 (auto)"),
                89: ("compute_89", "sm_89", "Ada SM89 (auto)"),
                86: ("compute_86", "sm_86", "Ampere SM86 (auto)"),
                80: ("compute_80", "sm_80", "Ampere SM80 (auto)"),
                75: ("compute_75", "sm_75", "Turing SM75 (auto)"),
            }

            if compute_cap in arch_map:
                return arch_map[compute_cap]
            return (f"compute_{major}{minor}", f"sm_{major}{minor}", f"SM{major}{minor} (auto)")
        print("WARNING: No CUDA device available, using default SM80")
        return ("compute_80", "sm_80", "Default (no CUDA)")
    except Exception as exc:
        print(f"WARNING: Failed to detect GPU arch: {exc}, using SM80")
        return ("compute_80", "sm_80", "Default (fallback)")


def resolve_arch(args):
    """Resolve (compute, sm, label) tuple from CLI arguments or auto-detect."""
    if args.arch:
        parts = args.arch.split(',')
        compute_arch = parts[0]
        sm_arch = parts[1] if len(parts) > 1 else parts[0].replace('compute', 'sm')
        return compute_arch, sm_arch, "User-specified"

    if args.target_gpu != "auto":
        spec = get_gpu_target(args.target_gpu)
        return spec.compute, spec.sm, spec.summary_line()

    return _autodetect_gpu_arch()


def _clone_tensor_sequence(sequence):
    """Clone tensors inside a sequence while keeping scalars untouched."""
    cloned = []
    tensor_map = {}
    for item in sequence:
        if isinstance(item, torch.Tensor):
            clone = item.clone()
            tensor_map[id(item)] = clone
            cloned.append(clone)
        else:
            cloned.append(item)
    return cloned, tensor_map


def _clone_bundle(inputs, output_tensors):
    """Clone inputs/output tensors so Triton and CUTE runs don't share buffers."""
    cloned_inputs, tensor_map = _clone_tensor_sequence(inputs)
    cloned_outputs = []
    for tensor in output_tensors:
        if isinstance(tensor, torch.Tensor):
            cloned_outputs.append(tensor_map.get(id(tensor), tensor.clone()))
        else:
            cloned_outputs.append(tensor)
    return cloned_inputs, cloned_outputs


def _run_triton_reference(triton_inputs, output_tensors):
    """Execute Triton kernel and return its output tensor."""
    triton_kernel(*triton_inputs)
    torch.cuda.synchronize()
    return triton_output_tensor_transform(output_tensors[0])


def _compare_outputs(reference, candidate):
    """Compare tensors with fallbacks."""
    if compare_results:
        return compare_results(reference, candidate, atol=1e-2, rtol=1e-2, test_type=["Triton", "CUTE"])
    reference_f32 = reference.to(torch.float32)
    candidate_f32 = candidate.to(torch.float32)
    ok = torch.allclose(reference_f32, candidate_f32, atol=1e-2, rtol=1e-2)
    print(f"✅ Result match: {ok}")
    return ok

# CUTLASS library path
CUTLASS_ROOT = os.environ.get('CUTLASS_ROOT', '/data/apps/project/cu2tri/elib/cutlass_latest')
CUTE_INCLUDE = f"{CUTLASS_ROOT}/include"

def parse_args():
    parser = argparse.ArgumentParser(description='CUTE kernel test')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing')
    parser.add_argument('--compile-only', action='store_true',
                       help='Only compile, do not run tests')
    parser.add_argument('--arch', type=str, default=None,
                       help='Override GPU architecture (e.g., compute_90,sm_90)')
    parser.add_argument('--target-gpu', choices=TARGET_GPU_CHOICES, default="h800_sxm",
                       help='Named GPU profile for nvcc gencode (default: h800_sxm). Use "auto" to detect.')
    parser.add_argument('--dynamic', action='store_true',
                       help='Run all registered shapes when get_all_triton_torch_inputs is available')
    return parser.parse_args()

def compile_cute_kernel(args):
    """Compile CUTE C++ kernel."""
    cute_dir = TESTCASE_ROOT_DIR / "cute_"
    kernel_cu = cute_dir / "kernel.cu"
    kernel_so = cute_dir / "kernel.so"
    
    if not kernel_cu.exists():
        print(f"ERROR: CUTE kernel not found: {kernel_cu}")
        return False
    
    # Detect or use specified architecture
    compute_arch, sm_arch, arch_name = resolve_arch(args)
    
    print(f"Target GPU Architecture: {arch_name} ({compute_arch}/{sm_arch})")
    
    # Compilation command
    compile_cmd = [
        'nvcc',
        '-std=c++17',
        '-O3',
        '--shared',
        '-Xcompiler', '-fPIC',
        f'-I{CUTE_INCLUDE}',
        '-I/usr/local/cuda/include',
        '-gencode', f'arch={compute_arch},code={sm_arch}',
        '-o', str(kernel_so),
        str(kernel_cu),
    ]
    
    print(f"\nCompiling CUTE kernel...")
    print(f"Command: {' '.join(compile_cmd)}\n")
    
    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(cute_dir)
        )
        
        if result.returncode != 0:
            print(f"❌ Compilation FAILED:")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            return False
        
        if result.stdout.strip():
            print(f"Compiler output:\n{result.stdout}")
        
        print(f"✅ Compilation successful: {kernel_so}")
        return True
    
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def load_cute_kernel():
    """Load compiled CUTE kernel as a callable function."""
    cute_dir = TESTCASE_ROOT_DIR / "cute_"
    kernel_so = cute_dir / "kernel.so"
    
    if not kernel_so.exists():
        print(f"ERROR: Compiled kernel not found: {kernel_so}")
        return None
    
    try:
        import ctypes
        lib = ctypes.CDLL(str(kernel_so))
        
        # Check if wrapper function exists
        try:
            _ = lib.cute_kernel_wrapper
        except AttributeError:
            print(f"ERROR: Function 'cute_kernel_wrapper' not found in {kernel_so}")
            print("Available symbols can be checked with: nm -D kernel.so")
            return None
        
        return lib
    except Exception as e:
        print(f"ERROR loading kernel: {e}")
        return None

def cute_kernel_wrapper_caller(lib, triton_all_inputs, params):
    """
    通用的 CUTE kernel 调用包装器
    根据 get_cuda_torch_inputs 返回的格式自动调用 kernel
    类似于 checker.py 中的设计
    """
    import ctypes
    
    # 从 triton_all_inputs 提取输入并调用 kernel
    # 假设最后一个是输出张量（类似 checker.py 的约定）
    
    # 简单策略：将所有张量的 data_ptr 和标量参数传递给 kernel
    args = []
    tensor_count = 0
    
    for inp in triton_all_inputs:
        if isinstance(inp, torch.Tensor):
            args.append(ctypes.c_void_p(inp.data_ptr()))
            tensor_count += 1
        elif isinstance(inp, int):
            args.append(ctypes.c_int(inp))
        elif isinstance(inp, float):
            args.append(ctypes.c_float(inp))
        else:
            # 尝试转换
            args.append(inp)
    
    # 调用 kernel
    try:
        lib.cute_kernel_wrapper(*args)
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"ERROR calling kernel: {e}")
        return False

def _run_case(lib, case_bundle, params, case_label=None):
    """Execute Triton reference + CUTE candidate for a single bundle."""
    triton_inputs_src, _, output_tensors_src = case_bundle
    cute_inputs, cute_outputs = _clone_bundle(triton_inputs_src, output_tensors_src)
    triton_inputs, triton_outputs = _clone_bundle(triton_inputs_src, output_tensors_src)

    if case_label:
        print(f"\n{'-'*80}")
        print(f"🔬 {case_label}")
        print(f"{'-'*80}")

    print("🔥 Running Triton reference kernel...")
    reference_output = _run_triton_reference(triton_inputs, triton_outputs)

    print("⚡ Running CUTE kernel candidate...")
    success = cute_kernel_wrapper_caller(lib, cute_inputs, params)
    if not success:
        print("❌ CUTE kernel execution failed")
        return False

    candidate_output = triton_output_tensor_transform(cute_outputs[0])
    checkok = _compare_outputs(reference_output, candidate_output)

    print("\n🔬 Sample Output (first 4 values):")
    print(f"   Triton : {reference_output.flatten()[:4]}")
    print(f"   CUTE   : {candidate_output.flatten()[:4]}")

    return checkok


def check_cute_vs_triton(args, params=None):
    """
    Check CUTE vs Triton reference implementation.
    """
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")

    if params is None:
        params = Params()

    print(f"\n{'='*80}")
    print(f"🔧 PHASE 1: CUTE Kernel Compilation")
    print(f"{'='*80}")

    if not compile_cute_kernel(args):
        print("\n❌ FAILED: Compilation failed")
        return False

    if args.compile_only:
        print("\n✅ Compile-only mode: PASSED")
        return True

    print(f"\n{'='*80}")
    print(f"⚡ PHASE 2: CUTE vs Triton Testing")
    print(f"{'='*80}")

    lib = load_cute_kernel()
    if lib is None:
        print("\n❌ FAILED: Could not load kernel")
        return False

    dynamic_mode = args.dynamic and get_all_triton_torch_inputs is not None
    if args.dynamic and get_all_triton_torch_inputs is None:
        print("⚠️  --dynamic requested but get_all_triton_torch_inputs is unavailable; falling back to single-shape mode.")

    if dynamic_mode:
        all_cases = get_all_triton_torch_inputs(params)
        overall_ok = True
        for idx, entry in enumerate(all_cases, start=1):
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], tuple):
                shape_info, bundle = entry
                label = f"Shape {idx}: {shape_info}"
            else:
                bundle = entry
                label = f"Shape {idx}"
            ok = _run_case(lib, bundle, params, case_label=label)
            overall_ok = overall_ok and ok
        return overall_ok

    case_bundle = get_triton_torch_inputs(params)
    checkok = _run_case(lib, case_bundle, params)

    if args.no_perf:
        print("\nPerformance testing disabled (--no-perf)")
    elif run_performance_test and checkok:
        print(f"\n{'='*80}")
        print(f"🚀 PHASE 3: Performance Benchmark (CUTE vs Triton)")
        print(f"{'='*80}")
        try:
            cute_perf_inputs, _ = _clone_bundle(case_bundle[0], case_bundle[2])
            triton_perf_inputs, _ = _clone_bundle(case_bundle[0], case_bundle[2])

            run_performance_test(
                [cute_perf_inputs],
                [triton_perf_inputs],
                lambda kernel_inputs: cute_kernel_wrapper_caller(lib, kernel_inputs, params),
                lambda kernel_inputs: triton_kernel(*kernel_inputs),
                test_type=["CUTE", "Triton"],
            )
        except Exception as exc:
            print(f"Performance test failed: {exc}")

    return checkok

if __name__ == "__main__":
    args = parse_args()

    success = check_cute_vs_triton(args)

    if not success:
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ All tests completed successfully!")
    print("="*80)
