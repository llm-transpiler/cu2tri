"""Kernel building utilities for LLM transpiler."""
import os
import subprocess
import ctypes

SEED = 46
CUDA_FOLDER_NAME = "cuda_"
TRITON_FOLDER_NAME = "triton_"
TORCH_FOLDER_NAME = "torch_"
C_FOLDER_NAME = "c_"


def compile_cuda_kernel(testcase_root_dir, cufile_basename="kernel", force_compile=False):
    """Compile CUDA kernel from .cu file to .so library."""
    lib_path = os.path.join(testcase_root_dir, CUDA_FOLDER_NAME, 'lib_cuda_kernel.so')
    cuda_source = os.path.join(testcase_root_dir, CUDA_FOLDER_NAME, f"{cufile_basename}.cu")

    if not os.path.exists(cuda_source):
        raise FileNotFoundError(f"CUDA source not found: {cuda_source}")

    # Check if recompilation needed
    if os.path.exists(lib_path):
        lib_mtime = os.path.getmtime(lib_path)
        src_mtime = os.path.getmtime(cuda_source)
        need_compile = src_mtime > lib_mtime
    else:
        need_compile = True

    if force_compile:
        print("🔧 Force compiling CUDA kernel...")
        need_compile = True

    if need_compile:
        print("🔧 Compiling CUDA kernel...")

        cmd = [
            'nvcc', '-O3', '--shared', '-Xcompiler', '-fPIC',
            '-arch=sm_89', '--use_fast_math',
            '-Wno-deprecated-gpu-targets',
            '-o', lib_path, cuda_source
        ]

        try:
            result = subprocess.run(cmd, cwd=testcase_root_dir, capture_output=True, text=True, check=True)
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines if 'warning' not in line.lower()]
                if error_lines:
                    print(f"Compilation warning: {result.stderr}")

            print("✅ CUDA kernel compiled successfully")
            return lib_path

        except subprocess.CalledProcessError as e:
            print(f"Command: {' '.join(cmd)}")
            print(f"Error output: {e.stderr}")
            raise RuntimeError("❌ CUDA compilation failed")
        except FileNotFoundError:
            raise RuntimeError("❌ nvcc not found")
    else:
        print("✅ CUDA kernel is up to date")
        return lib_path


def load_cuda_kernel(testcase_root_dir, argtypes, restype=None, force_compile=False):
    """Compile and load CUDA kernel as ctypes library."""
    lib_path = compile_cuda_kernel(testcase_root_dir, "kernel", force_compile=force_compile)
    if not lib_path:
        raise RuntimeError("CUDA kernel compilation failed")

    lib = ctypes.CDLL(lib_path)
    lib.cuda_kernel.argtypes = argtypes
    lib.cuda_kernel.restype = restype

    return lib.cuda_kernel


def compile_c_kernel(testcase_root_dir, cfile_basename="kernel", force_compile=False):
    """Compile C kernel from .c file to .so library."""
    lib_path = os.path.join(testcase_root_dir, C_FOLDER_NAME, 'lib_c_kernel.so')
    c_source = os.path.join(testcase_root_dir, C_FOLDER_NAME, f"{cfile_basename}.c")

    if not os.path.exists(c_source):
        raise FileNotFoundError(f"C source not found: {c_source}")

    # Check if recompilation needed
    if os.path.exists(lib_path):
        lib_mtime = os.path.getmtime(lib_path)
        src_mtime = os.path.getmtime(c_source)
        need_compile = src_mtime > lib_mtime
    else:
        need_compile = True

    if force_compile:
        print("🔧 Force compiling C kernel...")
        need_compile = True

    if need_compile:
        print("🔧 Compiling C kernel...")

        cmd = ['gcc', '-O3', '-shared', '-fPIC', '-o', lib_path, c_source]

        try:
            result = subprocess.run(cmd, cwd=testcase_root_dir, capture_output=True, text=True, check=True)
            if result.stderr:
                print(f"Compilation warning: {result.stderr}")

            print("✅ C kernel compiled successfully")
            return lib_path

        except subprocess.CalledProcessError as e:
            print(f"Command: {' '.join(cmd)}")
            print(f"Error output: {e.stderr}")
            raise RuntimeError("❌ C compilation failed")
        except FileNotFoundError:
            raise RuntimeError("❌ gcc not found")
    else:
        print("✅ C kernel is up to date")
        return lib_path


def load_c_kernel(testcase_root_dir, argtypes, restype=None, force_compile=False):
    """Compile and load C kernel as ctypes library."""
    lib_path = compile_c_kernel(testcase_root_dir, "kernel", force_compile=force_compile)
    if not lib_path:
        raise RuntimeError("C kernel compilation failed")

    lib = ctypes.CDLL(lib_path)
    lib.c_kernel.argtypes = argtypes
    lib.c_kernel.restype = restype

    return lib.c_kernel
