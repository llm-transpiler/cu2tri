
import os
import subprocess
import ctypes
SEED = 46
CUDA_FOLDER_NAME = "cuda_"
TRITON_FOLDER_NAME = "triton_"
TORCH_FOLDER_NAME = "torch_"
from eval_.common.loader import load_cuda_extension_from_cufile
from eval_.common.config import EvalConfig
def build_kernel(file_name, header_lib_path="./", build_dir="./build"):
    print("🔧 build_kernel\t:"+file_name)
    print("🔧 build_dir\t:"+build_dir)
    
    # 配置构建参数
    config = EvalConfig()
    config.build_dir = build_dir
    config.cuda_kernel_name = file_name
    
    # CUDA编译标志
    cuda_flags = {
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-std=c++17",
        # "-Idsl_template.cuh",
        "-I" + header_lib_path,  # 包含utils头文件
        # "-DNO_MMA_HGEMM_BIN",  # 启用PyTorch绑定而不是独立二进制
    }
    config.extra_cuda_cflags = list(set(config.extra_cuda_cflags) | cuda_flags)
    
    # 链接标志
    link_flags = [
        "-lcublas",  # 链接cuBLAS库
        "-L/usr/local/cuda/lib64",  # CUDA库路径
    ]
    config.extra_ldflags.extend(link_flags)
    
    # 加载内核
    lib = load_cuda_extension_from_cufile(
        cuda_file=f"{file_name}.cu",
        # cuda_file=[f"{file_name}.cu", "dsl_template.cuh"],
        config=config,
    )
    
    if isinstance(lib, str):
        print(f"❌ build_kernel: {lib}")
        return None
    
    print("✅ build_kernel: success")
    return lib

def compile_cuda_kernel(testcase_root_dir, cufile_basename="kernel", force_compile=False):
    """自动编译CUDA kernel"""
    script_dir = testcase_root_dir
    lib_path = os.path.join(script_dir, CUDA_FOLDER_NAME, 'lib_cuda_kernel.so')
    cuda_source = os.path.join(script_dir, CUDA_FOLDER_NAME, f"{cufile_basename}.cu")
    
    # 检查源文件是否存在
    if not os.path.exists(cuda_source):
        raise FileNotFoundError(f"CUDA Source File Not Found: {cuda_source}")
    
    # 检查是否需要重新编译
    need_compile = force_compile
    if os.path.exists(lib_path):
        lib_mtime = os.path.getmtime(lib_path)
        src_mtime = os.path.getmtime(cuda_source)
        need_compile = src_mtime > lib_mtime
    
    if need_compile:
        print("🔧 Compiling CUDA kernel...")
        
        # 编译命令
        cmd = [
            'nvcc', '-O3', '--shared', '-Xcompiler', '-fPIC',
            '-arch=sm_89', '--use_fast_math',
            '-Wno-deprecated-gpu-targets',
            '-o', lib_path, cuda_source
        ]
        
        try:
            result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, check=True)
            if result.stderr:
                # 过滤掉warning信息
                stderr_lines = result.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines if 'warning' not in line.lower()]
                if error_lines:
                    print(f"Compilation Warning: {result.stderr}")
            
            print("✅ CUDA kernel compiled successfully")
            return lib_path
            
        except subprocess.CalledProcessError as e:
            print(f"Command: {' '.join(cmd)}")
            print(f"Error output: {e.stderr}")
            raise RuntimeError("❌ CUDA compilation failed")
        except FileNotFoundError:
            raise RuntimeError("❌ nvcc not found, please ensure CUDA toolkit is installed and in PATH")
    else:
        print("✅ CUDA kernel is the latest version")
        return lib_path

def load_cuda_kernel(testcase_root_dir, argtypes, restype=None, force_compile=False):
    # 尝试编译kernel
    lib_path = compile_cuda_kernel(testcase_root_dir, "kernel", force_compile=force_compile)
    if not lib_path:
        raise RuntimeError("CUDA kernel compilation failed")
    else:
        print(f"✅ CUDA kernel compiled successfully: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
        
    # Set function signature - now directly receiving GPU pointers
    lib.cuda_kernel.argtypes = argtypes
    lib.cuda_kernel.restype = restype
    
    return lib.cuda_kernel

