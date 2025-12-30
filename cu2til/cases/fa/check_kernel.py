# %%
#!/usr/bin/env python3
"""
HGEMM CUDA内核测试，与Flash Attention对比
基于原有test_kernel.py的模式，专门用于HGEMM kernel与Flash Attention的性能对比
"""
import os
import random
import sys
import traceback
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_printoptions(
    precision=6, threshold=8, edgeitems=3, linewidth=120, sci_mode=False
)

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 1 for debug and check
os.environ["TORCH_CUDA_ARCH_LIST"] = "Ada"
CASE_NICKNAME = "fa"
# 在Jupyter notebook中，__file__不存在，使用当前工作目录或指定路径
import os
# current_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from cu2til.cases.dev.fa.torch.ref import naive_attn
# 添加eval_路径
sys.path.insert(0, '/workspace')  # 添加workspace根目录

from eval_.common.loader import load_cuda_extension_from_cufile
from eval_.common.config import EvalConfig

from eval_.common.benchmark import benchmark_kernel, benchmark_simple_e2e
def set_rand_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device_name

def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())

def build_kernel(cufile_basename, kernel_name, header_lib_path="/workspace/cu2til/include"):
    """构建单个FA CUDA内核"""
    print(f"🔧 Build {CASE_NICKNAME} ({cufile_basename}.cu) Kernel: {kernel_name}...")
    
    from pathlib import Path
    
    # 进入对应的kernel目录
    kernel_dir = Path(f"./cuda/{kernel_name}")
    if not kernel_dir.exists():
        print(f"❌ Kernel directory not found: {kernel_dir}")
        return None
        
    original_dir = os.getcwd()
    os.chdir(kernel_dir)
    
    try:
        # 配置构建参数
        config = EvalConfig()
        config.build_dir = "build"
        config.cuda_kernel_name = cufile_basename
        
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
            "-I" + header_lib_path,  # 包含utils头文件
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
            cuda_file=f"{cufile_basename}.cu",
            config=config,
        )
        
        if isinstance(lib, str):
            print(f"❌ Compile failed: {lib}")
            return None
        
        print(f"✅ {CASE_NICKNAME} CUDA Kernel [{cufile_basename}] built successfully!")
        return lib
        
    finally:
        os.chdir(original_dir)

def pretty_print_line(m: str = "", sep: str = "-", width: int = 120):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)

pretty_print_line()

def tflops(B: int, H: int, N: int, D: int, secs: float = 1.0):
    """计算Flash Attention等效的TFLOPS (只考虑两个主要矩阵乘法)"""
    # Q @ K^T: B*H*N*N*(2*D-1) ≈ B*H*N*N*2*D  
    flops_qk = B * H * N * N * 2 * D
    # P @ V: B*H*N*D*(2*N-1) ≈ B*H*N*D*2*N
    flops_pv = B * H * N * D * 2 * N
    
    total_flops = flops_qk + flops_pv
    tflops = total_flops * 1e-12 / secs
    return tflops

def check_all_close(ref: torch.Tensor, test: torch.Tensor, tag: str = "test"):
    """检查两个张量是否接近"""
    if any((ref is None, test is None)):
        return
        
    diff = torch.abs(ref - test)
    all_close = str(torch.allclose(ref, test, atol=1e-2, rtol=1e-2))
    pretty_print_line(
        f"{tag:<25}, all close: {all_close:<6}, "
        f"max diff: {diff.max().item():.6f}"
    )

batch_size_list = [1, 4, 8]
head_num_list = [1, 4, 8]
seq_len_list = [1024, 2048, 4096]
head_dim_list = [64, 128]

def get_qkv(B, H, N, D):
    """获取Flash Attention的QKV矩阵"""
    q = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    k = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    v = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    return q, k, v

def get_test_shapes():
    # return [(1, 1, 1024, 64), (1, 1, 1024, 128)]
    test_shapes = []
    for B in batch_size_list:
        for H in head_num_list:
            for N in seq_len_list:
                for D in head_dim_list:
                    test_shapes.append((B, H, N, D))
    dim = 2048
    bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]

    # for headdim in [64, 128, 256]:
    for headdim in [64, 128]:
        nheads = dim // headdim # [32, 16, 8] # 2048 // 64 = 32, 2048 // 128 = 16, 2048 // 256 = 8
        for batch_size, seqlen in bs_seqlen_vals:
            test_shapes.append((batch_size, nheads, seqlen, headdim))
    return test_shapes

# 预先编译两个kernel
kernel_name = "flash_attn_mma_stages_split_q_shared_kv"

# %%

current_dir = os.getcwd()
current_dir

# %%
import os

current_dir = os.getcwd()
# print(os.listdir(current_dir))
print(os.listdir(f"{current_dir}/cuda"))

# %%
cu_files = [
    # "ref",
    # "kernel",
    # "kernel_expanded",
    # "kernel_cleaned",
    "kernel_prim",
    # "kernel_expanded_converted",
    # "kernel_til",
]
basename2kernel = {}
# %%

def test_correctness():
    pretty_print_line(f"🔧 Start kernel correctness test: {kernel_name}...")
    for cufile_basename in cu_files:
        try:
            lib = build_kernel(cufile_basename, kernel_name)
            if lib is not None:
                basename2kernel[cufile_basename] = cur_kernel = getattr(lib, kernel_name)
                pretty_print_line(f"✅ Kernel {kernel_name} compiled and loaded successfully")
            else:
                pretty_print_line(f"❌ Kernel {kernel_name} compilation failed")
        except Exception as e:
            print(f"❌ Kernel {kernel_name} compilation failed: {e}")
        for batch_size, head_num, seq_len, head_dim in get_test_shapes():
            try:
                q, k, v = get_qkv(batch_size, head_num, seq_len, head_dim)
                o = torch.zeros_like(q)
                ref = naive_attn(q, k, v)
                cur_kernel(q, k, v, o)
                # check_all_close(ref, o, tag=f"B={batch_size}, H={head_num}, N={seq_len}, D={head_dim}")
                if not torch.allclose(ref, o, atol=1e-2, rtol=1e-2):
                    pretty_print_line(f"Testing kernel: {kernel_name}, B={batch_size}, H={head_num}, N={seq_len}, D={head_dim}")
                    print(f"\tref_output: {ref.shape}, kernel_output: {o.shape}")
                    check_all_close(ref, o, tag="kernel")
            except Exception as e:
                if not "headdim not support" in str(e):
                    print(f"❌ Kernel {kernel_name} failed: {e}...\n{traceback.format_exc()}...")
                    return

REF_KERNEL_BASENAME = "kernel"
def test_performance():
    if REF_KERNEL_BASENAME not in cu_files:
        return
    
    for cufile_basename, kernel in basename2kernel.items():
        if cufile_basename == REF_KERNEL_BASENAME:
            continue  # 跳过ref，已经作为baseline处理

        pretty_print_line(f"🔧 Start kernel performance test: {cufile_basename}...")
        if kernel is None:
            try:
                lib = build_kernel(cufile_basename, kernel_name)
                if lib is not None:
                    kernel = basename2kernel[cufile_basename] = getattr(lib, kernel_name)
                    pretty_print_line(f"✅ Kernel {cufile_basename} compiled and loaded successfully")
                else:
                    pretty_print_line(f"❌ Kernel {cufile_basename} compilation failed")
                    continue
            except Exception as e:
                print(f"❌ Kernel {cufile_basename} compilation failed: {e}")
                continue
        if basename2kernel[REF_KERNEL_BASENAME] is None:
            return
        for batch_size, head_num, seq_len, head_dim in get_test_shapes():
            pretty_print_line(f"Performance test: {cufile_basename}, B={batch_size}, H={head_num}, N={seq_len}, D={head_dim}")
            try:
                q, k, v = get_qkv(batch_size, head_num, seq_len, head_dim)
                o = torch.zeros_like(q)
                kernel_ms = benchmark_kernel(kernel, (q, k, v, o))
                ref_ms = benchmark_kernel(basename2kernel[REF_KERNEL_BASENAME], (q, k, v, o))
                if abs(ref_ms / kernel_ms - 1) < 0.05:
                    continue
                print(f"kernel:\t{kernel_ms:.4f}ms, ref:\t{ref_ms:.4f}ms, speedup:\t{ref_ms / kernel_ms:.2f}x")
                
            except Exception as e:
                print(f"❌ Kernel {cufile_basename} failed: {e}")
            finally:
                del q, k, v, o


# %%

if __name__ == "__main__":
    test_correctness()
    test_performance()


