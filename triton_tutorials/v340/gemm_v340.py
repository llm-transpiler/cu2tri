from typing import Callable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    c = torch.matmul(a, b.T)
    return c

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC

def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w, pre_hook=pre_hook) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64,128] \
        for s in ([3,4]) \
        for w in [4,8] \
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    # configs=[
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
    #                   num_warps=2),
    #     triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
    #                   num_warps=2),
    #     # Good config for fp8 inputs.
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
    #                   num_warps=4)
    # ],
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    # print(f"matmul: a.shape={a.shape}, b.shape={b.shape}, a.dtype={a.dtype}, b.dtype={b.dtype}")
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [128]  #
        for BN in [128, 256]  #
        for BK in [64, 128]  #
        for s in ([2, 3, 4])  #
        for w in [4, 8]  #
        for SUBTILE in [True, False]  #
    ]

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)



def matmul_tma_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c

from typing import Callable, List
def benchmark_kernel(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 1000,
    quantiles: List[float] = [0.2, 0.5, 0.8]
) -> float:
    ret = []
    for _ in range(10):
        res = _simple_perf(kernel_func, inputs, warmup, iterations, quantiles)
        ret.append(res["median"])
    ret = torch.tensor(ret, dtype=torch.float)
    return ret.median().item()

def _simple_perf(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 1000,
    quantiles: List[float] = [0.2, 0.5, 0.8],
) -> float | dict:
    # 预热
    for _ in range(warmup):
        kernel_func(*inputs)
    
    torch.cuda.synchronize()
    
    # 创建events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    # 测量时间
    torch.cuda.synchronize()
    for i in range(iterations):
        cache.zero_()
        start_events[i].record()
        kernel_func(*inputs)
        end_events[i].record()
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
    
    # if get_mean:
    #     return torch.mean(times).item()

    ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    
    res = {
        'mean': torch.mean(times).item(),
        'total': torch.sum(times).item(),
        'var': torch.var(times).item(),
        'median': torch.median(times).item(),
        **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
    }
    
    return res

def bench(K, dtype, reps=10000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    ms = benchmark_kernel(matmul, (a, b), warmup_reps, reps)
    print(f"simple_matmul: {ms} ms")
    b = b.T.contiguous()
    # reps = 1000
    # warmup_reps = 1000
    ms = benchmark_kernel(torch_matmul, (a, b), warmup_reps, reps)
    print(f"torch_matmul: {ms} ms")
    ms = benchmark_kernel(matmul, (a, b.T), warmup_reps, reps)
    print(f"simple_matmul: {ms} ms")
    ms = benchmark_kernel(matmul_tma_persistent, (a, b, True), warmup_reps, reps)
    print(f"matmul_tma_persistent: {ms} ms")
    
if __name__ == "__main__":
    bench(512, torch.float16)

'''
triton: 3.4.0
rep = 1000

GPU 4 L20
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
simple_matmul: 0.6845759749412537 ms
torch_matmul: 0.6453120112419128 ms
simple_matmul: 0.6986879706382751 ms
matmul_tma_persistent: 1.1872639656066895 ms

GPU 2 H100
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
simple_matmul: 0.22281600534915924 ms
torch_matmul: 0.176256000995636 ms
simple_matmul: 0.22329600155353546 ms
matmul_tma_persistent: 0.18825599551200867 ms
'''

'''
rep = 10000

GPU 2 H100
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
simple_matmul: 0.2260800004005432 ms
torch_matmul: 0.17740799486637115 ms
simple_matmul: 0.2271679937839508 ms
matmul_tma_persistent: 0.19155199825763702 ms


H100 换成tutorial中的config:
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
simple_matmul: 0.2168319970369339 ms
torch_matmul: 0.17670400440692902 ms
simple_matmul: 0.2542079985141754 ms
matmul_tma_persistent: 0.18652799725532532 ms

感觉是同一个kernel复用了cache导致的? rm -r ~/.triton/cache
(base) root@sigma106:/workspace/triton# rm -r ~/.triton/cache
第一次行优先不跑了:
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
torch_matmul: 0.1756799966096878 ms
simple_matmul: 0.23468799889087677 ms
matmul_tma_persistent: 0.1886720061302185 ms

换回原来的config:
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
torch_matmul: 0.17497600615024567 ms
simple_matmul: 0.22313599288463593 ms
matmul_tma_persistent: 0.18851199746131897 ms
'''
'''
s_xuruiyuan@sigma106:~$ docker exec -it xry_gpudsl bash
(base) root@sigma106:/#  /root/miniconda3/envs/kernel/bin/python
Python 3.12.11 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:09:17) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import triton
>>> triton.__version__
'3.4.0'
>>>
(base) root@sigma106:/# cd /workspace/triton
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm.py
simple_matmul: 0.21699200570583344 ms
torch_matmul: 0.17475199699401855 ms
simple_matmul: 0.21846400201320648 ms
matmul_tma_persistent: 0.18649600446224213 ms

使用多的config:
(base) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/gemm_v340.py
simple_matmul: 0.21145600080490112 ms
torch_matmul: 0.17555199563503265 ms
simple_matmul: 0.2542079985141754 ms
matmul_tma_persistent: 0.18435199558734894 ms
'''