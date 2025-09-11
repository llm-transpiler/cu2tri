"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = tl.program_id(axis=0)
#    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    pid_m = pid // grid_n
#    pid_n = pid % grid_n
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use only the first GPU for this tutorial.
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD CDNA2 devices.
# CDNA2 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_cdna2() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
'''
4
(kernel) root@sigma106:/workspace/triton# /root/miniconda3/envs/kernel/bin/python /workspace/triton/python/tutorials/03-matrix-multiplication.py
triton_output_with_fp16_inputs=tensor([[-18.1719,  -3.2539,   9.3984,  ..., -16.6719,  24.2031, -27.8438],
        [ 17.2188,   0.6802, -18.7656,  ...,   0.6616,   0.3096,  37.1562],
        [ 10.0547,  12.5859,   7.7695,  ..., -28.7344, -25.0312,  -5.8320],
        ...,
        [ 17.8125,   2.6836,  11.3906,  ..., -60.1250,  -1.6699, -19.4219],
        [ 67.0625, -12.5469,  48.5312,  ...,  38.0312,  -3.7598, -36.9375],
        [  4.7188, -41.5000, -47.8438,  ...,  -5.3711,  36.5625, -10.5234]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp16_inputs=tensor([[-18.1719,  -3.2539,   9.3984,  ..., -16.6719,  24.2031, -27.8438],
        [ 17.2188,   0.6802, -18.7656,  ...,   0.6616,   0.3096,  37.1562],
        [ 10.0547,  12.5859,   7.7695,  ..., -28.7344, -25.0312,  -5.8320],
        ...,
        [ 17.8125,   2.6836,  11.3906,  ..., -60.1250,  -1.6699, -19.4219],
        [ 67.0625, -12.5469,  48.5312,  ...,  38.0312,  -3.7598, -36.9375],
        [  4.7188, -41.5000, -47.8438,  ...,  -5.3711,  36.5625, -10.5234]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
triton_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0312,  ...,  -1.0039, -43.7812,  -8.3984],
        [ 10.0000,  37.0000,  -5.5703,  ...,  -6.1445,  12.1484, -14.4688],
        [ 19.5625,  -3.0039, -20.0469,  ..., -13.7422,   6.3398,   0.9868],
        ...,
        [-38.5312, -17.2188,  10.0000,  ...,  50.4375,  13.0625,  33.3125],
        [ -9.9531, -25.3281,  46.1875,  ...,  27.1406, -30.4375,  19.2188],
        [  1.1641, -13.2500,  21.8125,  ...,  21.6562,  10.9844,  -5.8711]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  -1.0010, -43.7812,  -8.3984],
        [ 10.0000,  37.0000,  -5.5664,  ...,  -6.1445,  12.1406, -14.4766],
        [ 19.5625,  -3.0078, -20.0469,  ..., -13.7500,   6.3398,   0.9902],
        ...,
        [-38.5312, -17.2188,  10.0000,  ...,  50.4375,  13.0703,  33.3125],
        [ -9.9531, -25.3281,  46.1875,  ...,  27.1406, -30.4375,  19.2344],
        [  1.1621, -13.2500,  21.7969,  ...,  21.6562,  10.9844,  -5.8711]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
matmul-performance-fp16:
         M       N       K      cuBLAS      Triton
0    256.0   256.0   256.0    4.788018    4.599017
1    384.0   384.0   384.0   15.123692   14.099379
2    512.0   512.0   512.0   32.768000   30.174850
3    640.0   640.0   640.0   61.363297   53.630115
4    768.0   768.0   768.0   98.991442   88.473602
5    896.0   896.0   896.0  144.095184  128.450565
6   1024.0  1024.0  1024.0  188.508043  156.430916
7   1152.0  1152.0  1152.0  235.929601  202.439587
8   1280.0  1280.0  1280.0  278.876596  258.524645
9   1408.0  1408.0  1408.0  302.876434  254.681515
10  1536.0  1536.0  1536.0  331.614069  288.525377
11  1664.0  1664.0  1664.0  337.987295  346.528508
12  1792.0  1792.0  1792.0  394.798643  398.737880
13  1920.0  1920.0  1920.0  416.150539  425.558442
14  2048.0  2048.0  2048.0  436.480398  315.620754
15  2176.0  2176.0  2176.0  397.995515  327.880203
16  2304.0  2304.0  2304.0  446.502281  358.878842
17  2432.0  2432.0  2432.0  394.308284  371.804318
18  2560.0  2560.0  2560.0  426.944612  393.535743
19  2688.0  2688.0  2688.0  426.363815  338.310414
20  2816.0  2816.0  2816.0  392.478805  316.834217
21  2944.0  2944.0  2944.0  390.488018  346.987163
22  3072.0  3072.0  3072.0  390.335921  376.937655
23  3200.0  3200.0  3200.0  409.845899  393.656902
24  3328.0  3328.0  3328.0  435.815623  410.024286
25  3456.0  3456.0  3456.0  431.528004  363.364811
26  3584.0  3584.0  3584.0  425.635005  383.946174
27  3712.0  3712.0  3712.0  380.583962  388.186397
28  3840.0  3840.0  3840.0  437.771396  409.244749
29  3968.0  3968.0  3968.0  394.999330  371.670076
30  4096.0  4096.0  4096.0  404.194179  393.257999
matmul-performance-fp8:
         M       N       K      Triton
0    256.0   256.0   256.0    5.242880
1    384.0   384.0   384.0   14.623736
2    512.0   512.0   512.0   32.388449
3    640.0   640.0   640.0   62.060606
4    768.0   768.0   768.0   91.623147
5    896.0   896.0   896.0  136.235438
6   1024.0  1024.0  1024.0  207.126128
7   1152.0  1152.0  1152.0  273.004261
8   1280.0  1280.0  1280.0  356.173905
9   1408.0  1408.0  1408.0  325.479160
10  1536.0  1536.0  1536.0  392.534497
11  1664.0  1664.0  1664.0  475.190060
12  1792.0  1792.0  1792.0  565.505600
13  1920.0  1920.0  1920.0  597.794604
14  2048.0  2048.0  2048.0  525.828526
15  2176.0  2176.0  2176.0  581.188361
16  2304.0  2304.0  2304.0  655.584806
17  2432.0  2432.0  2432.0  634.903161
18  2560.0  2560.0  2560.0  699.983980
19  2688.0  2688.0  2688.0  682.709656
20  2816.0  2816.0  2816.0  627.966120
21  2944.0  2944.0  2944.0  670.909998
22  3072.0  3072.0  3072.0  696.364071
23  3200.0  3200.0  3200.0  742.297920
24  3328.0  3328.0  3328.0  784.646277
25  3456.0  3456.0  3456.0  701.819951
26  3584.0  3584.0  3584.0  733.722450
27  3712.0  3712.0  3712.0  767.795141
28  3840.0  3840.0  3840.0  803.757396
29  3968.0  3968.0  3968.0  719.308455
30  4096.0  4096.0  4096.0  780.052176
'''
'''
xry_tiledsl
3.3.0
4
root@sigma106:/workspace/triton/python/tutorials# python 03-matrix-multiplication.py 
triton_output_with_fp16_inputs=tensor([[-18.1719,  -3.2539,   9.3984,  ..., -16.6719,  24.2031, -27.8438],
        [ 17.2188,   0.6802, -18.7656,  ...,   0.6616,   0.3096,  37.1562],
        [ 10.0547,  12.5859,   7.7695,  ..., -28.7344, -25.0312,  -5.8320],
        ...,
        [ 17.8125,   2.6836,  11.3906,  ..., -60.1250,  -1.6699, -19.4219],
        [ 67.0625, -12.5469,  48.5312,  ...,  38.0312,  -3.7598, -36.9375],
        [  4.7188, -41.5000, -47.8438,  ...,  -5.3711,  36.5625, -10.5234]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp16_inputs=tensor([[-18.1719,  -3.2539,   9.3984,  ..., -16.6719,  24.2031, -27.8438],
        [ 17.2188,   0.6802, -18.7656,  ...,   0.6616,   0.3096,  37.1562],
        [ 10.0547,  12.5859,   7.7695,  ..., -28.7344, -25.0312,  -5.8320],
        ...,
        [ 17.8125,   2.6836,  11.3906,  ..., -60.1250,  -1.6699, -19.4219],
        [ 67.0625, -12.5469,  48.5312,  ...,  38.0312,  -3.7598, -36.9375],
        [  4.7188, -41.5000, -47.8438,  ...,  -5.3711,  36.5625, -10.5234]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
triton_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0312,  ...,  -1.0039, -43.7812,  -8.3984],
        [ 10.0000,  37.0000,  -5.5703,  ...,  -6.1445,  12.1484, -14.4688],
        [ 19.5625,  -3.0039, -20.0469,  ..., -13.7422,   6.3398,   0.9868],
        ...,
        [-38.5312, -17.2188,  10.0000,  ...,  50.4375,  13.0625,  33.3125],
        [ -9.9531, -25.3281,  46.1875,  ...,  27.1406, -30.4375,  19.2188],
        [  1.1641, -13.2500,  21.8125,  ...,  21.6562,  10.9844,  -5.8711]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  -1.0010, -43.7812,  -8.3984],
        [ 10.0000,  37.0000,  -5.5664,  ...,  -6.1445,  12.1406, -14.4766],
        [ 19.5625,  -3.0078, -20.0469,  ..., -13.7500,   6.3398,   0.9902],
        ...,
        [-38.5312, -17.2188,  10.0000,  ...,  50.4375,  13.0703,  33.3125],
        [ -9.9531, -25.3281,  46.1875,  ...,  27.1406, -30.4375,  19.2344],
        [  1.1621, -13.2500,  21.7969,  ...,  21.6562,  10.9844,  -5.8711]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
matmul-performance-fp16:
         M       N       K      cuBLAS      Triton
0    256.0   256.0   256.0    5.115005    4.798975
1    384.0   384.0   384.0   14.807297   14.444669
2    512.0   512.0   512.0   31.895849   32.263877
3    640.0   640.0   640.0   56.496552   57.286714
4    768.0   768.0   768.0   92.220036   83.269271
5    896.0   896.0   896.0  143.634813  125.931917
6   1024.0  1024.0  1024.0  185.383607  144.943550
7   1152.0  1152.0  1152.0  237.690266  192.256515
8   1280.0  1280.0  1280.0  277.107827  249.660961
9   1408.0  1408.0  1408.0  300.787645  244.679983
10  1536.0  1536.0  1536.0  330.645874  289.262336
11  1664.0  1664.0  1664.0  344.455953  339.781931
12  1792.0  1792.0  1792.0  402.756520  399.180434
13  1920.0  1920.0  1920.0  426.584386  426.584386
14  2048.0  2048.0  2048.0  439.697722  299.927882
15  2176.0  2176.0  2176.0  397.504159  325.395026
16  2304.0  2304.0  2304.0  446.241624  358.710432
17  2432.0  2432.0  2432.0  397.797724  370.272987
18  2560.0  2560.0  2560.0  427.641102  406.740097
19  2688.0  2688.0  2688.0  432.593662  342.414047
20  2816.0  2816.0  2816.0  397.169796  324.005717
21  2944.0  2944.0  2944.0  401.195706  347.516474
22  3072.0  3072.0  3072.0  391.263072  375.921029
23  3200.0  3200.0  3200.0  411.327579  389.205641
24  3328.0  3328.0  3328.0  434.091090  415.684147
25  3456.0  3456.0  3456.0  435.057366  361.734470
26  3584.0  3584.0  3584.0  422.076067  383.843733
27  3712.0  3712.0  3712.0  407.952397  393.538711
28  3840.0  3840.0  3840.0  433.136776  419.107530
29  3968.0  3968.0  3968.0  401.229544  374.198939
30  4096.0  4096.0  4096.0  408.616441  397.571718
matmul-performance-fp8:
         M       N       K      Triton
0    256.0   256.0   256.0    4.766255
1    384.0   384.0   384.0   15.059336
2    512.0   512.0   512.0   35.098778
3    640.0   640.0   640.0   58.306053
4    768.0   768.0   768.0   97.963852
5    896.0   896.0   896.0  145.024824
6   1024.0  1024.0  1024.0  193.956262
7   1152.0  1152.0  1152.0  273.786491
8   1280.0  1280.0  1280.0  357.144414
9   1408.0  1408.0  1408.0  354.587042
10  1536.0  1536.0  1536.0  435.143926
11  1664.0  1664.0  1664.0  515.143435
12  1792.0  1792.0  1792.0  604.473233
13  1920.0  1920.0  1920.0  615.254517
14  2048.0  2048.0  2048.0  539.027011
15  2176.0  2176.0  2176.0  581.713377
16  2304.0  2304.0  2304.0  655.584806
17  2432.0  2432.0  2432.0  681.594258
18  2560.0  2560.0  2560.0  751.667368
19  2688.0  2688.0  2688.0  685.795373
20  2816.0  2816.0  2816.0  626.697207
21  2944.0  2944.0  2944.0  680.065276
22  3072.0  3072.0  3072.0  730.620690
23  3200.0  3200.0  3200.0  779.596504
24  3328.0  3328.0  3328.0  836.500167
25  3456.0  3456.0  3456.0  730.433256
26  3584.0  3584.0  3584.0  769.123893
27  3712.0  3712.0  3712.0  801.583483
28  3840.0  3840.0  3840.0  859.801720
29  3968.0  3968.0  3968.0  769.411976
30  4096.0  4096.0  4096.0  812.594311
'''