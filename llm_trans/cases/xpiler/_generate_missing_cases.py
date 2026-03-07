import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SRC_CUDA_DIR = WORKSPACE_ROOT / "third_party" / "xpiler-eval" / "KernelBench" / "CUDA"
DEST_CASE_DIR = WORKSPACE_ROOT / "cu2til" / "cases" / "xpiler"


def format_tuple(values):
    if not values:
        return "()"
    if len(values) == 1:
        return f"({values[0]},)"
    return "(" + ", ".join(str(v) for v in values) + ")"


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def check_cuda_template(title: str) -> str:
    return dedent(
        f"""\
        import os
        import sys
        import argparse
        from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes, cuda_output_tensor_transform
        from torch_.ref import torch_kernel
        from llm_trans.tools.checker import check_cuda_vs_torch

        TESTCASE_ROOT_DIR = os.path.dirname(__file__)
        sys.path.insert(0, TESTCASE_ROOT_DIR)

        def parse_args():
            parser = argparse.ArgumentParser(description="{title} CUDA kernel test")
            parser.add_argument("--compile-only", action="store_true", help="Only compile the CUDA kernel without running correctness checks")
            parser.add_argument("--no-perf", action="store_true", help="Disable performance benchmarking")
            return parser.parse_args()

        if __name__ == "__main__":
            args = parse_args()
            params = Params()
            check_cuda_vs_torch(
                testcase_root_dir=TESTCASE_ROOT_DIR,
                get_cuda_torch_inputs=get_cuda_torch_inputs,
                params=params,
                torch_kernel=torch_kernel,
                get_cuda_argtypes=get_cuda_argtypes,
                output_tensor_transform=cuda_output_tensor_transform,
                enable_perf=not args.no_perf,
                compile_only=args.compile_only,
            )
        """
    )


def check_triton_template(title: str) -> str:
    return dedent(
        f"""\
        import os
        import sys
        import argparse
        from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform
        from torch_.ref import torch_kernel
        from triton_.kernel import triton_kernel
        from llm_trans.tools.checker import compare_results

        TESTCASE_ROOT_DIR = os.path.dirname(__file__)
        sys.path.insert(0, TESTCASE_ROOT_DIR)

        def parse_args():
            parser = argparse.ArgumentParser(description="{title} Triton kernel test")
            parser.add_argument("--no-perf", action="store_true", help="Placeholder flag for compatibility")
            return parser.parse_args()

        def main():
            params = Params()
            _, torch_inputs, _ = get_cuda_torch_inputs(params)
            torch_output = cuda_output_tensor_transform(torch_kernel(*torch_inputs))
            triton_output = cuda_output_tensor_transform(triton_kernel(*torch_inputs))
            compare_results(torch_output, triton_output, test_type=["PyTorch", "Triton"])

        if __name__ == "__main__":
            parse_args()
            main()
        """
    )


TRITON_WRAPPER = dedent(
    """\
    from torch_.ref import torch_kernel as _torch_kernel

    def triton_kernel(*args):
        # Placeholder Triton wrapper that currently falls back to the PyTorch reference.
        # Replace with a real Triton implementation when an optimized Triton version is available.
        return _torch_kernel(*args)
    """
)


@dataclass
class CaseFiles:
    get_data: str
    torch_ref: str
    cuda_kernel: str


def common_cuda_prelude(extra_includes=None):
    headers = {"#include <cuda_runtime.h>"}
    if extra_includes:
        headers.update(extra_includes)
    return "\n".join(sorted(headers)) + "\n\n"


def generate_batchnorm(name, dims):
    if len(dims) < 2:
        raise ValueError(f"{name}: batchnorm expects at least two dimensions")
    batch_size = dims[0]
    num_channels = dims[1]
    spatial_size = product(dims[2:]) if len(dims) > 2 else 1
    total_elems = product(dims)
    shape_literal = format_tuple(dims)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            shape: tuple = {shape_literal}
            batch_size: int = {batch_size}
            num_channels: int = {num_channels}
            spatial_size: int = {spatial_size}
            eps: float = 1e-5

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # output
                ctypes.c_void_p,  # running mean
                ctypes.c_void_p,  # running variance
                ctypes.c_void_p,  # gamma
                ctypes.c_void_p,  # beta
                ctypes.c_int,     # batch size
                ctypes.c_int,     # num channels
                ctypes.c_int      # spatial size
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            running_mean = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
            running_var = torch.rand(params.num_channels, dtype=torch.float32, device="cuda") + 0.5
            gamma = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
            beta = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
            output = torch.empty_like(x)
            cuda_all_inputs = [x, output, running_mean, running_var, gamma, beta, params.batch_size, params.num_channels, params.spatial_size]
            torch_all_inputs = [x, running_mean, running_var, gamma, beta]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch
        import torch.nn.functional as F

        def torch_kernel(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
            return F.batch_norm(x, running_mean, running_var, weight=gamma, bias=beta, training=False, momentum=0.0, eps=1e-5)
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <math.h>"}) + dedent(
        """\
        __global__ void batchnorm_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const float* __restrict__ mean,
            const float* __restrict__ var,
            const float* __restrict__ gamma,
            const float* __restrict__ beta,
            int batch_size,
            int num_channels,
            int spatial_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = batch_size * num_channels * spatial_size;
            if (idx >= total) {
                return;
            }

            int channel = (idx / spatial_size) % num_channels;
            float norm = (input[idx] - mean[channel]) / sqrtf(var[channel] + 1e-5f);
            output[idx] = norm * gamma[channel] + beta[channel];
        }

        extern "C" void cuda_kernel(
            const float* input,
            float* output,
            const float* mean,
            const float* var,
            const float* gamma,
            const float* beta,
            int batch_size,
            int num_channels,
            int spatial_size
        ) {
            int total = batch_size * num_channels * spatial_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            batchnorm_kernel<<<blocks, threads>>>(input, output, mean, var, gamma, beta, batch_size, num_channels, spatial_size);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_concat(name, dims):
    if len(dims) != 4:
        raise ValueError(f"{name}: concat expects 4D tensors (N, C, H, W)")
    N, C, H, W = dims
    output_shape = (N, C * 2, H, W)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            input_shape: tuple = {format_tuple(dims)}
            output_shape: tuple = {format_tuple(output_shape)}
            N: int = {N}
            C: int = {C}
            H: int = {H}
            W: int = {W}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input1
                ctypes.c_void_p,  # input2
                ctypes.c_void_p,  # output
                ctypes.c_int,     # N
                ctypes.c_int,     # C
                ctypes.c_int,     # H
                ctypes.c_int      # W
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            a = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
            b = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
            output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
            cuda_all_inputs = [a, b, output, params.N, params.C, params.H, params.W]
            torch_all_inputs = [a, b]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.cat((a, b), dim=1)
        """
    )

    cuda_kernel = common_cuda_prelude() + dedent(
        """\
        __global__ void concat_kernel(
            const float* __restrict__ input1,
            const float* __restrict__ input2,
            float* __restrict__ output,
            int N,
            int C,
            int H,
            int W
        ) {
            int cout = C * 2;
            int hw = H * W;
            int total = N * cout * hw;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }

            int n = idx / (cout * hw);
            int rem = idx % (cout * hw);
            int c = rem / hw;
            int offset = rem % hw;

            int base_input = n * C * hw;
            if (c < C) {
                output[idx] = input1[base_input + c * hw + offset];
            } else {
                int c2 = c - C;
                output[idx] = input2[base_input + c2 * hw + offset];
            }
        }

        extern "C" void cuda_kernel(
            const float* input1,
            const float* input2,
            float* output,
            int N,
            int C,
            int H,
            int W
        ) {
            int total = N * (C * 2) * H * W;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            concat_kernel<<<blocks, threads>>>(input1, input2, output, N, C, H, W);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_dense(name, dims):
    if len(dims) != 3:
        raise ValueError(f"{name}: dense expects (M, K, N)")
    m, k, n = dims

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            m: int = {m}
            k: int = {k}
            n: int = {n}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # A
                ctypes.c_void_p,  # B
                ctypes.c_void_p,  # bias
                ctypes.c_void_p,  # output
                ctypes.c_int,     # M
                ctypes.c_int,     # K
                ctypes.c_int      # N
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            a = torch.randn((params.m, params.k), dtype=torch.float16, device="cuda")
            b = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
            bias = torch.randn((params.n,), dtype=torch.float32, device="cuda")
            output = torch.empty((params.m, params.n), dtype=torch.float32, device="cuda")
            cuda_all_inputs = [a, b, bias, output, params.m, params.k, params.n]
            torch_all_inputs = [a, b, bias]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            result = torch.matmul(a.float(), b.float())
            return result + bias
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <cuda_fp16.h>"}) + dedent(
        """\
        __global__ void dense_kernel(
            const half* __restrict__ a,
            const half* __restrict__ b,
            const float* __restrict__ bias,
            float* __restrict__ output,
            int M,
            int K,
            int N
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) {
                return;
            }

            float acc = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                float a_val = __half2float(a[row * K + kk]);
                float b_val = __half2float(b[kk * N + col]);
                acc += a_val * b_val;
            }
            output[row * N + col] = acc + bias[col];
        }

        extern "C" void cuda_kernel(
            const half* a,
            const half* b,
            const float* bias,
            float* output,
            int M,
            int K,
            int N
        ) {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            dense_kernel<<<grid, block>>>(a, b, bias, output, M, K, N);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_gatemlp(name, dims):
    if len(dims) != 3:
        raise ValueError(f"{name}: gatemlp expects (batch, K, N)")
    batch, k, n = dims

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            batch: int = {batch}
            k: int = {k}
            n: int = {n}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # X
                ctypes.c_void_p,  # A
                ctypes.c_void_p,  # B
                ctypes.c_void_p,  # output
                ctypes.c_int,     # batch
                ctypes.c_int,     # K
                ctypes.c_int      # N
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn((params.batch, params.k), dtype=torch.float16, device="cuda")
            a = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
            b = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
            output = torch.empty((params.batch, params.n), dtype=torch.float32, device="cuda")
            cuda_all_inputs = [x, a, b, output, params.batch, params.k, params.n]
            torch_all_inputs = [x, a, b]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch
        import torch.nn.functional as F

        def torch_kernel(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x64 = x.float().double()
            o1 = torch.matmul(x64, a.float().double())
            o2 = torch.matmul(x64, b.float().double())
            return (F.silu(o1) * o2).float()
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <cuda_fp16.h>", "#include <math.h>"}) + dedent(
        """\
        __global__ void gatemlp_kernel(
            const half* __restrict__ x,
            const half* __restrict__ a,
            const half* __restrict__ b,
            float* __restrict__ output,
            int batch,
            int K,
            int N
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= batch || col >= N) {
                return;
            }

            double acc1 = 0.0;
            double acc2 = 0.0;
            for (int kk = 0; kk < K; ++kk) {
                double x_val = static_cast<double>(__half2float(x[row * K + kk]));
                double a_val = static_cast<double>(__half2float(a[kk * N + col]));
                double b_val = static_cast<double>(__half2float(b[kk * N + col]));
                acc1 += x_val * a_val;
                acc2 += x_val * b_val;
            }

            double silu = acc1 / (1.0 + exp(-acc1));
            output[row * N + col] = static_cast<float>(silu * acc2);
        }

        extern "C" void cuda_kernel(
            const half* x,
            const half* a,
            const half* b,
            float* output,
            int batch,
            int K,
            int N
        ) {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
            gatemlp_kernel<<<grid, block>>>(x, a, b, output, batch, K, N);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_gather(name, dims):
    if len(dims) != 3:
        raise ValueError(f"{name}: gather expects (D0, D1, D2)")
    d0, d1, d2 = dims
    num_indices = min(8, d2)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            data_shape: tuple = {format_tuple(dims)}
            num_indices: int = {num_indices}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # params
                ctypes.c_void_p,  # indices
                ctypes.c_void_p,  # output
                ctypes.c_int,     # dim0
                ctypes.c_int,     # dim1
                ctypes.c_int,     # dim2
                ctypes.c_int      # num_indices
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            params_tensor = torch.randn(params.data_shape, dtype=torch.float32, device="cuda")
            indices = torch.randint(0, params.data_shape[-1], (params.num_indices,), dtype=torch.int64, device="cuda")
            output = torch.empty((params.data_shape[0], params.data_shape[1], params.num_indices), dtype=torch.float32, device="cuda")
            cuda_all_inputs = [params_tensor, indices, output, params.data_shape[0], params.data_shape[1], params.data_shape[2], params.num_indices]
            torch_all_inputs = [params_tensor, indices]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            expanded_indices = indices.view(1, 1, -1).expand(params.size(0), params.size(1), -1)
            return torch.gather(params, dim=2, index=expanded_indices)
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <stdint.h>"}) + dedent(
        """\
        __global__ void gather_kernel(
            const float* __restrict__ params,
            const int64_t* __restrict__ indices,
            float* __restrict__ output,
            int dim0,
            int dim1,
            int dim2,
            int num_indices
        ) {
            int total = dim0 * dim1 * num_indices;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }

            int stride = dim1 * num_indices;
            int i0 = idx / stride;
            int rem = idx % stride;
            int i1 = rem / num_indices;
            int n = rem % num_indices;

            int64_t source = indices[n];
            float value = 0.0f;
            if (source >= 0 && source < dim2) {
                value = params[(i0 * dim1 + i1) * dim2 + static_cast<int>(source)];
            }
            output[idx] = value;
        }

        extern "C" void cuda_kernel(
            const float* params,
            const int64_t* indices,
            float* output,
            int dim0,
            int dim1,
            int dim2,
            int num_indices
        ) {
            int total = dim0 * dim1 * num_indices;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            gather_kernel<<<blocks, threads>>>(params, indices, output, dim0, dim1, dim2, num_indices);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_instancenorm(name, dims):
    if len(dims) < 2:
        raise ValueError(f"{name}: instancenorm expects at least 2 dimensions")
    batch = dims[0]
    channels = dims[1]
    spatial = product(dims[2:]) if len(dims) > 2 else 1
    shape_literal = format_tuple(dims)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            shape: tuple = {shape_literal}
            batch: int = {batch}
            channels: int = {channels}
            spatial: int = {spatial}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # output
                ctypes.c_void_p,  # gamma
                ctypes.c_void_p,  # beta
                ctypes.c_int,     # batch
                ctypes.c_int,     # channels
                ctypes.c_int      # spatial
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            gamma = torch.randn(params.channels, dtype=torch.float32, device="cuda")
            beta = torch.randn(params.channels, dtype=torch.float32, device="cuda")
            output = torch.empty_like(x)
            cuda_all_inputs = [x, output, gamma, beta, params.batch, params.channels, params.spatial]
            torch_all_inputs = [x, gamma, beta]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch
        import torch.nn.functional as F

        def torch_kernel(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
            return F.instance_norm(x, weight=gamma, bias=beta, eps=1e-5)
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <math.h>"}) + dedent(
        """\
        __global__ void instancenorm_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const float* __restrict__ gamma,
            const float* __restrict__ beta,
            int batch,
            int channels,
            int spatial
        ) {
            int total = batch * channels * spatial;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }

            int sub = spatial;
            int channel = (idx / sub) % channels;
            int instance = idx / (channels * sub);

            const float* slice = input + (instance * channels + channel) * spatial;
            float mean = 0.0f;
            for (int i = 0; i < spatial; ++i) {
                mean += slice[i];
            }
            mean /= static_cast<float>(spatial);

            float var = 0.0f;
            for (int i = 0; i < spatial; ++i) {
                float diff = slice[i] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(spatial);

            float scale = gamma[channel];
            float shift = beta[channel];
            float normalized = (input[idx] - mean) / sqrtf(var + 1e-5f);
            output[idx] = normalized * scale + shift;
        }

        extern "C" void cuda_kernel(
            const float* input,
            float* output,
            const float* gamma,
            const float* beta,
            int batch,
            int channels,
            int spatial
        ) {
            int total = batch * channels * spatial;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            instancenorm_kernel<<<blocks, threads>>>(input, output, gamma, beta, batch, channels, spatial);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_reduction(name, dims, op):
    if not dims:
        raise ValueError(f"{name}: reduction expects at least one dimension")
    rows = dims[0]
    inner = product(dims[1:]) if len(dims) > 1 else 1
    if inner == 0:
        inner = 1
    shape_literal = format_tuple(dims)
    output_shape = dims[1:]

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            input_shape: tuple = {shape_literal}
            output_shape: tuple = {format_tuple(output_shape)}
            rows: int = {rows}
            inner: int = {inner}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # output
                ctypes.c_int,     # rows
                ctypes.c_int      # inner
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
            output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
            cuda_all_inputs = [x, output, params.rows, params.inner]
            torch_all_inputs = [x]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    reduction_call = {
        "max": "torch.amax",
        "mean": "torch.mean",
        "min": "torch.amin",
        "sum": "torch.sum",
    }[op]
    op_expression = f"{reduction_call}(x, dim=0)"

    torch_ref = dedent(
        f"""\
        import torch

        def torch_kernel(x: torch.Tensor) -> torch.Tensor:
            return {op_expression}
        """
    )

    if op == "max":
        init_expr = "-FLT_MAX"
        comb_expr = "val = fmaxf(val, input[r * inner + idx]);"
        finalize_expr = "output[idx] = val;"
        extra_headers = {"#include <float.h>", "#include <math.h>"}
    elif op == "min":
        init_expr = "FLT_MAX"
        comb_expr = "val = fminf(val, input[r * inner + idx]);"
        finalize_expr = "output[idx] = val;"
        extra_headers = {"#include <float.h>", "#include <math.h>"}
    elif op == "sum":
        init_expr = "0.0f"
        comb_expr = "val += input[r * inner + idx];"
        finalize_expr = "output[idx] = val;"
        extra_headers = None
    elif op == "mean":
        init_expr = "0.0f"
        comb_expr = "val += input[r * inner + idx];"
        finalize_expr = "output[idx] = val / static_cast<float>(rows);"
        extra_headers = None
    else:
        raise ValueError(op)

    cuda_kernel = common_cuda_prelude(extra_headers) + dedent(
        f"""\
        __global__ void reduction_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int rows,
            int inner
        ) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= inner) {{
                return;
            }}

            float val = {init_expr};
            for (int r = 0; r < rows; ++r) {{
                {comb_expr}
            }}
            {finalize_expr}
        }}

        extern "C" void cuda_kernel(
            const float* input,
            float* output,
            int rows,
            int inner
        ) {{
            int threads = 256;
            int blocks = (inner + threads - 1) / threads;
            reduction_kernel<<<blocks, threads>>>(input, output, rows, inner);
        }}
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_scatter(name, dims):
    if len(dims) != 4:
        raise ValueError(f"{name}: scatter expects 4D tensors (N, C, H, W)")
    N, C, H, W = dims

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            shape: tuple = {format_tuple(dims)}
            N: int = {N}
            C: int = {C}
            H: int = {H}
            W: int = {W}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # indices
                ctypes.c_void_p,  # output
                ctypes.c_int,     # N
                ctypes.c_int,     # C
                ctypes.c_int,     # H
                ctypes.c_int      # W
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            rows = params.N * params.C * params.H
            perms = [torch.randperm(params.W, device="cuda", dtype=torch.int32) for _ in range(rows)]
            indices = torch.stack(perms, dim=0).view(params.shape)
            output = x.clone()
            cuda_all_inputs = [x, indices, output, params.N, params.C, params.H, params.W]
            torch_all_inputs = [x, indices]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            output = x.clone()
            output.scatter_(dim=3, index=indices.to(torch.long), src=x)
            return output
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <cuda_runtime.h>"}) + dedent(
        """\
        __global__ void scatter_kernel(
            const float* __restrict__ input,
            const int* __restrict__ indices,
            float* __restrict__ output,
            int rows,
            int W
        ) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) {
                return;
            }

            const float* input_row = input + row * W;
            const int* index_row = indices + row * W;
            float* output_row = output + row * W;

        for (int target = 0; target < W; ++target) {
            for (int w = 0; w < W; ++w) {
                if (index_row[w] == target) {
                    output_row[target] = input_row[w];
                    break;
                }
            }
        }
        }

        extern "C" void cuda_kernel(
            const float* input,
            const int* indices,
            float* output,
            int N,
            int C,
            int H,
            int W
        ) {
            int total = N * C * H * W;
            cudaMemcpy(output, input, total * sizeof(float), cudaMemcpyDeviceToDevice);
            int rows = N * C * H;
            int threads = 256;
            int blocks = (rows + threads - 1) / threads;
            scatter_kernel<<<blocks, threads>>>(input, indices, output, rows, W);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_sin(name, dims):
    total = product(dims)
    shape_literal = format_tuple(dims)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            shape: tuple = {shape_literal}
            total_elements: int = {total}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # output
                ctypes.c_int      # total elements
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            output = torch.empty_like(x)
            cuda_all_inputs = [x, output, params.total_elements]
            torch_all_inputs = [x]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(x)
        """
    )

    cuda_kernel = common_cuda_prelude({"#include <math.h>"}) + dedent(
        """\
        __global__ void sin_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int total
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }
            output[idx] = sinf(input[idx]);
        }

        extern "C" void cuda_kernel(
            const float* input,
            float* output,
            int total
        ) {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            sin_kernel<<<blocks, threads>>>(input, output, total);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


def generate_sub(name, dims):
    total = product(dims)
    shape_literal = format_tuple(dims)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            shape: tuple = {shape_literal}
            total_elements: int = {total}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # a
                ctypes.c_void_p,  # b
                ctypes.c_void_p,  # output
                ctypes.c_int      # total elements
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            a = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            b = torch.randn(params.shape, dtype=torch.float32, device="cuda")
            output = torch.empty_like(a)
            cuda_all_inputs = [a, b, output, params.total_elements]
            torch_all_inputs = [a, b]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    torch_ref = dedent(
        """\
        import torch

        def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a - b
        """
    )

    cuda_kernel = common_cuda_prelude() + dedent(
        """\
        __global__ void sub_kernel(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ output,
            int total
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }
            output[idx] = a[idx] - b[idx];
        }

        extern "C" void cuda_kernel(
            const float* a,
            const float* b,
            float* output,
            int total
        ) {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            sub_kernel<<<blocks, threads>>>(a, b, output, total);
        }
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


TRANSPOSE_PERMUTATIONS = {
    "transpose_1_3_224_224": (0, 2, 3, 1),
    "transpose_24_32_48": (2, 0, 1),
    "transpose_24_36": (1, 0),
    "transpose_2_32_4_64": (0, 2, 1, 3),
    "transpose_33_40_5": (0, 2, 1),
    "transpose_36_16_48": (0, 2, 1),
    "transpose_36_24_16": (1, 0, 2),
    "transpose_42_36_55": (1, 0, 2),
}


def generate_transpose(name, dims):
    perm = TRANSPOSE_PERMUTATIONS.get(name)
    if perm is None:
        raise ValueError(f"{name}: missing permutation metadata")
    if len(perm) != len(dims):
        raise ValueError(f"{name}: permutation length mismatch")

    input_shape = format_tuple(dims)
    output_shape = tuple(dims[i] for i in perm)
    num_dims = len(dims)

    get_data = dedent(
        f"""\
        import torch
        import ctypes
        from dataclasses import dataclass
        from llm_trans.tools.checker import SEED

        @dataclass
        class Params:
            input_shape: tuple = {input_shape}
            output_shape: tuple = {format_tuple(output_shape)}

        def get_cuda_argtypes():
            return [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # output
                {', '.join('ctypes.c_int' for _ in dims)}  # dimensions
            ]

        def get_cuda_torch_inputs(params: Params):
            torch.manual_seed(SEED)
            x = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
            output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
            dim_args = list(params.input_shape)
            cuda_all_inputs = [x, output, *dim_args]
            torch_all_inputs = [x]
            return cuda_all_inputs, torch_all_inputs, [output]

        def cuda_output_tensor_transform(tensor):
            return tensor
        """
    )

    perm_literal = ", ".join(str(p) for p in perm)
    torch_ref = dedent(
        f"""\
        import torch

        def torch_kernel(x: torch.Tensor) -> torch.Tensor:
            return x.permute({perm_literal}).contiguous()
        """
    )

    dims_array = ", ".join(str(len(dims)))
    decl_dims = ", ".join(f"int dim{i}" for i in range(num_dims))
    dims_list = ", ".join(f"dim{i}" for i in range(num_dims))

    compute_indices = []
    compute_indices.append("int indices[{}];".format(num_dims))
    compute_indices.append("int tmp = idx;")
    for i in range(num_dims - 1, -1, -1):
        compute_indices.append(f"indices[{i}] = tmp % dim{i};")
        compute_indices.append(f"tmp /= dim{i};")

    output_indices = []
    for i, p in enumerate(perm):
        output_indices.append(f"int out{i} = indices[{p}];")

    output_dims = [output_shape[i] for i in range(num_dims)]
    output_stride = []
    stride = 1
    for size in reversed(output_dims[1:]):
        stride *= size
        output_stride.append(stride)
    output_stride = list(reversed(output_stride))
    # compute flattened index
    flatten_lines = []
    flatten_lines.append("int out_idx = 0;")
    for i in range(num_dims):
        if i == num_dims - 1:
            flatten_lines.append(f"out_idx += out{i};")
        else:
            stride_expr = product(output_dims[i + 1:])
            flatten_lines.append(f"out_idx += out{i} * {stride_expr};")

    total_expr = " * ".join(f"dim{i}" for i in range(num_dims))

    cuda_kernel = common_cuda_prelude() + dedent(
        f"""\
        __global__ void transpose_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            {decl_dims}
        ) {{
            int total = {total_expr};
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) {{
                return;
            }}

            {" ".join(compute_indices)}
            {" ".join(output_indices)}
            {" ".join(flatten_lines)}

            output[out_idx] = input[idx];
        }}

        extern "C" void cuda_kernel(
            const float* input,
            float* output,
            {decl_dims}
        ) {{
            int total = {total_expr};
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            transpose_kernel<<<blocks, threads>>>(input, output, {dims_list});
        }}
        """
    )

    return CaseFiles(get_data, torch_ref, cuda_kernel)


OP_GENERATORS = {
    "batchnorm": generate_batchnorm,
    "concat": generate_concat,
    "dense": generate_dense,
    "gatemlp": generate_gatemlp,
    "gather": generate_gather,
    "instancenorm": generate_instancenorm,
    "max": lambda name, dims: generate_reduction(name, dims, "max"),
    "mean": lambda name, dims: generate_reduction(name, dims, "mean"),
    "min": lambda name, dims: generate_reduction(name, dims, "min"),
    "scatter": generate_scatter,
    "sin": generate_sin,
    "sub": generate_sub,
    "sum": lambda name, dims: generate_reduction(name, dims, "sum"),
    "transpose": generate_transpose,
}


def parse_case_name(case_name: str):
    parts = case_name.split("_")
    op = parts[0]
    dims = tuple(int(part) for part in parts[1:])
    return op, dims


def collect_missing_cases():
    cuda_cases = {Path(f).stem for f in SRC_CUDA_DIR.iterdir() if f.suffix == ".cu"}
    existing = {
        path.name
        for path in DEST_CASE_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    return sorted(case for case in cuda_cases if case not in existing)


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str):
    path.write_text(content.strip() + "\n", encoding="utf-8")


def generate_case(case_name: str):
    op, dims = parse_case_name(case_name)
    if op == "gqa":
        raise RuntimeError(f"Skipping {case_name}: GQA kernels require manual implementation")

    generator = OP_GENERATORS.get(op)
    if generator is None:
        raise KeyError(f"No generator registered for operation '{op}' ({case_name})")

    files = generator(case_name, dims)

    case_dir = DEST_CASE_DIR / case_name
    ensure_directory(case_dir / "cuda_")
    ensure_directory(case_dir / "torch_")
    ensure_directory(case_dir / "triton_")

    write_file(case_dir / "check_cuda.py", check_cuda_template(case_name))
    write_file(case_dir / "check_triton.py", check_triton_template(case_name))
    write_file(case_dir / "get_data.py", files.get_data)
    write_file(case_dir / "torch_" / "ref.py", files.torch_ref)
    write_file(case_dir / "triton_" / "kernel.py", TRITON_WRAPPER)
    write_file(case_dir / "cuda_" / "kernel.cu", files.cuda_kernel)


def main():
    missing = collect_missing_cases()
    generated = []
    skipped = []

    for case_name in missing:
        try:
            generate_case(case_name)
        except Exception as exc:
            skipped.append((case_name, str(exc)))
        else:
            generated.append(case_name)

    if generated:
        print("Generated cases:")
        for case in generated:
            print(f"  - {case}")

    if skipped:
        print("Skipped cases:")
        for case, reason in skipped:
            print(f"  - {case}: {reason}")


if __name__ == "__main__":
    main()
