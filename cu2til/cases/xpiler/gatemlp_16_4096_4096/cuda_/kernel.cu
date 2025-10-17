#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

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
