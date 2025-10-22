#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
