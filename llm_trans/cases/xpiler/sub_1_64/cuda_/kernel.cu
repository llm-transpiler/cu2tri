#include <cuda_runtime.h>

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
