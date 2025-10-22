#include <cuda_runtime.h>
#include <math.h>

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
