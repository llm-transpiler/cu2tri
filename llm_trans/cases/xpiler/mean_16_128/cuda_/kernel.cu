#include <cuda_runtime.h>

__global__ void reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner) {
        return;
    }

    float val = 0.0f;
    for (int r = 0; r < rows; ++r) {
        val += input[r * inner + idx];
    }
    output[idx] = val / static_cast<float>(rows);
}

extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int rows,
    int inner
) {
    int threads = 256;
    int blocks = (inner + threads - 1) / threads;
    reduction_kernel<<<blocks, threads>>>(input, output, rows, inner);
}
