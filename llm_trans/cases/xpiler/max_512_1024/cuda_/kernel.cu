#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

    float val = -FLT_MAX;
    for (int r = 0; r < rows; ++r) {
        val = fmaxf(val, input[r * inner + idx]);
    }
    output[idx] = val;
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
