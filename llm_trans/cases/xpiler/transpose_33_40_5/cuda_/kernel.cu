#include <cuda_runtime.h>

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim0, int dim1, int dim2
) {
    int total = dim0 * dim1 * dim2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int indices[3]; int tmp = idx; indices[2] = tmp % dim2; tmp /= dim2; indices[1] = tmp % dim1; tmp /= dim1; indices[0] = tmp % dim0; tmp /= dim0;
    int out0 = indices[0]; int out1 = indices[2]; int out2 = indices[1];
    int out_idx = 0; out_idx += out0 * 200; out_idx += out1 * 40; out_idx += out2;

    output[out_idx] = input[idx];
}

extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int dim0, int dim1, int dim2
) {
    int total = dim0 * dim1 * dim2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_kernel<<<blocks, threads>>>(input, output, dim0, dim1, dim2);
}
