#include <cuda_runtime.h>

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
