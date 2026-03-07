#include <cuda_runtime.h>

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
