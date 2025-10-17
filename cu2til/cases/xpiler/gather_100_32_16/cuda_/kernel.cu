#include <cuda_runtime.h>
#include <stdint.h>

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
