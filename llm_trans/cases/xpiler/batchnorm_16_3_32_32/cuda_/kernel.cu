#include <cuda_runtime.h>
#include <math.h>

__global__ void batchnorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size,
    int num_channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_channels * spatial_size;
    if (idx >= total) {
        return;
    }

    int channel = (idx / spatial_size) % num_channels;
    float norm = (input[idx] - mean[channel]) / sqrtf(var[channel] + 1e-5f);
    output[idx] = norm * gamma[channel] + beta[channel];
}

extern "C" void cuda_kernel(
    const float* input,
    float* output,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    int batch_size,
    int num_channels,
    int spatial_size
) {
    int total = batch_size * num_channels * spatial_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batchnorm_kernel<<<blocks, threads>>>(input, output, mean, var, gamma, beta, batch_size, num_channels, spatial_size);
}
