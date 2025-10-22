#include <cuda_runtime.h>
#include <math.h>

__global__ void instancenorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch,
    int channels,
    int spatial
) {
    int total = batch * channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int sub = spatial;
    int channel = (idx / sub) % channels;
    int instance = idx / (channels * sub);

    const float* slice = input + (instance * channels + channel) * spatial;
    float mean = 0.0f;
    for (int i = 0; i < spatial; ++i) {
        mean += slice[i];
    }
    mean /= static_cast<float>(spatial);

    float var = 0.0f;
    for (int i = 0; i < spatial; ++i) {
        float diff = slice[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(spatial);

    float scale = gamma[channel];
    float shift = beta[channel];
    float normalized = (input[idx] - mean) / sqrtf(var + 1e-5f);
    output[idx] = normalized * scale + shift;
}

extern "C" void cuda_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch,
    int channels,
    int spatial
) {
    int total = batch * channels * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    instancenorm_kernel<<<blocks, threads>>>(input, output, gamma, beta, batch, channels, spatial);
}
