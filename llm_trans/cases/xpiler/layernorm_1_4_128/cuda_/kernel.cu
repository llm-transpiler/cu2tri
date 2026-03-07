#include <assert.h>

__global__ void _cuda_kernel_impl(float *A, float *gamma, float *beta, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 4) {
    float mean = 0.0;
    float variance = 0.0;
    float diff[128];
    // Calculate mean
    for (int i_mean = 0; i_mean < 128; i_mean++) {
      mean += A[idx * 128 + i_mean];
    }
    mean /= 128;
    // Calculate variance
    for (int i_diff = 0; i_diff < 128; i_diff++) {
      diff[i_diff] = A[idx * 128 + i_diff] - mean;
    }

    for (int i_pow = 0; i_pow < 128; i_pow++) {
      diff[i_pow] = diff[i_pow] * diff[i_pow];
    }
    for (int i_var = 0; i_var < 128; i_var++) {
      variance += diff[i_var];
    }
    variance = sqrt(variance / 128);

    // Normalize A
    for (int i_norm = 0; i_norm < 128; i_norm++) {
      diff[i_norm] = (A[idx * 128 + i_norm] - mean);
    }

    for (int i_mul = 0; i_mul < 128; i_mul++) {
      diff[i_mul] = diff[i_mul] * gamma[i_mul];
    }

    for (int i_div = 0; i_div < 128; i_div++) {
      diff[i_div] = diff[i_div] / (variance + 1e-5f);
    }

    for (int i_bet = 0; i_bet < 128; i_bet++) {
      B[idx * 128 + i_bet] = diff[i_bet] + beta[i_bet];
    }
  }
}

extern "C" void cuda_kernel(float *A, float *gamma, float *beta, float *B,
                                 int batch_size, int seq_length, int d_model) {

  // Define grid and block dimensions
  int block_size = 4;
  int num_blocks = (batch_size * seq_length + block_size - 1) / block_size;
  
  _cuda_kernel_impl<<<num_blocks, block_size>>>(A, gamma, beta, B);
}