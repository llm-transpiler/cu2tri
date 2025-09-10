#include <assert.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void _cuda_kernel_impl(half *A, half *B, float *C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  int blockRow = blockIdx.y * 16;
  int blockCol = blockIdx.x * 16;

  if (blockRow < 1024 && blockCol < 4096) {

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < 128; k += 16) {

      wmma::load_matrix_sync(a_frag, A + blockRow * 128 + k, 128);
      wmma::load_matrix_sync(b_frag, B + k * 4096 + blockCol, 4096);

      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockRow * 4096 + blockCol, c_frag, 4096,
                            wmma::mem_row_major);
  }
}

extern "C" void cuda_kernel(half *A, half *B, float *C,  int m, int k, int n) {
  dim3 blockSize(32);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}