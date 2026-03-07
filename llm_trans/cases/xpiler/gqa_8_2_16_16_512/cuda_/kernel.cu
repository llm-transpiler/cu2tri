#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

#define BATCH 1
#define NUM_Q_HEADS 16
#define NUM_KV_HEADS 2
#define SEQ_Q 16
#define SEQ_KV 512
#define HEAD_DIM 16

// GQA kernel: O = softmax(Q @ K^T / sqrt(d)) @ V
__global__ void _cuda_kernel_impl(const half *__restrict__ Q, const half *__restrict__ K,
                    const half *__restrict__ V, half *__restrict__ O,
                    int num_q_heads, int num_kv_heads) {
  int batch = blockIdx.z / num_q_heads;
  int q_head = blockIdx.z % num_q_heads;
  int kv_head = q_head / (num_q_heads / num_kv_heads);  // GQA grouping
  int tile_row = blockIdx.x;
  int row0 = tile_row * BLOCK_M;

  // Shared memory for attention computation
  __shared__ float attn_tile[BLOCK_M * SEQ_KV];
  __shared__ half softmax_tile[BLOCK_M * SEQ_KV];

  // GQA layout: Q has num_q_heads, K/V share fewer num_kv_heads
  const half *q_ptr = Q + (batch * num_q_heads + q_head) * SEQ_Q * HEAD_DIM + row0 * HEAD_DIM;
  const half *k_ptr = K + (batch * num_kv_heads + kv_head) * SEQ_KV * HEAD_DIM;
  const half *v_ptr = V + (batch * num_kv_heads + kv_head) * SEQ_KV * HEAD_DIM;

  // Step 1: Compute S = Q @ K^T
  for (int col = 0; col < SEQ_KV; col += BLOCK_N) {
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);

    for (int k = 0; k < HEAD_DIM; k += BLOCK_K) {
      wmma::load_matrix_sync(q_frag, q_ptr + k, HEAD_DIM);
      wmma::load_matrix_sync(k_frag, k_ptr + col * HEAD_DIM + k, HEAD_DIM);
      wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    }
    wmma::store_matrix_sync(attn_tile + col, s_frag, SEQ_KV, wmma::mem_row_major);
  }

  __syncthreads();

  // Step 2: Softmax normalization with scaling
  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  int rows_valid = min(BLOCK_M, SEQ_Q - row0);
  for (int i = 0; i < rows_valid; ++i) {
    float max_val = -1e30f;
    for (int j = 0; j < SEQ_KV; ++j)
      max_val = fmaxf(max_val, attn_tile[i * SEQ_KV + j] * scale);

    float sum_val = 0.0f;
    for (int j = 0; j < SEQ_KV; ++j) {
      float e = expf(attn_tile[i * SEQ_KV + j] * scale - max_val);
      softmax_tile[i * SEQ_KV + j] = __float2half(e);
      sum_val += e;
    }
    for (int j = 0; j < SEQ_KV; ++j)
      softmax_tile[i * SEQ_KV + j] = __float2half(__half2float(softmax_tile[i * SEQ_KV + j]) / sum_val);
  }

  __syncthreads();

  // Step 3: Compute O = softmax @ V
  wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> s_frag;
  wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> v_frag;
  wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> o_frag;
  wmma::fill_fragment(o_frag, 0.0f);

  for (int k = 0; k < SEQ_KV; k += BLOCK_K) {
    wmma::load_matrix_sync(s_frag, softmax_tile + k, SEQ_KV);
    wmma::load_matrix_sync(v_frag, v_ptr + k * HEAD_DIM, HEAD_DIM);
    wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
  }

  // Store output (reuse attn_tile as scratch space)
  wmma::store_matrix_sync(attn_tile, o_frag, HEAD_DIM, wmma::mem_row_major);
  
  half *o_ptr = O + (batch * num_q_heads + q_head) * SEQ_Q * HEAD_DIM + row0 * HEAD_DIM;
  for (int i = 0; i < rows_valid; ++i)
    for (int j = 0; j < HEAD_DIM; ++j)
      o_ptr[i * HEAD_DIM + j] = __float2half(attn_tile[i * HEAD_DIM + j]);
}

extern "C" void cuda_kernel(half *Q, half *K, half *V, half *O, int batch,
                           int num_q_heads, int num_kv_heads, int M, int K_dim, int N) {
  dim3 grid((M + BLOCK_M - 1) / BLOCK_M, 1, batch * num_q_heads);
  dim3 block(32);
  _cuda_kernel_impl<<<grid, block>>>(Q, K, V, O, num_q_heads, num_kv_heads);
}

