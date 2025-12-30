__global__ void kernel(float *Q, float *K, float *V, float *output) {
  const int batch_size = 1;
  const int seq_len = 2048;
  const int num_heads = 6;
  const int head_dim = 256;
  /**
   * Q[0:batch_size][0:seq_len][0:num_heads][0:head_dim]
   * K[0:batch_size][0:seq_len][0:num_heads][0:head_dim]
   * V[0:batch_size][0:seq_len][0:num_heads][0:head_dim]
   * output[0:batch_size][0:seq_len][0:num_heads][0:head_dim]
   * score[0:num_heads][0:num_heads]
   */
  __shared__ float score[36]; // 使用共享内存存储 score, 大小为 heads * heads
  float scaling_factor = 1.0f / sqrtf((float)head_dim);
  int i = blockIdx.x;  // batch index
  int j = blockIdx.y;  // query index within sequence length
  int m = threadIdx.x; // head index

  for (int n = 0; n < num_heads; n++) {
    /**
     * tmp[0:head_dim] = Q[i][j][m][0:head_dim] * K[i][j][n][0:head_dim]
     * score[m][n] = sum(tmp[0:head_dim])
     */
    score[m * num_heads + n] = 0.0;
    for (int p = 0; p < head_dim; p++) {
      /**
       * tmp[p] = Q[i][j][m][p] * K[i][j][n][p]
       * score[m][n] += tmp[p]
       */
      score[m * num_heads + n] += Q[i * seq_len * num_heads * head_dim + j * num_heads * head_dim + m * head_dim + p] *
                          K[i * seq_len * num_heads * head_dim + j * num_heads * head_dim + n * head_dim + p];
    }
  }

  // score
  /**
   * score[m][0:num_heads] = score[m][0:num_heads] * scaling_factor
   */
  for (int n_sc = 0; n_sc < num_heads; n_sc++) {
    /**
     * score[m][n_sc] = score[m][n_sc] * scaling_factor
     */
    score[m * num_heads + n_sc] = score[m * num_heads + n_sc] * scaling_factor;
  }

  float sum = 0;

  for (int i_ex = 0; i_ex < num_heads; ++i_ex) {
    /**
     * score[m][i_ex] = expf(score[m][i_ex])
     */
    score[m * num_heads + i_ex] = expf(score[m * num_heads + i_ex]);
  }
  for (int i_sf = 0; i_sf < num_heads; ++i_sf) {
    /**
     * sum += score[m][i_sf]
     */
    sum += score[m * num_heads + i_sf];
  }
  for (int k_sf = 0; k_sf < num_heads; ++k_sf) {
    /**
     * score[m][k_sf] = score[m][k_sf] / sum
     */
    score[m * num_heads + k_sf] = score[m * num_heads + k_sf] / sum;
  }

  // The final Matmul
  for (int n_fl = 0; n_fl < head_dim; ++n_fl) {
    /**
     * output[i][j][m][n_fl] = 0.0
     */
    output[i * seq_len * num_heads * head_dim + j * num_heads * head_dim + m * head_dim + n_fl] = 0.0;
    for (int k_fl = 0; k_fl < num_heads; ++k_fl) {
      /**
       * output[i][j][m][n_fl] += score[m][k_fl] * V[i][j][k_fl][n_fl]
       */
      output[i * seq_len * num_heads * head_dim + j * num_heads * head_dim + m * head_dim + n_fl] +=
          score[m * num_heads + k_fl] *
          V[i * seq_len * num_heads * head_dim + j * num_heads * head_dim + k_fl * head_dim + n_fl];
    }
  }
}

extern "C" void cuda_kernel(float *d_queries, float *d_keys, float *d_values, float *d_output) {
  const int batch_size = 1;
  const int seq_len = 2048;
  const int num_heads = 6;
  const int head_dim = 256;
  dim3 grid(batch_size, seq_len);
  dim3 block(num_heads);

  kernel<<<grid, block>>>(d_queries, d_keys, d_values, d_output);
}
