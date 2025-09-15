XPILER_ALL_CASES = {
    "add": [
        "add_1_15_64",
        "add_3_3_256", 
        "add_4_4_4_64",
        "add_18_128",
        "add_21_192",
        "add_64",
        "add_100_2_10_1024",
        "add_320"
    ],
    
    "avgpool": [
        "avgpool_1_5_5_64_5_5_1_1",
        "avgpool_4_8_8_64_5_5_3_3",
        "avgpool_4_35_35_192_5_5_2_2",
        "avgpool_4_56_56_128_5_5_2_2",
        "avgpool_5_32_32_64_5_5_3_3",
        "avgpool_5_112_112_64_5_5_3_3",
        "avgpool_16_64_64_64_5_5_2_2",
        "avgpool_16_112_112_64_5_5_3_3"
    ],
    
    "bmm": [
        "bmm_1_128_128_128",
        "bmm_1_128_256_512",
        "bmm_1_256_256_256", 
        "bmm_1_512_512_512",
        "bmm_4_128_128_128",
        "bmm_4_128_256_512",
        "bmm_4_256_256_256",
        "bmm_4_512_512_512"
    ],
    
    "conv1d": [
        "conv1d_5_7",
        "conv1d_25_27",
        "conv1d_44_46",
        "conv1d_126_128",
        "conv1d_190_192",
        "conv1d_222_224",
        "conv1d_256_258", 
        "conv1d_312_314"
    ],
    
    "conv2d": [
        "conv2d_16_8_8_64_64_2_2_64_2_0",
        "conv2d_16_8_8_64_64_2_2_64_3_0",
        "conv2d_16_8_8_128_64_2_2_128_2_0",
        "conv2d_16_8_8_128_64_2_2_128_3_0",
        "conv2d_32_8_8_64_64_2_2_64_2_0",
        "conv2d_32_8_8_64_64_2_2_64_3_0",
        "conv2d_32_8_8_128_64_2_2_128_2_0",
        "conv2d_32_8_8_128_64_2_2_128_3_0"
    ],
    
    "conv2dnchw": [
        "conv2dnchw_16_64_8_8_128_64_2_2_2_0",
        "conv2dnchw_16_64_8_8_128_64_2_2_3_0",
        "conv2dnchw_16_128_8_8_64_128_2_2_2_0",
        "conv2dnchw_16_128_8_8_64_128_2_2_3_0",
        "conv2dnchw_32_64_8_8_128_64_2_2_2_0",
        "conv2dnchw_32_64_8_8_128_64_2_2_3_0",
        "conv2dnchw_32_128_8_8_64_128_2_2_2_0",
        "conv2dnchw_32_128_8_8_64_128_2_2_3_0"
    ],
    
    "deformable": [
        "deformable_1_8_256_100_4_4",
        "deformable_1_8_256_200_4_4", 
        "deformable_1_8_512_100_4_4",
        "deformable_1_8_512_200_4_4",
        "deformable_4_8_256_100_4_4",
        "deformable_4_8_256_200_4_4",
        "deformable_4_8_512_100_4_4",
        "deformable_4_8_512_200_4_4"
    ],
    
    "depthwiseconv": [
        "depthwiseconv_6_3_3",
        "depthwiseconv_6_3_128",
        "depthwiseconv_128_3_3",
        "depthwiseconv_128_3_128",
        "depthwiseconv_192_3_3", 
        "depthwiseconv_192_3_128",
        "depthwiseconv_256_3_3",
        "depthwiseconv_256_3_128"
    ],
    
    "gelu": [
        "gelu_3_4_5",
        "gelu_5_7_3_32",
        "gelu_5_12_23_128",
        "gelu_5_128",
        "gelu_7_1_6_7",
        "gelu_8_10_64",
        "gelu_12_3_128",
        "gelu_45_25"
    ],
    
    "gemm": [
        "gemm_32_32_128",
        "gemm_32_32_1024",
        "gemm_32_128_128", 
        "gemm_32_128_1024",
        "gemm_1024_16_128",
        "gemm_1024_16_1024",
        "gemm_1024_128_128",
        "gemm_1024_128_4096"
    ],
    
    "gemv": [
        "gemv_3_16",
        "gemv_3_512",
        "gemv_32_64",
        "gemv_32_512",
        "gemv_112_128",
        "gemv_112_224",
        "gemv_125_128", 
        "gemv_125_320"
    ],
    
    "layernorm": [
        "layernorm_1_4_32",
        "layernorm_1_4_128",
        "layernorm_1_8_32",
        "layernorm_1_8_128",
        "layernorm_2_4_32",
        "layernorm_2_4_128",
        "layernorm_2_8_32",
        "layernorm_2_8_128"
    ],
    
    "maxpool": [
        "maxpool_1_5_5_64_5_5_1_1",
        "maxpool_4_8_8_64_5_5_3_3",
        "maxpool_4_35_35_192_5_5_3_3",
        "maxpool_4_56_56_128_5_5_2_2",
        "maxpool_5_32_32_64_5_5_3_3",
        "maxpool_5_112_112_64_5_5_2_2",
        "maxpool_16_64_64_64_5_5_2_2",
        "maxpool_16_112_112_64_5_5_3_3"
    ],
    
    "mha": [
        "mha_1_2048_6_256",
        "mha_1_2048_12_256",
        "mha_1_4096_6_256",
        "mha_1_4096_12_256",
        "mha_1_4096_12_512",
        "mha_64_2048_12_256",
        "mha_64_2048_12_512",
        "mha_64_4096_12_256"
    ],
    
    "minpool": [
        "minpool_1_5_5_64_5_5_1_1",
        "minpool_4_8_8_64_5_5_3_3",
        "minpool_4_35_35_192_5_5_3_3",
        "minpool_4_56_56_128_5_5_3_3",
        "minpool_5_32_32_64_5_5_3_3",
        "minpool_5_112_112_64_5_5_3_3",
        "minpool_16_64_64_64_5_5_3_3",
        "minpool_16_112_112_64_5_5_3_3"
    ],
    
    "relu": [
        "relu_3_4_5",
        "relu_5_7_3_32",
        "relu_5_12_23_128",
        "relu_5_128",
        "relu_7_1_6_7",
        "relu_8_10_64", 
        "relu_12_3_128",
        "relu_45_25"
    ],
    
    "rmsnorm": [
        "rmsnorm_2048_2048",
        "rmsnorm_2048_4096",
        "rmsnorm_2048_8192",
        "rmsnorm_4096_2048",
        "rmsnorm_4096_4096",
        "rmsnorm_4096_8192",
        "rmsnorm_8192_4096",
        "rmsnorm_8192_8192"
    ],
    
    "sigmoid": [
        "sigmoid_3_4_5",
        "sigmoid_5_7_3_32",
        "sigmoid_5_12_23_128",
        "sigmoid_5_128",
        "sigmoid_7_1_6_7", 
        "sigmoid_8_10_64",
        "sigmoid_12_3_128",
        "sigmoid_45_25"
    ],
    
    "sign": [
        "sign_3_4_5",
        "sign_5_7_3_32",
        "sign_5_12_23_128",
        "sign_5_128",
        "sign_7_1_6_7",
        "sign_8_10_64",
        "sign_12_3_128",
        "sign_45_25"
    ],
    
    "softmax": [
        "softmax_3_4_5",
        "softmax_5_7_3_32",
        "softmax_5_12_23_128",
        "softmax_5_128",
        "softmax_7_1_6_7",
        "softmax_8_10_64",
        "softmax_12_3_128",
        "softmax_45_25"
    ],
    
    "sumpool": [
        "sumpool_1_5_5_64_3_3_2_2",
        "sumpool_4_8_8_64_3_3_2_2",
        "sumpool_4_35_35_192_5_5_2_2",
        "sumpool_4_56_56_128_5_5_3_3",
        "sumpool_5_32_32_64_5_5_3_3",
        "sumpool_5_112_112_64_3_3_2_2",
        "sumpool_16_64_64_64_5_5_1_1",
        "sumpool_16_112_112_64_5_5_3_3"
    ]
}
LEETCUDA_DYNAMIC_ALL_CASES = {
    "add": [
        "add_f16x8_pack",
        "add_f32x4"
    ],
    "dot_prod": [
        "dot_prod_f16x8_pack_f32",
        "dot_prod_f32x4_f32"
    ],
    "elu": [
        "elu_f16x8_pack",
        "elu_f32x4"
    ],
    "embedding": [
        "embedding_f16x8_pack",
        "embedding_f32x4",
        "embedding_f32x4_pack"
    ],
    "flash_attn": [
        "flash_attn_mma_stages_split_q_shared_kv"
    ],
    "gelu": [
        "gelu_f16x8_pack",
        "gelu_f32x4"
    ],
    "hgemm": [
        "hgemm"
    ],
    "hgemv": [
        "hgemv_k128_f16x4",
        "hgemv_k16_f16",
        "hgemv_k32_f16"
    ],
    "hardshrink": [
        "hardshrink_f16x8_pack",
        "hardshrink_f32x4"
    ],
    "hardswish": [
        "hardswish_f16x8_pack",
        "hardswish_f32x4"
    ],
    "layer_norm": [
        "layer_norm_f16x8_pack_f32",
        "layer_norm_f32x4"
    ],
    "relu": [
        "relu_f16x8pack",
        "relu_f32x4"
    ],
    "rmsnorm": [
        "rmsnorm_f16x8packf16",
        "rmsnorm_f32x4"
    ],
    "safe_softmax": [
        "safe_softmax_f16x8_pack_f32_per_token",
        "safe_softmax_f32x4_online_pack_per_token"
    ],
    "sgemm": [
        "sgemm",
        "sgemm_wmma_tf32_stage_dsmem"
    ],
    "sgemv": [
        "sgemv_k128f32x4"
    ],
    "sigmoid": [
        "sigmoid_f16x8pack",
        "sigmoid_f32x4"
    ],
    "sum": [
        "sum_f16x8_pack_f16",
        "sum_f32x4_f32"
    ],
    "swish": [
        "swish_f16x8_pack",
        "swish_f32x4"
    ]
}