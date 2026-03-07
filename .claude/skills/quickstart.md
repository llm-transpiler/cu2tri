# cu2tri 快速启动指南

## 环境变量

```bash
export PROJECT_ROOT=/cu2tri
```

## 安装

```bash
cd /cu2tri
pip install -e .
```

## 运行 LLM 转译器

```bash
# 基本运行
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler

# 快速测试 (单用例)
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler --case-types add --first-only --max-attempts 1 --max-rounds 1

# 高并发运行
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler --concurrency 30 --temperature 1.0
```

## 运行 NVGPU 服务器

```bash
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

## 测试用例类型

288 个用例，37 种操作类型：add, avgpool, batchnorm, bmm, concat, conv1d, conv2d, deformable, dense, gelu, gemm, gemv, gqa, layernorm, mha, maxpool, relu, softmax, transpose 等
