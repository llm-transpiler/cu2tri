# llm-trans 参数组合指南

## 基本运行

### 单用例快速测试

```bash
# 最简单的测试 (单用例, 1轮, 1次尝试)
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --first-only --max-rounds 1 --max-attempts 1
```

### 指定算子类型

```bash
# 测试特定算子
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add gelu softmax layernorm

# 排除某些算子
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --exclude-case-types mha gqa
```

## 并发控制

### 低并发 (调试)

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --concurrency 1 --temperature 0.3
```

### 高并发 (生产)

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --concurrency 30 --temperature 1.0 --max-attempts 10
```

## 使用 NVGPU 服务器

### 本地 GPU 服务器

```bash
# 先启动 NVGPU 服务器
nvgpu-server --gpus 0 1

# 在另一个终端运行转译器
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --use-nvgpu --nvgpu-server http://localhost:8080
```

### NVGPU 高并发

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --use-nvgpu --nvgpu-server http://localhost:8080 \
  --concurrency 20 --temperature 1.0
```

## 尝试策略

### 首次成功即停止

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --max-attempts 5 --attempt-policy first_success
```

### 全面尝试 (所有尝试)

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --max-attempts 5 --attempt-policy exhaustive
```

## 调试模式

### 跳过性能测试

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --first-only --max-rounds 1 --skip-perf
```

### 详细日志

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --log-level DEBUG
```

### 恢复之前的对话

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler \
  --case-types add --resume-conversation
```

## 完整示例

### 典型的完整测试

```bash
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --case-types add gelu softmax layernorm relu \
  --temperature 0.8 \
  --max-rounds 5 \
  --max-attempts 3 \
  --attempt-policy first_success \
  --concurrency 10 \
  --use-nvgpu \
  --nvgpu-server http://localhost:8080 \
  --skip-perf \
  --max-retries 10 \
  --retry-wait 60
```

### 快速验证

```bash
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --case-types add \
  --first-only \
  --max-rounds 1 \
  --max-attempts 1 \
  --temperature 0.3 \
  --skip-perf \
  --concurrency 1
```

## 参数速查表

| 参数 | 简写 | 说明 | 常用值 |
|------|------|------|--------|
| `--model` | | LLM 模型名称 | gpt_oss_120b_local_5880x4 |
| `--testset` | | 测试集名称 | xpiler, triton_tutorial |
| `--case-types` | | 算子类型 | add, gelu, softmax, etc. |
| `--exclude-case-types` | | 排除算子 | mha, gqa |
| `--temperature` | | 温度参数 | 0.3-1.0 |
| `--max-rounds` | | 最大测试轮数 | 1-5 |
| `--max-attempts` | | 每个用例最大尝试次数 | 1-10 |
| `--attempt-policy` | | 尝试策略 | first_success, exhaustive |
| `--concurrency` | | 并发数 | 1-30 |
| `--use-nvgpu` | | 使用 NVGPU 服务器 | flag |
| `--nvgpu-server` | | NVGPU 服务器地址 | http://localhost:8080 |
| `--first-only` | | 只测试每个算子的第一个用例 | flag |
| `--skip-perf` | | 跳过性能测试 | flag |
| `--resume-conversation` | | 恢复对话 | flag |
| `--max-retries` | | API 最大重试次数 | 10 |
| `--retry-wait` | | 重试等待秒数 | 60 |

## 结果位置

运行结果保存在:
```
llm_trans/runs/cu2tri/<testset>/<model>/<timestamp>/
```

每个用例的结果:
```
<case_name>/attempt_<N>/attempt_result.json
```
