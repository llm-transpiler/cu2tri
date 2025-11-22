# Performance Testing for Successful Triton Kernels

这个脚本用于对已经成功的triton kernel进行性能测试。

## 功能特性

- 📊 **基于统计数据**: 从 `case_success.json` 文件读取成功的case
- 🔍 **智能发现**: 自动发现所有成功的kernel文件
- 🚀 **批量测试**: 批量提交性能测试到NVGPU server
- 🎯 **精确过滤**: 支持按model、timestamp、case类型、具体case过滤
- 🔧 **独占GPU**: 使用exclusive mode测试，默认GPU 7
- 📋 **结果保存**: 自动保存性能测试结果到JSON文件

## 使用方法

### 1. 基本使用

```bash
cd /data/apps/project/cu2tri/cu2til/llm_trans/perf

# 测试所有成功的kernel (默认GPU 7)
./perf_test.sh all

# 使用不同的GPU
./perf_test.sh all --gpu 3
```

### 2. 过滤测试

```bash
# 测试特定model
./perf_test.sh model gpt_oss_120b

# 测试特定model和timestamp
./perf_test.sh model gpt_oss_120b --timestamp 20251122_054219

# 使用GPU 5测试特定model
./perf_test.sh model gpt_5_mini --gpu 5

# 测试特定case类型（所有add类型的case）
./perf_test.sh all --case-type add

# 测试特定具体case
./perf_test.sh all --case-name add_1_15_64

# 组合过滤：特定model + 特定case类型
./perf_test.sh model gpt_oss_120b --case-type gemm

# 组合过滤：特定model + 具体case + GPU
./perf_test.sh model gpt_5_mini --case-name add_1_15_64 --gpu 5
```

### 3. 干运行模式

```bash
# 查看将要测试的kernel，不实际运行
./perf_test.sh dry-run

# 查看特定model将要测试的kernel
./perf_test.sh dry-run --model gpt_oss_120b

# 查看特定case类型将要测试的kernel
./perf_test.sh dry-run --case-type add

# 查看特定具体case将要测试的kernel
./perf_test.sh dry-run --case-name add_1_15_64

# 组合干运行
./perf_test.sh dry-run --model gpt_oss_120b --case-type gemm
```

### 4. 直接使用Python脚本

```bash
# 测试所有成功的kernel
python3 run_performance_tests.py

# 带参数测试
python3 run_performance_tests.py --model gpt_5_mini --gpu 3

# 测试特定case类型
python3 run_performance_tests.py --case-type add

# 测试特定具体case
python3 run_performance_tests.py --case-name add_1_15_64

# 组合过滤
python3 run_performance_tests.py --model gpt_oss_120b --case-type gemm --gpu 5

# 干运行
python3 run_performance_tests.py --dry-run --case-type add
```

## 命令行参数

### run_performance_tests.py

- `--stats-root PATH`: stats根目录 (默认: `/data/apps/project/cu2tri/cu2til/llm_trans/stats`)
- `--runs-root PATH`: runs根目录 (默认: `/data/apps/project/cu2tri/cu2til/llm_trans/runs`)
- `--cu2tri NAME`: cu2tri目录名 (默认: `cu2tri`)
- `--xpiler NAME`: xpiler目录名 (默认: `xpiler`)
- `--model MODEL`: 只测试指定model
- `--timestamp TS`: 只测试指定timestamp
- `--case-type TYPE`: 只测试指定case类型 (e.g., 'add' for all add cases)
- `--case-name CASE`: 只测试指定具体case (e.g., 'add_1_15_64')
- `--gpu ID`: 独占GPU ID (默认: 7)
- `--output-dir PATH`: 结果输出目录
- `--dry-run`: 干运行模式

## 工作流程

1. **扫描统计文件**: 从 `stats/cu2tri/xpiler/` 扫描所有 `case_success.json` 文件
2. **提取成功案例**: 从 `case_success.json` 中提取 `stat_final_success=true` 的case
3. **应用过滤器**: 根据--case-type、--case-name等参数过滤要测试的case
4. **定位kernel文件**: 根据model, timestamp, case_name, attempt_number定位到 `runs/cu2tri/xpiler/{model}/{timestamp}/dev_/{case_name}/attempt_{num}/triton_/kernel.py`
5. **提交性能测试**: 提交到NVGPU server进行性能测试，使用exclusive mode
6. **等待完成**: 监控测试进度，等待完成或超时
7. **保存结果**: 保存测试结果到JSON文件

## 输出结果

性能测试结果保存在 `/data/apps/project/cu2tri/cu2til/llm_trans/perf/results/` 目录下，文件名格式为：
`performance_results_gpu{gpu_id}_{timestamp}.json`

结果格式：
```json
[
  {
    "model": "gpt_oss_120b",
    "timestamp": "20251122_054219",
    "case_type": "add",
    "case_name": "add_1_15_64",
    "attempt_number": 1,
    "successful_round": 3,
    "kernel_path": "/path/to/kernel.py",
    "task_id": "task_123456",
    "gpu_id": 7,
    "status": "completed",
    "error": null,
    "performance_metrics": {
      "execution_time_ms": 12.34,
      "memory_usage_mb": 1024
    }
  }
]
```

## 前置条件

1. **NVGPU Server**: 确保 `/data/apps/project/cu2tri/server/nvgpu` 服务器正在运行
2. **统计数据**: 确保已经运行过statkit生成了 `case_success.json` 文件
3. **Kernel文件**: 确保 `runs/` 目录下存在对应的成功kernel文件

## 错误处理

- **Kernel未找到**: 如果找不到kernel文件，会记录为 `kernel_not_found`
- **提交失败**: 如果NVGPU server提交失败，会记录为 `submission_failed`
- **测试失败**: 如果性能测试失败，会记录错误信息
- **超时处理**: 性能测试超时时间为20分钟

## 示例输出

```
Performance Testing for Successful Triton Kernels
===============================================
GPU ID: 7 (exclusive mode)
Stats root: /data/apps/project/cu2tri/cu2til/llm_trans/stats
Runs root: /data/apps/project/cu2tri/cu2til/llm_trans/runs
✅ NVGPU server is running

📊 Found 2 case_success.json files

📖 Reading: /path/to/stats/cu2tri/xpiler/gpt_oss_120b/20251122_054219/case_success.json
  ✅ Found 45 successful cases

🎯 Total successful kernels to test: 45

🚀 Starting performance testing...
Using GPU 7 in exclusive mode

[1/45] Testing add_1_15_64 from gpt_oss_120b/20251122_054219...
  📂 Found kernel: /path/to/kernel.py
  📤 Task submitted: task_123456
  ⏳ Waiting for performance test completion...
  ✅ Performance test completed successfully
     Execution time: 12.34 ms
     Memory usage: 1024 MB

...

🏁 Performance testing completed!
Successfully tested: 42
Failed to test: 3
Total kernels: 45
Performance results saved to: /path/to/perf/results/performance_results_gpu7_20251123_123456.json
```