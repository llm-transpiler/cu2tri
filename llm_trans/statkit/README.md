# CU2TRI Statistics Toolkit (statkit)

一个用于从 xpiler JSONL 文件中提取统计信息的完整工具集。

## 功能特性

- 📊 **双格式输出**：生成完整的详细信息和极简的 case 成功统计
- 🔄 **智能过滤**：支持最大 attempt 数和 round 数的过滤
- ⚡ **批量处理**：自动发现并处理所有时间戳目录
- 💾 **增量更新**：默认跳过已存在的文件，支持强制覆盖
- 🗂️ **目录结构保持**：输出目录结构保持与 runs 目录一致
- 📈 **自然排序**：case 名称和 attempt 号码按照自然顺序排序

## 文件说明

### 核心脚本

1. **`extract_stats.py`** - 主要的统计提取脚本
2. **`batch_process.py`** - 批量处理所有时间戳的脚本
3. **`statkit.py`** - 交互式入口脚本
4. **`statkit.sh`** - 命令行快捷脚本

## 输出文件

### 完整信息文件：`full_info.json` / `full_info_at{N}_r{M}.json`

包含完整的元数据、摘要信息和详细的 case 数据：

```json
{
  "metadata": {...},
  "summary": {
    "total_cases": 288,
    "case_types": ["add", "gemm", ...],
    "total_attempts": 4320
  },
  "cases": {
    "add_1_15_64": {
      "case_type": "add",
      "total_attempts": 15,
      "successful_attempts": 12,
      "attempts": [...]
    }
  }
}
```

### 配置文件：`config.json`

包含原始数据的完整信息和统计配置：

```json
{
  "model_name": "openai/gpt-5-mini",
  "timestamp": "20251024_065400",
  "case_base_info": {
    "total_cases": 286,
    "case_types": ["add", "avgpool", "batchnorm", ...]
  }
}
```

#### 字段说明：
- `total_cases`: 该数据集的总case数量
- `case_types`: 所有的case类型列表

### 极简统计文件：`case_success.json` / `case_success_at{N}_r{M}.json`

按 case_type 组织的极简统计信息：

```json
{
  "add": {
    "add_1_15_64": {
      "stat_final_success": true,
      "stat_total_attempts": 2,
      "attempts": [
        {
          "attempt_number": 1,
          "success": true,
          "round_final": 3
        },
        {
          "attempt_number": 2,
          "success": false,
          "round_final": 4
        }
      ]
    }
  },
  "gemm": {
    "gemm_32_32_128": {
      "stat_final_success": false,
      "stat_total_attempts": 1,
      "attempts": [
        {
          "attempt_number": 1,
          "success": false,
          "round_final": 3
        }
      ]
    }
  }
}
```

#### Field Description:
- `stat_final_success`: Whether this case eventually succeeded
- `stat_total_attempts`: Number of attempts counted (subject to max_attempts limit)
- `attempts[]`: Details for each attempt
  - `attempt_number`: Attempt number
  - `success`: Whether this attempt succeeded
  - `round_final`: The successful round number (integer) for successful attempts, or the last attempted round number for failed attempts

## 使用方法

### 1. 交互式模式

```bash
cd /data/apps/project/cu2tri/cu2til/llm_trans/statkit
python3 statkit.py
```

### 2. 命令行快捷方式

```bash
# 处理所有文件
./statkit.sh all

# 处理所有文件，限制最大3个attempts
./statkit.sh all --max-attempts 3

# 干运行查看将要处理的文件
./statkit.sh dry-run

# 处理特定模型
./statkit.sh all --model gpt_5_mini

# 处理特定时间戳
./statkit.sh all --model gpt_5_mini --timestamp 20251024_065400

# 处理单个文件
./statkit.sh single /path/to/file.jsonl
```

### 3. 直接使用Python脚本

```bash
# 处理单个文件
python3 extract_stats.py /path/to/xpiler.jsonl

# 处理目录中的所有文件
python3 extract_stats.py /path/to/directory/

# 带过滤参数
python3 extract_stats.py /path/to/file.jsonl --max-attempts 5 --max-rounds 2

# 强制覆盖已存在的文件
python3 extract_stats.py /path/to/file.jsonl --overwrite

# 只生成极简统计
python3 extract_stats.py /path/to/file.jsonl --minimal-only
```

### 4. 批量处理

```bash
# 批量处理所有时间戳
python3 batch_process.py

# 带过滤参数的批量处理
python3 batch_process.py --max-attempts 3 --model gpt_5_mini

# 干运行模式
python3 batch_process.py --dry-run
```

## 命令行参数

### extract_stats.py

- `jsonl_path`: JSONL文件或目录路径（默认：runs/cu2tri）
- `--max-attempts N`: 最大attempt数量
- `--max-rounds N`: 最大round数量
- `--output-dir PATH`: 输出目录
- `--overwrite`: 覆盖已存在的文件
- `--minimal-only`: 只生成极简统计文件

### batch_process.py

- `--runs-root PATH`: runs根目录
- `--stats-root PATH`: stats根目录
- `--model MODEL`: 只处理指定模型
- `--timestamp TS`: 只处理指定时间戳
- `--dry-run`: 干运行模式
- 其他参数同extract_stats.py

## 输出目录结构

```
stats/
└── cu2tri/
    └── xpiler/
        └── {model_name}/
            └── {timestamp}/
                ├── full_info.json
                ├── full_info_at3.json
                ├── case_success.json
                └── case_success_at3_r2.json
```

## 发现的模型和文件

目前系统已发现以下模型的时间戳数据：
- minimax_m2_free (1个时间戳)
- minimax_m2 (3个时间戳)
- gpt_5_mini (2个时间戳)
- qwen3_vl_235b_a22b_thinking (2个时间戳)
- gpt_oss_20b (12个时间戳)
- deepseek_v3_2_exp (1个时间戳)
- gemini_2_5_flash (1个时间戳)
- test_model (1个时间戳)
- qwen3_235b_a22b_thinking_2507 (5个时间戳)
- gpt_oss_120b (11个时间戳)

总共 **38个JSONL文件** 可供处理。