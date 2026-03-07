# cu2tri 常见问题解决

## 安装问题

### ModuleNotFoundError

```bash
pip install -e /cu2tri
export PROJECT_ROOT=/cu2tri
```

### nvgpu-server 找不到模块

```bash
pip install -e /cu2tri
# 或
python -m server.nvgpu.main --gpu-config ...
```

## 配置问题

### 模型配置文件路径

编辑 `llm_trans/config/model_clients.yaml` 配置模型端点和 API key。

### GPU 配置文件路径

编辑 `server/nvgpu/configs/gpu_resources/*.yml` 配置可用 GPU。

## 运行问题

### 找不到测试用例

```bash
# 检查用例位置
ls llm_trans/cases/xpiler/ | wc -l  # 应显示 280+ 个用例

# 如果缺少用例，从 git 恢复
git checkout main -- llm_trans/cases/
```

### 找不到 tools 脚本

```bash
# 从 git 恢复
git checkout main -- llm_trans/tools/
```

### NVGPU 路径问题

**重要**: 始终从项目根目录 `/cu2tri` 启动 nvgpu-server

```bash
# 正确 ✅
cd /cu2tri
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 错误 ❌
cd /cu2tri/server/nvgpu
python main.py --gpu-config configs/gpu_resources/P250_A6000.yml
```

### 任务一直 pending (NVGPU)

1. 检查 GPU 是否被注册:
   ```bash
   tail -20 server/nvgpu/logs/nvgpu_server_*.log | grep "Registered GPU"
   ```

2. 检查 GPU 配置文件中 `enabled: true`

3. 确认配置文件路径正确:
   ```bash
   nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
   ```

### ModuleNotFoundError: No module named 'config'

这是已修复的bug，确保更新到最新代码:
```bash
git pull origin dev-gpu
pip install -e /cu2tri
```

## 目录结构

```
/cu2tri/
├── server/nvgpu/      # NVGPU 服务器
├── server/common/     # 服务器共享代码
├── llm_trans/         # LLM 转译器
│   ├── cases/         # 测试用例
│   ├── tools/         # 工具脚本
│   └── prompts/       # LLM 提示词
├── llm/               # LLM 提供商接口
└── .claude/skills/    # Claude Code 技能文档
```

## Git 相关

### 切换分支

```bash
git checkout main      # 主分支
git checkout dev-gpu   # GPU 开发分支
```

### 恢复特定文件

```bash
git checkout main -- llm_trans/cases/xpiler/add_100_2_10_1024/
```

### 查看文件历史

```bash
git log --oneline -- llm_trans/prompts/cuda2triton.py
```
