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
ls llm_trans/cases/xpiler/ | wc -l  # 应显示 298

# 如果缺少用例，从 git 恢复
git checkout main -- cu2til/cases/
cp -r cu2til/cases/xpiler/* llm_trans/cases/xpiler/
```

### 找不到 tools 脚本

```bash
# 从 git 恢复
git checkout main -- cu2til/tools/
```

## 目录结构

```
/cu2tri/
├── server/nvgpu/      # NVGPU 服务器
├── server/common/     # 服务器共享代码
├── llm_trans/         # LLM 转译器
├── llm/               # LLM 提供商接口
├── cu2til/cases/      # 原始测试用例
└── cu2til/tools/      # 工具脚本
```
