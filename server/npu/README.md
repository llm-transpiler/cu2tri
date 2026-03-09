# NPU Server

面向 Ascend/CANN 的 NPU 任务调度服务器，支持多 NPU 设备管理、任务队列和 REST API。

## 快速开始

```bash
cd /workspace
python -m server.npu.main --npu-config server/npu/configs/npu_resources/ascend_910b_sample.yml
```

自动探测并按主机名生成配置文件（如 `aicc-06_npu_config.yml`）：

```bash
python -m server.npu.tools.npu_detect --print-command
```

或使用安装后的命令：

```bash
npu-server --npu-config server/npu/configs/npu_resources/ascend_910b_sample.yml
```

## 常用参数

- `--host`：监听地址（默认 `0.0.0.0`）
- `--port`：监听端口（默认 `8080`）
- `--npu-config`：NPU YAML 配置路径
- `--npus`：直接指定启动注册的 NPU ID
- `--npu-mode`：默认模式（`exclusive`/`shared`）

## NPU 可见设备变量

任务启动时默认只注入 `ASCEND_RT_VISIBLE_DEVICES`（任务自定义 `env` 可覆盖）。

## API 端点（核心）

- `GET /health`
- `GET /npus`
- `GET /npus/{npu_id}`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/cancel`
- `GET /stats`
