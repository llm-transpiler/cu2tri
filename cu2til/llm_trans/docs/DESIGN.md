# cu2til/llm_trans 设计说明（架构、端点轮换与日志规范）

本模块将原有单文件脚本重构为职责清晰的包结构，统一数据模型与日志格式，并引入端点轮换机制以增强鲁棒性。

## 架构概览
- 配置与运行时
  - `config/args.py`：CLI 参数声明
  - `config/settings.py`：`Settings` 数据类（推导工作目录、时间戳、测试集等）
  - `core/runtime.py`：`RuntimeContext`（聚合 settings、模型客户端、日志锁、可用用例）
- 客户端
  - `clients/model.py`：`ModelClients`/`create_model_clients`，支持单端点与端点池，统一 OpenAI 兼容参数构建
  - `clients/nvgpu.py`：惰性导入 NVGPU 客户端与可用性标志
- 数据与 I/O
  - `data/models.py`：`RetryRecord`、`RoundRecord`、`TestRoundRecord`、`AttemptTimingStats`、`AttemptResult`、`CaseResult`、`BatchSummary`
  - `io/logging.py`：文件+控制台日志
  - `io/jsonl.py`：异步安全 JSONL 写入
- 业务服务
  - `services/cases.py`：测试集解析与筛选
  - `services/conversation.py`：抽取 thinking、持久化会话 JSONL、保存会话快照
  - `services/history.py`：`AttemptHistoryManager`，支持恢复/续跑
  - `services/retry.py`：网络/限流判定、重试上下文序列化、重试事件写盘
  - `services/testing.py`：本地/NVGPU 测试轮次与反馈修复
  - `services/attempts.py`：单 attempt 主流程与多 attempt 调度策略
  - `services/runner.py`：批量 orchestrator、汇总与总览

## 模型注册表与端点轮换
通过 `config/model_clients.yaml` 将“模型服务商 → 提供方 → 模型”层次配置化：

- `vendors.<vendor>.providers.<provider>`：定义单个端点（`base_url`、`api_key`、`calling`），字段可直接写值，也支持 `$ENV` 或 `{env: ENV, default: ...}` 形式引用环境变量；示例中同时预置了 `local`、`local-127-800{1,2,3}`、`local-172-800{1,2,3}` 等固定地址，便于快速切换本地/机房推理节点。
- 如需批量扩充分发的模型映射，可参考 `/data/apps/project/cu2tri/tools/price/openrouter_api.csv` 提供的 OpenRouter 型号列表。
- `vendors.<vendor>.models.<model>.providers`：声明该模型可用提供方顺序；`default_provider` 可覆盖默认首选项。
- `aliases.<alias>`：为 CLI 提供友好别名，可选带上固定的 `provider`。
- 环境变量 `LLM_MODEL_CONFIG` 可指定自定义 YAML 路径。

端点轮换（Endpoint Rotation）：若某模型在 YAML 中列出多个提供方，则在出现限流、过载或网络异常时会自动切换至下一个端点继续尝试。

- 顺序：CLI `--model-provider`（若提供） → alias 固定 provider → `default_provider` → 其余 providers 顺序。
- 触发：
  - 首次生成阶段（`services/attempts.py`）的可重试错误
  - 反馈阶段（`services/testing.py`）的可重试错误
- 仍遵循 `--max-retries` 与 `--retry-wait`，若仅配置单一提供方则不会发生轮换。

## 日志与 JSONL 规范

### 会话交互（logs/conversations/all_conversations.jsonl）
- 统一字段（无冗余别名）：
  - `attempt_number`（1 基）
  - `round_number`（1 基）
  - `retry_index`（0 基）
  - `timestamp`、`interaction_type`（request/response）、`role`、`content`
  - 可选：`thinking`、`content_without_thinking`、`model_used`、`success`

示例：
```
{"attempt_number":1, "round_number":1, "retry_index":0, "interaction_type":"request",  ...}
{"attempt_number":1, "round_number":1, "retry_index":0, "interaction_type":"response", ...}
```

### 重试事件（logs/retry_events.jsonl）
- 字段：`_type`=`retry_event`、`event`（attempt_start/attempt_finish/retry_wait）、`stage`、`case_type`、`case_name`、`attempt_number`、`round_id`、`retry_index`、`timestamp`、`context`

### 实时统计（根目录 *.jsonl）
- 由 `AttemptTimingStats`、`CaseResult`、`BatchSummary` 写入，字段见 `data/models.py`
  - `attempt_number` 明确为 1 基

## 命名建议：attempt_number vs attempt_index vs attempt_num
- 推荐使用 `attempt_number`（序数语义，1 基）。`index` 更常用于 0 基，容易与 `retry_index` 混淆；`attempt_num` 容易被误解为“数量（count）”。
- 对于重试维度，保留 `retry_index`（0 基）更贴切工程语义。
