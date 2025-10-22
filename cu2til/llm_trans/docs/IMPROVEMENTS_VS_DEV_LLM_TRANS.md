# 相较于 legacy trans/dev_/llm_trans.py 的改进

本模块在功能等价的基础上，针对可维护性、健壮性与可观测性做了系统性改进：

## 结构与职责
- 单文件巨石拆分为模块化包：配置、运行时、客户端、数据模型、服务（会话/历史/重试/测试/尝试/编排）分层清晰
- 明确的数据模型：以 dataclass（如 `AttemptTimingStats`/`CaseResult`/`BatchSummary`）替代散落 dict，统一字段与类型
- 运行时上下文 `RuntimeContext`：集中持有 logger、模型客户端、用例集合与 I/O 锁

## 模型端点与鲁棒性
- 引入 `config/model_clients.yaml`：集中描述模型服务商、提供方和模型的层次关系，可按别名/完整名在 CLI 中选择
- `--model-provider` 支持强制使用指定提供方；若配置多个提供方则在限流/网络异常时按顺序轮换
- Provider 配置支持直接写常量或 `$ENV` / `{env: ENV, default: ...}` 形式，兼容本地部署、聚合服务或官方接口，并在解析阶段完成环境变量替换；同时预置 `local-127-800{1,2,3}`、`local-172-800{1,2,3}` 等常见机房节点便于一键切换

## 会话与历史
- `AttemptHistoryManager`：基于 `ConversationTree` 管理多轮会话，支持 `--resume-conversation` 续跑
- 会话 JSONL 统一：仅写入 `attempt_number`（1 基）、`round_number`（1 基）、`retry_index`（0 基），不再写入冗余别名字段

## 测试与执行
- 本地/NVGPU 双模式；NVGPU 详尽时延分解（pending/queue/waiting/execution/total），并写入结构化 `TestRoundRecord`
- 异步并发调度：`runner.py` 使用 `asyncio` 与信号量进行批量并发

## 日志与可观测性
- JSONL 实时统计由数据模型统一输出，避免字段漂移
- retry 事件独立落盘（`retry_events.jsonl`）用于审计与追踪
- 细化毫秒格式显示，可配置 `--ms-format`（千分位或纯数值）

## 命名与语义
- 采用 `attempt_number`（1 基）与 `retry_index`（0 基）的混合语义，既对人友好，又与工程习惯一致；避免使用易歧义的 `attempt_num`

## 兼容性
- 保留 legacy 启动路径（`python cu2til/trans/dev_/llm_trans.py`）与新入口（`python -m cu2til.llm_trans`）
- 会话 JSONL 统一使用 `attempt_number`/`round_number`，旧字段停止输出以避免歧义
