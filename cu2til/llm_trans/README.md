# cu2til/llm_trans 模块化重构说明

本目录已经拆分为若干子 package，以职责驱动的方式组织代码，并用显式的数据模型取代散落的字典结构，所有对外接口（含旧路径）保持兼容。下文按分层介绍各模块及关键类/函数，便于后续维护与扩展。

## 入口层
- `cli.py`：负责解析命令行、构建 `Settings`/`RuntimeContext`、触发异步执行入口 `services.runner.run`。导出 `prepare_context`、`main` 供外部调用。
- `__main__.py`：允许 `python -m cu2til.llm_trans ...` 直接启动。
- `trans/dev_/llm_trans.py`：保留历史脚本路径，内部仅代理至 `cu2til.llm_trans.cli.main`。

## 配置与运行时
- `config/args.py`：定义 `parse_cli_args`，集中维护所有 CLI 参数声明。
- `config/settings.py`：实现 `Settings` 数据类与 `build_settings` 工厂，负责加载环境变量、推导工作目录、管理测试集元数据等。
- `core/runtime.py`：声明 `RuntimeContext`，封装运行期所需的 logger、模型客户端、用例集合以及 JSONL/重试日志锁。

## 客户端与通用工具
- `clients/model.py`：提供 `ModelClients` 数据类及 `create_model_clients` 工厂，内部根据 CLI 参数选择 LLM 接入方式，同时暴露 `async_openai_llm_call` 等底层工具函数。
- `clients/nvgpu.py`：惰性导入 NVGPU 客户端，并导出 `NVGPU_AVAILABLE` 标志。
- `utils/formatting.py`：提供 `format_ms` 毫秒格式化工具。
- `io/logging.py`：根据 `Settings` 创建文件/控制台双写的日志记录器，避免重复附加 handler。
- `io/jsonl.py`：异步安全地写入 JSONL，自动兼容 dataclass/Mapping，两把锁分别由 `RuntimeContext` 管理。

## 数据模型
- `data/models.py`：定义 `RetryRecord`、`RoundRecord`、`TestRoundRecord`、`AttemptTimingStats`、`AttemptResult`、`CaseResult`、`BatchSummary` 等 dataclass。所有统计信息与输出日志均依赖这些结构，摒弃原先的动态 dict 拼装，便于类型检查与复用。

## 业务服务子 package
- `services/cases.py`：封装测试集解析、筛选逻辑，并为 CLI 选择提供统一接口。
- `services/conversation.py`：负责持久化会话、提取思维链、保存 JSONL 交互。
- `services/history.py`：封装 `AttemptHistoryManager`，基于 `llm.history.ConversationTree` 记录多轮对话、尝试与重试信息，并支持从历史记录恢复会话。
- `services/retry.py`：统一处理网络/限流错误判定、重试上下文序列化、重试事件落盘。
- `services/testing.py`：异步测试与修复主流程，接入历史管理器并协调本地 / NVGPU 双模式。
- `services/attempts.py`：以 dataclass 聚合单次尝试的全量统计信息，并支持多 attempt 调度策略；对外仅暴露异步接口。
- `services/runner.py`：批量 orchestrator，负责并发调度各用例、汇总并写出 `BatchSummary`，同时输出结构化的 `CaseResult`。

## 兼容性与使用方式
- 运行入口保持不变：
  ```bash
  python -m cu2til.llm_trans --model gpt --testset xpiler
  ```
  - 用例来源改为 YAML/目录扫描：`cu2til/llm_trans/config/case_config.yaml`
  - `--direction {cu2tri, tri2cute}` 控制 CUDA→Triton 或 Triton→CUTE 转译流程
  - `--target-gpu {auto,h800_sxm,h100_pcie,h800_pcie,rtx6000_ada,a800_sxm,rtx5090}` 选择提示/测试所针对的 GPU 画像（默认：H800 SXM，自动携带 nvcc gencode）
  - `--testset` 现已覆盖 `xpiler*`、`triton_tutorial`、`flaggems_ops`, `unsloth_kernels`, `ligerkernel_ops` 等目录，不同来源的 Triton kernel 均可独立批测
  - 列出用例工具：
    ```bash
    python -m cu2til.llm_trans.utils.list_cases --testset xpiler --format json
    ```
  或通过历史脚本 `python cu2til/trans/dev_/llm_trans.py ...`。

## 模型 / 端点配置
- 所有 LLM 供应商、模型、提供方均集中在 `config/model_clients.yaml` 中维护：
  - `vendors.<vendor>.providers.<provider>` 描述具体端点（`base_url`、`api_key`、`calling`）。
  - `vendors.<vendor>.models.<model>.providers` 给出可用提供方顺序，`default_provider` 指定默认项（缺省为列表首个），`display_name` 可选。
  - `aliases` 允许为常用模型定义短名称（与 CLI `--model` 参数兼容旧值）。
  - `base_url` / `api_key` 字段可直接写常量，也可使用 `$ENV_NAME` 或 `{env: ENV_NAME, default: ...}` 形式引用环境变量。
  - 预置多组本地端点：`local`、`local-127-800{1,2,3}`、`local-172-800{1,2,3}` 等，可通过 `--model-provider local-172-8002` 直接切换到对应 IP/端口的自建 vLLM/sglang 服务。
  - 更多 OpenRouter 模型可以参考 `/data/apps/project/cu2tri/tools/price/openrouter_api.csv` 并按需追加到 YAML。
- CLI 用法：
  - `--model` 接受别名或完整模型名（如 `openai/gpt-5-mini`）。
  - `--model-provider` 可强制优先使用某个提供方（需在 YAML 中定义）。
  - 环境变量 `LLM_MODEL_CONFIG` 指向自定义 YAML 时，可覆盖默认文件位置。
- 预置别名示例：`gpt`→`openai/gpt-5-mini`、`gpt5`→`openai/gpt-5`、`gpt_codex`→`openai/gpt-5-codex`、`gpt_local_172_8002`→`openai/gpt-oss-120b@local-172-8002`、`qwen_local_127_8001`→`qwen/qwen3-max@127.0.0.1:8001`、`deepseek_local`→`deepseek/deepseek-r1-0528@127.0.0.1:8003`、`grok`→`xai/grok-4`、`gemini`→`google/gemini-2.5-pro`、`claude`→`anthropic/claude-sonnet-4.5`、`deepseek_v32`→`deepseek/deepseek-v3.2` 等。

## 端点轮换
- 若模型配置了多个提供方，则在请求失败且被判定为可重试时会按顺序轮换。
- 顺序遵循：命令行 `--model-provider`（若提供） → 别名指定的 `provider` → `default_provider` → 其余列表顺序。
- 若希望禁用轮换，可在 YAML 中仅为该模型保留一个提供方。

## 设计收益概述
1. **结构清晰**：各子 package 按领域职责划分，避免单文件巨石代码。
2. **数据模型统一**：所有原本用 dict 描述的统计结果均升级为 dataclass，调用方获得明确字段与类型提示。
3. **接口分层**：同步工具（配置、日志、格式化）与异步业务（LLM 调度、测试执行）分离，便于针对性测试与扩展。
4. **向后兼容**：旧的模块导入/脚本调用路径仍然可用，迁移成本极低。

后续若需扩展更多后端或调度策略，只需在对应子 package 内新增实现，并通过数据模型/服务接口衔接即可。

## 会话历史与恢复
- CLI 新增 `--resume-conversation`，可在 `attempt_XX` 目录下保留 `logs/history/conversation.json` 时继续先前会话，而不会重复初始化提示。
- `AttemptHistoryManager` 会在每次用户/助手交互、错误与等待期间更新历史文件，可在多次运行间共享上下文。
- 历史文件与原有 `save_conversation_history` JSON 互补：前者用于结构化恢复，后者仍便于快速排查。
