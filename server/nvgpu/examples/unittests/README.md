NVGPU 示例回归单测套件

概述
- 该套件将 examples 中的演示脚本转化为“面向行为的回归测试”，避免仅以返回码判断是否成功。
- 测试覆盖提交/等待/取消/日志/模式切换/并发上限/参数智能默认与覆盖等核心行为，并在必要场景对服务器日志进行顺序校验。

运行方式
- 启动服务端（需要 GPU 和依赖）：
  - 进入 `server/nvgpu` 目录，执行 `python main.py`
  - 可设置环境变量 `NVGPU_SERVER_LOG=/workspace/server/nvgpu/logs/nvgpu_server_YYYYMMDD_HHMMSS.log` 指定当前会话日志；否则测试会使用 `logs` 下最新的 `nvgpu_server_*.log`
- 运行测试：
  - `python -m unittest discover -s server/nvgpu/examples/unittests -p 'test_*.py' -v`

测试清单与断言重点
- test_example_basic.py：
  - 提交 `simple_functional_test` 并等待完成
  - 断言：TaskResult 字段、任务日志包含成功标志、服务器主日志包含本次 task_id 的“submitted … simple_functional_test.py”与“completed on GPU”记录（顺序校验）
- test_example_task_parameters.py：
  - 覆盖智能默认与覆盖策略：functional→shared、performance→exclusive、both→exclusive、functional+exclusive、performance+shared
  - 断言：`get_task` 返回的 `task_mode` 与期望一致，并等待完成
- test_example_cancel_running_task.py：
  - 提交长任务等到 running，`force=True` 取消
  - 断言：最终状态 `cancelled`，错误消息包含“Cancelled”；服务器主日志包含该 task 的 “force cancelled” 记录（顺序校验：submitted→(queued|running)→terminated/force cancelled）
- test_example_log_handling.py：
  - 断言 `/tasks/{id}/log` 接口在 `summary/stdout/stderr` 三类日志返回结构与数据正确；`get_full_task_log` 包含“Simple Functional Test”
- test_example_batch_submit.py：
  - 同时提交功能/性能三类任务（性能任务独占），断言全部 `completed` 且 `exit_code=0`
- test_example_concurrent_tasks.py：
  - 将某 GPU 设置为 `shared` 且 `max_concurrent_tasks=2`，提交 3 个长任务
  - 断言：采样期内 `running_task_count` 从未超过 2（并发上限约束）
- test_example_gpu_management.py：
  - 将 GPU 设为 `exclusive` 后，先后提交两个任务
  - 断言：第二个任务在第一个完成前处于 `pending/queued`，随后能完成；最后恢复为 `shared`
- test_example_mode_switching.py：
  - `shared` 下提交两个任务开始运行→切换为 `exclusive`→提交新任务
  - 断言：新任务在前两个完成前保持 `pending/queued`，完成后能恢复运行；最后恢复为 `shared`

强顺序匹配策略
- 仅在“日志可作为协议”的场景使用强顺序（例如取消流程与基本提交流程），采用相对稳健的关键标记序列匹配：
  - 格式：使用 `assert_patterns_in_order` 对同一 task_id 的若干关键事件（regex）做相对顺序校验
  - 容忍：允许中间穿插其他异步日志；不要求连续行，仅要求相对顺序
- 不在高并发/调度存在自然抖动的场景对所有细粒度事件做强顺序（如“exact 时刻/每次轮询的状态”），改用“安全不变量”断言（如`running_task_count ≤ 上限`、“新任务在旧任务完成前不得运行”等）。

稳定性与确定性
- 任务时长：长任务使用较短可控时长与轮询间隔，降低执行时间同时保证可观测性；
- 日志定位：通过 `NVGPU_SERVER_LOG` 或“最新日志”+ 文件偏移，仅匹配本次测试新增内容，避免历史日志干扰；
- 清理：大多数测试等待任务结束；如需进一步缩短时间，可将示例脚本参数再下调（如 `--duration`）。

环境依赖
- 服务端 HTTP：默认 `http://localhost:8080`
- GPU：示例脚本依赖 CUDA（PyTorch），在无 GPU 环境下这些测试应跳过（由 `@requires_server` 控制）

可配置项
- `NVGPU_SERVER_LOG`：服务器当前会话日志路径
- 未来可扩展：
  - `NVGPU_TEST_FAST=1` 用于降低任务规模/时长
  - `NVGPU_TEST_STRICT_LOG=1` 启动更多日志顺序校验

如何扩展测试
- 在 `server/nvgpu/examples/unittests` 下添加新的 `test_*.py` 文件；
- 复用 `utils.py` 中的工具函数（获取客户端、定位日志、顺序断言）；
- 优先断言“可观测的、与需求直接相关的行为”，避免依赖易波动的实现细节或脆弱的精确时间顺序。

