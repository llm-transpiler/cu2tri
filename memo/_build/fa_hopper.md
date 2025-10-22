FlashAttention-3 (hopper) 构建问题排查与修复记录

更新时间：2025-10-21

一、问题概述
- 构建入口：third_party/flash-attention/hopper，命令 `python setup.py install`。
- 起始错误（logs/build-fa.txt）：
  - fatal error: cuda/std/utility: No such file or directory
  - fatal error: cub/cub.cuh: No such file or directory
- 原因：CUDA 13 将 libcudacxx（cuda/std）、CUB、Thrust 等头文件放在 `include/cccl/` 下，但编译器未包含该路径，导致找不到头文件。

二、环境与证据
- CUDA 安装：`/usr/local/cuda -> /usr/local/cuda-13.0/`；`/usr/local/cuda/include -> targets/x86_64-linux/include`。
- 实际文件存在：
  - `/usr/local/cuda/targets/x86_64-linux/include/cccl/cuda/std/utility`
  - `/usr/local/cuda/targets/x86_64-linux/include/cccl/cub/cub.cuh`
- 但顶层没有传统的 `include/cuda/std` 与 `include/cub` 直达目录，需要显式包含 `cccl`。

三、首次修复尝试（仅环境变量）
- 设置：`export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:$CPATH`
- 效果：头文件查找错误消失，但出现新报错：
  - `/usr/local/cuda/.../cccl/cuda/std/__cccl/cuda_toolkit.h:39:6: error: "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"`

四、根因复盘（新错误）
- 构建脚本 `hopper/setup.py` 会在 CUDA 不是 12.8 时自动下载并强制使用 NVCC/ptxas：
  - 下载 NVCC 12.6.85 与 PTXAS 12.8.93（见日志中 “copy …/cuda_nvcc-…-12.6.85-archive/bin …”）。
  - 设置环境变量 `PYTORCH_NVCC` 指向下载的 NVCC，从而用 12.x 编译器去编译 CUDA 13 的头文件。
- 这与本机 CUDA 13.0 头文件不匹配，触发 libcudacxx 自检报错（即“编译器与工具包头文件不兼容”）。

五、最终修复方案（补丁 + 正确使用系统 NVCC）
1) 修改 setup.py（新增环境变量开关，避免强制覆盖 NVCC）
   - 文件：`third_party/flash-attention/hopper/setup.py`
   - 变更要点：
     - 新增环境变量 `FLASH_ATTENTION_USE_SYSTEM_NVCC`（TRUE 时使用系统 NVCC，不下载/不覆盖 `PYTORCH_NVCC`）。
     - 将 `CUDA_HOME/include/cccl` 加入 `include_dirs`，不再依赖外部 `CPATH`。
   - 关键片段（示意）：
     - 定义开关：
       `USE_SYSTEM_NVCC = os.getenv("FLASH_ATTENTION_USE_SYSTEM_NVCC", "FALSE") == "TRUE"`
     - 跳过下载并覆盖 NVCC：
       `if bare_metal_version != Version("12.8") and not USE_SYSTEM_NVCC:`
     - 仅在未启用开关时设置 `PYTORCH_NVCC`：
       `if not USE_SYSTEM_NVCC: os.environ["PYTORCH_NVCC"] = nvcc_path_new`
     - 添加 CCCL 头路径：
       `include_dirs.append(Path(CUDA_HOME) / "include" / "cccl")  # 若存在`

2) 构建命令（建议步骤）
   - 环境准备：
     - 建议设置：`export FLASH_ATTENTION_USE_SYSTEM_NVCC=TRUE`
      - 只有这个好像就可以编译了
     - 可选：保留或取消之前的 `CPATH`；补丁已将 `cccl` 放入 `include_dirs`，不再强依赖 `CPATH`。
   - 清理并重建：
     - `rm -rf build`
     - `python setup.py install`（或 `pip install -v .`）

六、为什么这样可行
- 使用系统 NVCC 13.0 与本机 CUDA 13.0 头文件版本一致，避免“编译器与头文件不兼容”的硬错误。
- 明确把 `include/cccl` 加到 `include_dirs`，确保 `<cuda/std/...>` 与 `<cub/...>` 能解析。
- PyTorch 版本为 `2.9.0+cu130`，与 CUDA 13.0 匹配；日志中 “There are no g++ version bounds defined for CUDA 13.0” 仅为警告，不影响构建。

七、可选替代方案
- 系统级安装 CCCL 转发头（若不想改 setup.py）：
  - 安装 CUDA 13 对应的 cccl/dev 包，使顶层 `include/cuda/std`、`include/cub` 等转发就位。
  - 例如（NVIDIA 官方 apt 源）：`sudo apt install cuda-cccl-dev-13-0`。
  - 或安装 `libcudacxx-dev`、`libthrust-dev`、`libcublas-dev` 等对应包。
- 另一种思路（不推荐在本环境）：将 CUDA 降至 12.x 并与脚本默认下载的 12.x NVCC 对齐，但会与当前 PyTorch cu130 不一致。

八、排障要点与检查清单
- 头文件找不到：检查是否包含 `.../include/cccl`；可用 `CPATH` 或在构建系统 include_dirs 中加入。
- 工具链不兼容：检查是否被 `setup.py` 覆盖了 `PYTORCH_NVCC`，以及 NVCC 版本与 CUDA 头是否一致。
- 架构标志：脚本会对 `_sm80/_sm90` 源文件添加 `-gencode`，H100（sm_90a）与 A100（sm_80）均覆盖，无需手动修改。

九、复现场景（简要日志片段）
- 早期失败：
  - `fatal error: cuda/std/utility: No such file or directory`
  - `fatal error: cub/cub.cuh: No such file or directory`
- 加入 `CPATH` 后新失败：
  - `error: "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"`
  - 同时可见下载并强制使用 NVCC 12.6 的日志：`copy .../cuda_nvcc-...-12.6.85-archive/bin ...`

十、后续建议
- 保留本次补丁（USE_SYSTEM_NVCC 开关 + include_dirs 添加 cccl）在仓库中，方便不同 CUDA 版本环境通用构建。
- 在 CI 或多人环境中，明确约定：
  - CUDA 12.x 机器：默认脚本行为即可（不设 USE_SYSTEM_NVCC）。
  - CUDA 13.x 机器：设置 `FLASH_ATTENTION_USE_SYSTEM_NVCC=TRUE`，避免混用 12.x NVCC。

十一、核心步骤总结（速查）
- 关键环境变量（CUDA 13.x 场景）：
  - `export FLASH_ATTENTION_USE_SYSTEM_NVCC=TRUE`  使用系统 NVCC，避免 setup.py 强制下载 12.x 工具链。
  - 可选：`unset PYTORCH_NVCC`  确保不会意外指向下载的 nvcc 包装器。
  - 可选：`export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:$CPATH`  若未打补丁时临时添加 CCCL 头路径。
- 修改点（setup.py）：
  - 定义开关：`third_party/flash-attention/hopper/setup.py:47` 增加 `USE_SYSTEM_NVCC = ...`。
  - 跳过工具链下载：`third_party/flash-attention/hopper/setup.py:410` 条件改为 `and not USE_SYSTEM_NVCC`。
  - 避免覆盖 `PYTORCH_NVCC`：`third_party/flash-attention/hopper/setup.py:442` 仅在未启用开关时设置。
  - 添加 CCCL 头目录：`third_party/flash-attention/hopper/setup.py:581` 若 `CUDA_HOME/include/cccl` 存在则加入 `include_dirs`。
- 重建命令：
  - `cd third_party/flash-attention/hopper`
  - `rm -rf build`
  - `python setup.py install`  或 `pip install -v .`

