TVM 源码编译与配置记录（CUDA + LLVM）

概述
- 源码位置：`/workspace/third_party/tvm`
- 目标：开启 CUDA/cuDNN/cuBLAS、LLVM、CUTLASS，并生成共享库与 Python FFI
- 最终产物：
  - `third_party/tvm/build/libtvm.so`
  - `third_party/tvm/build/libtvm_runtime.so`
  - `third_party/tvm/build/libtvm_allvisible.so`
  - `third_party/tvm/build/lib/libtvm_ffi.so`

编译环境
- GCC/G++: 13.3.0
- CUDA: 12.9 (`/usr/local/cuda`)，`CMAKE_CUDA_ARCHITECTURES=89`
- cuDNN: `/usr/lib/x86_64-linux-gnu/libcudnn.so`（CMake 自动找到）
- cuBLAS: `/usr/local/cuda/lib64/libcublas.so`（CMake 自动找到）
- LLVM: 18（`/usr/bin/llvm-config-18`）
- Python: 3.12（CMake 自动加入 Cython/FFI 构建）
- PyTorch CMake 配置目录：`/usr/local/lib/python3.12/dist-packages/torch/share/cmake`（用于 `USE_LIBTORCH`）

环境变量（运行/开发时）
- 在 shell 中设置（可加入 `~/.bashrc`）：
  - `export CUDA_ARCH_LIST=8.9`（与 `CMAKE_CUDA_ARCHITECTURES=89` 对应，Ada 架构）
  - `export TVM_HOME=/workspace/third_party/tvm`
  - `export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/ffi/python:$PYTHONPATH`
  - `export TVM_LIBRARY_PATH=/workspace/third_party/tvm/build`
  - 可选：`export LD_LIBRARY_PATH=$TVM_LIBRARY_PATH:$LD_LIBRARY_PATH`（若运行时找不到 `libtvm.so`）

关键配置修改
- 模板文件：`/workspace/third_party/tvm/cmake/config.cmake`
- 最终使用：`/workspace/third_party/tvm/build/config.cmake`

与模板相比，我做了如下开启/新增（只列出差异）：
- `set(USE_CUDA ON)`，并新增 `set(CMAKE_CUDA_ARCHITECTURES 89)`
- `set(USE_CUDNN ON)`
- `set(USE_CUBLAS ON)`
- `set(USE_LLVM /usr/bin/llvm-config-18)`
- `set(HIDE_PRIVATE_SYMBOLS ON)`
- `set(USE_CUTLASS ON)`
- `set(USE_LIBTORCH /usr/local/lib/python3.12/dist-packages/torch/share/cmake)`
- 新增：`set(CMAKE_BUILD_TYPE RelWithDebInfo)`

完整的最终配置见：`third_party/tvm/build/config.cmake`

构建步骤
1) 进入源码目录并准备构建目录
   - `cd /workspace/third_party/tvm`
   - `mkdir -p build && cp cmake/config.cmake build/config.cmake`
   - 按“关键配置修改”一节更新 `build/config.cmake`。

2) 生成构建文件
   - `cd build`
   - `cmake ..`（可显式加上 `-DCMAKE_BUILD_TYPE=RelWithDebInfo`）
   - 关键输出（来自 tmux 日志）包含：
     - 找到 CUDA 12.9，cuDNN、cuBLAS 路径
     - `Use llvm-config=/usr/bin/llvm-config-18`
     - `Build with CUTLASS`、`Build with contrib.random`、`Build with contrib.sort`
     - `Add Cython build into the default build step`
     - `Configuring done` / `Generating done` / `Build files have been written to: ...`

3) 编译
   - `make -j$(nproc)`
   - 产物时间戳：
     - `third_party/tvm/build/libtvm_runtime.so`（05:09）
     - `third_party/tvm/build/3rdparty/libflash_attn/src/libflash_attn.so`（04:20）
     - `third_party/tvm/build/libtvm.so` 与 `libtvm_allvisible.so`（05:11）

为什么最后安装成功（问题排查与关键信息）
- 关键信息都记录在：`/workspace/logs/tmux/tvm.log`
- 日志显示：
  - 成功找到 CUDA/LLVM/cuDNN/cuBLAS，并启用 CUTLASS 与 Python FFI（Cython）。
  - `Configuring done` 与 `Generating done` 表明 CMake 配置成功；随后 `Build files have been written to`。
  - 后续 `make` 阶段成功生成上述共享库，说明最终构建完成。
- 早期构建中（参考另一个 `tvm.log` 片段）曾出现 `__hfma2` 未定义等 CUDA 半精度相关报错，通常由架构编译目标不匹配或头文件/宏缺失引起。此次成功构建的关键在于：
  - 明确设置了 `CMAKE_CUDA_ARCHITECTURES=89`，确保为实际 GPU 架构生成合适的 PTX/二进制；
  - 正确启用了 cuDNN/cuBLAS 并指定了 LLVM 18 路径，使 CMake 能一次性解析出完整依赖；
  - CMake 输出显示已将 Cython/FFI 纳入默认构建步骤，从而同时生成 `libtvm_ffi.so`。

常见问题提示
- 如遇 `__hfma2` 未定义或半精度相关编译错误：
  - 确认 `CMAKE_CUDA_ARCHITECTURES` 与目标 GPU 匹配（如 Ada 为 89）。
  - 确认使用的 CUDA 版本与 GPU/驱动兼容（本次为 12.9）。
- 如 `llvm-config` 未找到：
  - 调整 `USE_LLVM` 为正确的 `llvm-config` 路径（例如 `/usr/bin/llvm-config-18`）。
- 如 Python 绑定未生成：
  - 确认 CMake 有检测到 Python 解释器，并看到 “Add Cython build into the default build step”。

验证
- 查看生成的库是否存在：
  - `ls -la /workspace/third_party/tvm/build/libtvm*.so`
  - `ls -la /workspace/third_party/tvm/build/lib/libtvm_ffi.so`
- 设置环境并在 Python 中验证：
  - `export TVM_HOME=/workspace/third_party/tvm`
  - `export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/ffi/python:$PYTHONPATH`
  - `export TVM_LIBRARY_PATH=/workspace/third_party/tvm/build`
  - `python - <<'PY'
import tvm
print('cuda enabled:', tvm.runtime.enabled('cuda'))
from tvm.contrib import utils
print('tvm version:', tvm.__version__)
PY`



```shell
cd tvm
rm -rf build && mkdir build && cd build
# Specify the build configuration via CMake options
cp ../cmake/config.cmake .

...

apt install clang-18 lldb-18 lld-18 llvm-18-dev
apt install clang-tidy-18 clang-format-18
# conda install -c conda-forge clangxx=18 clang=18 lld=18 clang-tools=18 llvmdev=18
...


python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
export CUDA_ARCH_LIST=8.9
cmake ..
cmake --build . --parallel $(nproc) -v
cd ../3rdparty/tvm-ffi; pip install .; cd ..
export TVM_HOME=/path-to-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/ffi/python:$PYTHONPATH
export TVM_LIBRARY_PATH=/path-to-tvm/build
pip install -e /path-to-tvm/python
python -c "import tvm; print(tvm.base._LIB)"
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
python -c "import tvm; print(tvm.cuda().exist)"
pip3 install ml_dtypes tornado psutil 'xgboost>=1.1.0' cloudpickle scipy einops ninja
```
