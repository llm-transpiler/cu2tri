# Deformable-DETR 自定义算子在 PyTorch 2.x / CUDA 13 的兼容性修改

> 适用路径：`third_party/Deformable-DETR/models/ops`

## 背景与报错概述
在 PyTorch 2.x 中，`Tensor.type()` 被标记为弃用，不再适合作为 `AT_DISPATCH_*` 宏的入参；应改用 `Tensor.scalar_type()`。同时，`tensor.type().is_cuda()` 也应替换为 `tensor.is_cuda()`。否则在编译自定义 CUDA 扩展时会出现如下错误：

- `no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists`
- 报错位置（示例）：
  - `src/cuda/ms_deform_attn_cuda.cu:64`（forward）
  - `src/cuda/ms_deform_attn_cuda.cu:134`（backward）
  - 头文件中会出现 `Tensor.type()` 的弃用告警

## 修改清单（你已完成）
1) `AT_DISPATCH_*` 宏的第一个入参：
   - 将 `value.type()` → `value.scalar_type()`。
   - 受影响位置：
     - `src/cuda/ms_deform_attn_cuda.cu` 中 forward 分发处（约第 64 行）。
     - `src/cuda/ms_deform_attn_cuda.cu` 中 backward 分发处（约第 134 行）。

   示例：
   ```cpp
   // before
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
       value.type(), "ms_deform_attn_forward_cuda", [&] { /* ... */ });

   // after
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
       value.scalar_type(), "ms_deform_attn_forward_cuda", [&] { /* ... */ });
   ```

2) CUDA 设备判断：
   - 将 `value.type().is_cuda()` → `value.is_cuda()`。
   - 受影响位置：`src/ms_deform_attn.h`（forward/backward 的 CUDA 分支检查）。

   示例：
   ```cpp
   // before
   if (value.type().is_cuda()) { /* ... */ }

   // after
   if (value.is_cuda()) { /* ... */ }
   ```

3) 可选的现代化更新（按需）：
   - 若代码中仍使用 `tensor.data<T>()`，建议替换为 `tensor.data_ptr<T>()`（PyTorch 2.x 更推荐）。
   - 若有使用 `tensor.type().scalar_type()` 或 `tensor.type().backend()`：
     - 改为 `tensor.scalar_type()` 与 `tensor.device()`。

## 受影响文件
- `third_party/Deformable-DETR/models/ops/src/cuda/ms_deform_attn_cuda.cu`
  - forward/backward 的 `AT_DISPATCH_*` 入口入参改为 `value.scalar_type()`
- `third_party/Deformable-DETR/models/ops/src/ms_deform_attn.h`
  - `value.type().is_cuda()` 改为 `value.is_cuda()`

## 编译与清理步骤
如需重新编译：

```bash
cd /data/apps/project/cu2tri/third_party/Deformable-DETR/models/ops
python setup.py clean || true
rm -rf build
sh ./make.sh
```

说明：
- 日志中的 `There are no g++ version bounds defined for CUDA version 13.0` 为警告，不影响本次修复点。
- 若使用 H100/Ada 等新卡，`nvcc` 的 `-gencode` 已包含 `compute_90/sm_90`；如需变更，请在 `setup.py` 或构建脚本中调整。
- 请确认 `CUDA_HOME=/usr/local/cuda`，并与安装的 PyTorch CUDA 版本匹配。

## 预期结果
完成以上修改后，`MultiScaleDeformableAttention` 扩展应能在 PyTorch 2.x + CUDA 13 环境下顺利编译通过。
