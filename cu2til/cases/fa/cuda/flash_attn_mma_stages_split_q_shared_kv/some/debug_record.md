好的！我来修复 `kernel_prim.cu` 中的shuffle相关bug，应用刚才修复的DSL API。
Ran tool

Ran tool

Ran tool

Ran tool

Ran tool

## ✅ 修复完成！Flash Attention DSL内核已修正

我已经成功修复了 `kernel_prim.cu` 中的所有shuffle相关bug，现在它应该与 `kernel_expanded.cu` 在数学上等价了。

### 🎯 具体修复内容

#### 1. **Sub-Warp Max Reduction修复** (第484-485行)
```cpp
// ❌ 修复前（错误）：4被当作mask参数，实际在32个thread内reduction
lane_row_max_new[0][0] = warp_shuffle_max<fp32>(lane_row_max_new[0][0], 4);
lane_row_max_new[0][1] = warp_shuffle_max<fp32>(lane_row_max_new[0][1], 4);

// ✅ 修复后（正确）：4是width参数，正确地在4个thread内reduction
lane_row_max_new[0][0] = warp_shuffle_max_width<fp32>(lane_row_max_new[0][0], 4);
lane_row_max_new[0][1] = warp_shuffle_max_width<fp32>(lane_row_max_new[0][1], 4);
```

#### 2. **Sub-Warp Sum Reduction修复** (第530-531行)
```cpp
// ❌ 修复前（错误）：4被当作mask参数，实际在32个thread内reduction
lane_row_sum_new[0][0] = warp_shuffle_sum<fp32>(lane_row_sum_new[0][0], 4);
lane_row_sum_new[0][1] = warp_shuffle_sum<fp32>(lane_row_sum_new[0][1], 4);

// ✅ 修复后（正确）：4是width参数，正确地在4个thread内reduction
lane_row_sum_new[0][0] = warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][0], 4);
lane_row_sum_new[0][1] = warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][1], 4);
```

#### 3. **MMA结果重分布修复** (第711行和739行)
```cpp
// ❌ 修复前（错误）：缺少width参数，在32个thread内shuffle
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0], R_Q[1][0], lane_id);
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Z[0], R_Z[1], lane_id);

// ✅ 修复后（正确）：添加width=4参数，正确地在4个thread内shuffle
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0], R_Q[1][0], lane_id, 4);
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Z[0], R_Z[1], lane_id, 4);
```

### 🚀 修复效果

现在两个内核版本应该：

1. **数学等价**：`kernel_prim.cu` 和 `kernel_expanded.cu` 产生完全相同的计算结果
2. **语义正确**：所有shuffle操作都在正确的thread组内进行（4-thread sub-warp而非32-thread warp）
3. **Flash Attention算法正确**：softmax计算的max reduction和sum reduction都在正确的范围内进行

### 🔍 验证要点

Flash Attention内核现在：
- Max reduction正确地在每4个thread内找最大值（而非整个warp的32个thread）
- Sum reduction正确地在每4个thread内求和（而非整个warp的32个thread）  
- MMA结果重分布正确地在4-thread组内进行数据交换（而非跨组交换）

这些修复解决了原始bug报告中提到的"两个kernel不等价"问题的根本原因。现在DSL版本应该与原始PTX版本产生位精度相同的计算结果。