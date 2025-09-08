
#!/bin/bash

# ============================================================================
# 控制要测试的case类型列表 - 可以自由修改这个列表来选择要测试的类型
# ============================================================================
# 使用示例:
# 1. 测试所有类型(默认):       保持下面的列表包含所有21种类型
# 2. 只测试池化操作:          case_type_list=("avgpool" "maxpool" "minpool" "sumpool")
# 3. 只测试激活函数:          case_type_list=("relu" "sigmoid" "gelu" "softmax" "sign")
# 4. 只测试卷积相关:          case_type_list=("conv1d" "conv2d" "conv2dnchw" "depthwiseconv" "deformable")
# 5. 只测试矩阵运算:          case_type_list=("gemm" "gemv" "bmm")
# ============================================================================

case_type_list=(
    "add"
    "avgpool"
    "bmm"
    "conv1d"
    "conv2d"
    "conv2dnchw"
    "deformable"
    "depthwiseconv"
    "gelu"
    "gemm"
    "gemv"
    "layernorm"
    "maxpool"
    "mha"
    "minpool"
    "relu"
    "rmsnorm"
    "sigmoid"
    "sign"
    "softmax"
    "sumpool"
)

# 定义21种case类型，每种类型8个具体case
declare -A case_types=(
    # 1. 加法 (add) - 8个案例
    ["add"]="/workspace/cu2til/cases/xpiler/add_1_15_64
/workspace/cu2til/cases/xpiler/add_3_3_256
/workspace/cu2til/cases/xpiler/add_4_4_4_64
/workspace/cu2til/cases/xpiler/add_18_128
/workspace/cu2til/cases/xpiler/add_21_192
/workspace/cu2til/cases/xpiler/add_64
/workspace/cu2til/cases/xpiler/add_100_2_10_1024
/workspace/cu2til/cases/xpiler/add_320"

    # 2. 平均池化 (avgpool) - 8个案例
    ["avgpool"]="/workspace/cu2til/cases/xpiler/avgpool_1_5_5_64_5_5_1_1
/workspace/cu2til/cases/xpiler/avgpool_4_8_8_64_5_5_3_3
/workspace/cu2til/cases/xpiler/avgpool_4_35_35_192_5_5_2_2
/workspace/cu2til/cases/xpiler/avgpool_4_56_56_128_5_5_2_2
/workspace/cu2til/cases/xpiler/avgpool_5_32_32_64_5_5_3_3
/workspace/cu2til/cases/xpiler/avgpool_5_112_112_64_5_5_3_3
/workspace/cu2til/cases/xpiler/avgpool_16_64_64_64_5_5_2_2
/workspace/cu2til/cases/xpiler/avgpool_16_112_112_64_5_5_3_3"

    # 3. 批量矩阵乘法 (bmm) - 8个案例
    ["bmm"]="/workspace/cu2til/cases/xpiler/bmm_1_128_128_128
/workspace/cu2til/cases/xpiler/bmm_1_128_256_512
/workspace/cu2til/cases/xpiler/bmm_1_256_256_256
/workspace/cu2til/cases/xpiler/bmm_1_512_512_512
/workspace/cu2til/cases/xpiler/bmm_4_128_128_128
/workspace/cu2til/cases/xpiler/bmm_4_128_256_512
/workspace/cu2til/cases/xpiler/bmm_4_256_256_256
/workspace/cu2til/cases/xpiler/bmm_4_512_512_512"

    # 4. 1D卷积 (conv1d) - 8个案例
    ["conv1d"]="/workspace/cu2til/cases/xpiler/conv1d_5_7
/workspace/cu2til/cases/xpiler/conv1d_25_27
/workspace/cu2til/cases/xpiler/conv1d_44_46
/workspace/cu2til/cases/xpiler/conv1d_126_128
/workspace/cu2til/cases/xpiler/conv1d_190_192
/workspace/cu2til/cases/xpiler/conv1d_222_224
/workspace/cu2til/cases/xpiler/conv1d_256_258
/workspace/cu2til/cases/xpiler/conv1d_312_314"

    # 5. 2D卷积 (conv2d) - 8个案例
    ["conv2d"]="/workspace/cu2til/cases/xpiler/conv2d_16_8_8_64_64_2_2_64_2_0
/workspace/cu2til/cases/xpiler/conv2d_16_8_8_64_64_2_2_64_3_0
/workspace/cu2til/cases/xpiler/conv2d_16_8_8_128_64_2_2_128_2_0
/workspace/cu2til/cases/xpiler/conv2d_16_8_8_128_64_2_2_128_3_0
/workspace/cu2til/cases/xpiler/conv2d_32_8_8_64_64_2_2_64_2_0
/workspace/cu2til/cases/xpiler/conv2d_32_8_8_64_64_2_2_64_3_0
/workspace/cu2til/cases/xpiler/conv2d_32_8_8_128_64_2_2_128_2_0
/workspace/cu2til/cases/xpiler/conv2d_32_8_8_128_64_2_2_128_3_0"

    # 6. NCHW格式2D卷积 (conv2dnchw) - 8个案例
    ["conv2dnchw"]="/workspace/cu2til/cases/xpiler/conv2dnchw_16_64_8_8_128_64_2_2_2_0
/workspace/cu2til/cases/xpiler/conv2dnchw_16_64_8_8_128_64_2_2_3_0
/workspace/cu2til/cases/xpiler/conv2dnchw_16_128_8_8_64_128_2_2_2_0
/workspace/cu2til/cases/xpiler/conv2dnchw_16_128_8_8_64_128_2_2_3_0
/workspace/cu2til/cases/xpiler/conv2dnchw_32_64_8_8_128_64_2_2_2_0
/workspace/cu2til/cases/xpiler/conv2dnchw_32_64_8_8_128_64_2_2_3_0
/workspace/cu2til/cases/xpiler/conv2dnchw_32_128_8_8_64_128_2_2_2_0
/workspace/cu2til/cases/xpiler/conv2dnchw_32_128_8_8_64_128_2_2_3_0"

    # 7. 可变形卷积 (deformable) - 8个案例
    ["deformable"]="/workspace/cu2til/cases/xpiler/deformable_1_8_256_100_4_4
/workspace/cu2til/cases/xpiler/deformable_1_8_256_200_4_4
/workspace/cu2til/cases/xpiler/deformable_1_8_512_100_4_4
/workspace/cu2til/cases/xpiler/deformable_1_8_512_200_4_4
/workspace/cu2til/cases/xpiler/deformable_4_8_256_100_4_4
/workspace/cu2til/cases/xpiler/deformable_4_8_256_200_4_4
/workspace/cu2til/cases/xpiler/deformable_4_8_512_100_4_4
/workspace/cu2til/cases/xpiler/deformable_4_8_512_200_4_4"

    # 8. 深度卷积 (depthwiseconv) - 8个案例
    ["depthwiseconv"]="/workspace/cu2til/cases/xpiler/depthwiseconv_6_3_3
/workspace/cu2til/cases/xpiler/depthwiseconv_6_3_128
/workspace/cu2til/cases/xpiler/depthwiseconv_128_3_3
/workspace/cu2til/cases/xpiler/depthwiseconv_128_3_128
/workspace/cu2til/cases/xpiler/depthwiseconv_192_3_3
/workspace/cu2til/cases/xpiler/depthwiseconv_192_3_128
/workspace/cu2til/cases/xpiler/depthwiseconv_256_3_3
/workspace/cu2til/cases/xpiler/depthwiseconv_256_3_128"

    # 9. GELU激活 (gelu) - 8个案例
    ["gelu"]="/workspace/cu2til/cases/xpiler/gelu_3_4_5
/workspace/cu2til/cases/xpiler/gelu_5_7_3_32
/workspace/cu2til/cases/xpiler/gelu_5_12_23_128
/workspace/cu2til/cases/xpiler/gelu_5_128
/workspace/cu2til/cases/xpiler/gelu_7_1_6_7
/workspace/cu2til/cases/xpiler/gelu_8_10_64
/workspace/cu2til/cases/xpiler/gelu_12_3_128
/workspace/cu2til/cases/xpiler/gelu_45_25"

    # 10. 通用矩阵乘法 (gemm) - 8个案例
    ["gemm"]="/workspace/cu2til/cases/xpiler/gemm_32_32_128
/workspace/cu2til/cases/xpiler/gemm_32_32_1024
/workspace/cu2til/cases/xpiler/gemm_32_128_128
/workspace/cu2til/cases/xpiler/gemm_32_128_1024
/workspace/cu2til/cases/xpiler/gemm_1024_16_128
/workspace/cu2til/cases/xpiler/gemm_1024_16_1024
/workspace/cu2til/cases/xpiler/gemm_1024_128_128
/workspace/cu2til/cases/xpiler/gemm_1024_128_4096"

    # 11. 矩阵向量乘法 (gemv) - 8个案例
    ["gemv"]="/workspace/cu2til/cases/xpiler/gemv_3_16
/workspace/cu2til/cases/xpiler/gemv_3_512
/workspace/cu2til/cases/xpiler/gemv_32_64
/workspace/cu2til/cases/xpiler/gemv_32_512
/workspace/cu2til/cases/xpiler/gemv_112_128
/workspace/cu2til/cases/xpiler/gemv_112_224
/workspace/cu2til/cases/xpiler/gemv_125_128
/workspace/cu2til/cases/xpiler/gemv_125_320"

    # 12. 层归一化 (layernorm) - 8个案例
    ["layernorm"]="/workspace/cu2til/cases/xpiler/layernorm_1_4_32
/workspace/cu2til/cases/xpiler/layernorm_1_4_128
/workspace/cu2til/cases/xpiler/layernorm_1_8_32
/workspace/cu2til/cases/xpiler/layernorm_1_8_128
/workspace/cu2til/cases/xpiler/layernorm_2_4_32
/workspace/cu2til/cases/xpiler/layernorm_2_4_128
/workspace/cu2til/cases/xpiler/layernorm_2_8_32
/workspace/cu2til/cases/xpiler/layernorm_2_8_128"

    # 13. 最大池化 (maxpool) - 8个案例
    ["maxpool"]="/workspace/cu2til/cases/xpiler/maxpool_1_5_5_64_5_5_1_1
/workspace/cu2til/cases/xpiler/maxpool_4_8_8_64_5_5_3_3
/workspace/cu2til/cases/xpiler/maxpool_4_35_35_192_5_5_3_3
/workspace/cu2til/cases/xpiler/maxpool_4_56_56_128_5_5_2_2
/workspace/cu2til/cases/xpiler/maxpool_5_32_32_64_5_5_3_3
/workspace/cu2til/cases/xpiler/maxpool_5_112_112_64_5_5_2_2
/workspace/cu2til/cases/xpiler/maxpool_16_64_64_64_5_5_2_2
/workspace/cu2til/cases/xpiler/maxpool_16_112_112_64_5_5_3_3"

    # 14. 多头注意力 (mha) - 8个案例
    ["mha"]="/workspace/cu2til/cases/xpiler/mha_1_2048_6_256
/workspace/cu2til/cases/xpiler/mha_1_2048_12_256
/workspace/cu2til/cases/xpiler/mha_1_4096_6_256
/workspace/cu2til/cases/xpiler/mha_1_4096_12_256
/workspace/cu2til/cases/xpiler/mha_1_4096_12_512
/workspace/cu2til/cases/xpiler/mha_64_2048_12_256
/workspace/cu2til/cases/xpiler/mha_64_2048_12_512
/workspace/cu2til/cases/xpiler/mha_64_4096_12_256"

    # 15. 最小池化 (minpool) - 8个案例
    ["minpool"]="/workspace/cu2til/cases/xpiler/minpool_1_5_5_64_5_5_1_1
/workspace/cu2til/cases/xpiler/minpool_4_8_8_64_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_4_35_35_192_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_4_56_56_128_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_5_32_32_64_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_5_112_112_64_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_16_64_64_64_5_5_3_3
/workspace/cu2til/cases/xpiler/minpool_16_112_112_64_5_5_3_3"

    # 16. ReLU激活 (relu) - 8个案例
    ["relu"]="/workspace/cu2til/cases/xpiler/relu_3_4_5
/workspace/cu2til/cases/xpiler/relu_5_7_3_32
/workspace/cu2til/cases/xpiler/relu_5_12_23_128
/workspace/cu2til/cases/xpiler/relu_5_128
/workspace/cu2til/cases/xpiler/relu_7_1_6_7
/workspace/cu2til/cases/xpiler/relu_8_10_64
/workspace/cu2til/cases/xpiler/relu_12_3_128
/workspace/cu2til/cases/xpiler/relu_45_25"

    # 17. RMS归一化 (rmsnorm) - 8个案例
    ["rmsnorm"]="/workspace/cu2til/cases/xpiler/rmsnorm_2048_2048
/workspace/cu2til/cases/xpiler/rmsnorm_2048_4096
/workspace/cu2til/cases/xpiler/rmsnorm_2048_8192
/workspace/cu2til/cases/xpiler/rmsnorm_4096_2048
/workspace/cu2til/cases/xpiler/rmsnorm_4096_4096
/workspace/cu2til/cases/xpiler/rmsnorm_4096_8192
/workspace/cu2til/cases/xpiler/rmsnorm_8192_4096
/workspace/cu2til/cases/xpiler/rmsnorm_8192_8192"

    # 18. Sigmoid激活 (sigmoid) - 8个案例
    ["sigmoid"]="/workspace/cu2til/cases/xpiler/sigmoid_3_4_5
/workspace/cu2til/cases/xpiler/sigmoid_5_7_3_32
/workspace/cu2til/cases/xpiler/sigmoid_5_12_23_128
/workspace/cu2til/cases/xpiler/sigmoid_5_128
/workspace/cu2til/cases/xpiler/sigmoid_7_1_6_7
/workspace/cu2til/cases/xpiler/sigmoid_8_10_64
/workspace/cu2til/cases/xpiler/sigmoid_12_3_128
/workspace/cu2til/cases/xpiler/sigmoid_45_25"

    # 19. 符号函数 (sign) - 8个案例
    ["sign"]="/workspace/cu2til/cases/xpiler/sign_3_4_5
/workspace/cu2til/cases/xpiler/sign_5_7_3_32
/workspace/cu2til/cases/xpiler/sign_5_12_23_128
/workspace/cu2til/cases/xpiler/sign_5_128
/workspace/cu2til/cases/xpiler/sign_7_1_6_7
/workspace/cu2til/cases/xpiler/sign_8_10_64
/workspace/cu2til/cases/xpiler/sign_12_3_128
/workspace/cu2til/cases/xpiler/sign_45_25"

    # 20. Softmax激活 (softmax) - 8个案例
    ["softmax"]="/workspace/cu2til/cases/xpiler/softmax_3_4_5
/workspace/cu2til/cases/xpiler/softmax_5_7_3_32
/workspace/cu2til/cases/xpiler/softmax_5_12_23_128
/workspace/cu2til/cases/xpiler/softmax_5_128
/workspace/cu2til/cases/xpiler/softmax_7_1_6_7
/workspace/cu2til/cases/xpiler/softmax_8_10_64
/workspace/cu2til/cases/xpiler/softmax_12_3_128
/workspace/cu2til/cases/xpiler/softmax_45_25"

    # 21. 求和池化 (sumpool) - 8个案例
    ["sumpool"]="/workspace/cu2til/cases/xpiler/sumpool_1_5_5_64_3_3_2_2
/workspace/cu2til/cases/xpiler/sumpool_4_8_8_64_3_3_2_2
/workspace/cu2til/cases/xpiler/sumpool_4_35_35_192_5_5_2_2
/workspace/cu2til/cases/xpiler/sumpool_4_56_56_128_5_5_3_3
/workspace/cu2til/cases/xpiler/sumpool_5_32_32_64_5_5_3_3
/workspace/cu2til/cases/xpiler/sumpool_5_112_112_64_3_3_2_2
/workspace/cu2til/cases/xpiler/sumpool_16_64_64_64_5_5_1_1
/workspace/cu2til/cases/xpiler/sumpool_16_112_112_64_5_5_3_3"
)

# 统计信息
total_case_types=0
total_cases=0
passed_cases=0
failed_cases=0

echo "========================================"
echo "开始执行CUDA测试 (共${#case_type_list[@]}种case类型)"
echo "要测试的类型: ${case_type_list[*]}"
echo "========================================"

# 遍历指定的case类型列表
for case_type in "${case_type_list[@]}"; do
    # 检查该case类型是否在定义中存在
    if [[ ! -v case_types[$case_type] ]]; then
        echo "⚠️  警告: 未找到case类型 '$case_type' 的定义，跳过..."
        continue
    fi
    total_case_types=$((total_case_types + 1))
    echo ""
    echo "📁 正在测试 Case类型 #${total_case_types}: ${case_type} ($(echo "${case_type}" | tr '[:lower:]' '[:upper:]'))"
    echo "=========================================="
    
    case_count=0
    type_passed=0
    type_failed=0
    
    # 解析当前类型的所有case路径
    IFS=$'\n' read -d '' -ra case_folders <<< "${case_types[$case_type]}"
    
    # 遍历当前类型的每个具体case
    for folder in "${case_folders[@]}"; do
        case_count=$((case_count + 1))
        total_cases=$((total_cases + 1))
        
        echo "  🔄 [${case_count}/8] 测试案例: $(basename "$folder")"
        
        # 检查文件夹是否存在
        if [ -d "$folder" ]; then
            # 进入文件夹
            cd "$folder" || continue
            
            # 检查check_cuda.py是否存在
            if [ -f "check_cuda.py" ]; then
                # 运行测试并捕获退出状态
                # if python check_cuda.py > /dev/null 2>&1; then
                if python check_cuda.py 2>&1; then
                    echo "    ✅ 通过: $(basename "$folder")"
                    passed_cases=$((passed_cases + 1))
                    type_passed=$((type_passed + 1))
                else
                    echo "    ❌ 失败: $(basename "$folder")"
                    failed_cases=$((failed_cases + 1))
                    type_failed=$((type_failed + 1))
                fi
            else
                echo "    ⚠️  警告: 在 $folder 中没有找到 check_cuda.py"
                failed_cases=$((failed_cases + 1))
                type_failed=$((type_failed + 1))
            fi
            
            # 返回原来的目录
            cd - > /dev/null
        else
            echo "    ❌ 错误: 文件夹 $folder 不存在"
            failed_cases=$((failed_cases + 1))
            type_failed=$((type_failed + 1))
        fi
    done
    
    echo "  📊 ${case_type} 类型总结: 通过 ${type_passed}/${case_count}, 失败 ${type_failed}/${case_count}"
done

echo ""
echo "========================================"
echo "🎉 所有测试完成！最终统计:"
echo "========================================"
echo "📊 测试类型总数: ${total_case_types}/${#case_type_list[@]}"
echo "📊 测试案例总数: ${total_cases}"
echo "✅ 通过案例数量: ${passed_cases}"
echo "❌ 失败案例数量: ${failed_cases}"
if [ "$total_cases" -gt 0 ]; then
    echo "📈 成功率: $(( passed_cases * 100 / total_cases ))%"
else
    echo "📈 成功率: N/A (没有测试案例)"
fi
echo "========================================"
