#!/bin/bash


# ============================================================================
# 动态装载待测用例：从 llm_trans YAML/扫描获取目录清单
# - 可通过 CASE_TYPES="add relu" 仅测试指定算子
# - 可通过 FIRST_ONLY=1 仅取每类首个样例
# - PERF_FLAG 仍按原样支持（传给 check_cuda.py）
# ============================================================================

set -euo pipefail

TESTSET="${TESTSET:-xpiler}"
CASE_TYPES=${CASE_TYPES:-}
FIRST_ONLY=${FIRST_ONLY:-0}
# PERF_FLAG 默认允许为空，避免 set -u 报错
: "${PERF_FLAG:=}"

PYTHON_LISTER="python -m cu2til.llm_trans.utils.list_cases --testset ${TESTSET}"
if [[ -n "${CASE_TYPES}" ]]; then
  PYTHON_LISTER+=" --case-types ${CASE_TYPES}"
fi
if [[ "${FIRST_ONLY}" == "1" ]]; then
  PYTHON_LISTER+=" --first-only"
fi

declare -A case_types
eval "$(${PYTHON_LISTER} --format bash)"

case_type_list=($(printf "%s\n" "${!case_types[@]}" | sort))

# 统计信息
total_case_types=0
total_cases=0
passed_cases=0
failed_cases=0

echo "========================================"
echo "Starting CUDA tests (Total ${#case_type_list[@]} case types)"
echo "Case types to test: ${case_type_list[*]}"
echo "========================================"
echo "FIRST_ONLY mode: ${FIRST_ONLY} (1=first case only, 0=all)"

# 遍历指定的case类型列表
for case_type in "${case_type_list[@]}"; do
    # 检查该case类型是否在定义中存在
    if [[ ! -v case_types[$case_type] ]]; then
        echo "⚠️  WARNING: Case type '$case_type' not found, skipping..."
        continue
    fi
    total_case_types=$((total_case_types + 1))
    echo ""
    echo "📁 Testing Case type #${total_case_types}: ${case_type} ($(echo "${case_type}" | tr '[:lower:]' '[:upper:]'))"
    echo "=========================================="
    
    case_count=0
    type_passed=0
    type_failed=0
    
    # 解析当前类型的所有case路径（按换行拆分，多行 → 数组）
    mapfile -t case_folders <<< "${case_types[$case_type]}"
    total_cases_for_type=${#case_folders[@]}
    
    # 遍历当前类型的每个具体case
    for folder in "${case_folders[@]}"; do
        case_count=$((case_count + 1))
        total_cases=$((total_cases + 1))
        echo ""
        echo "----------------------------------------"
        echo ""
        echo "  🔄 [${case_count}/${total_cases_for_type}] Testing case: $(basename "$folder")"
        
        # 检查文件夹是否存在
        if [ -d "$folder" ]; then
            # 进入文件夹
            cd "$folder" || continue
            
            # 检查check_cuda.py是否存在
            if [ -f "check_cuda.py" ]; then
                # 运行测试并捕获退出状态
                # if python check_cuda.py > /dev/null 2>&1; then
                if python check_cuda.py ${PERF_FLAG:-} 2>&1; then
                    # echo "    ✅ PASSED: $(basename "$folder")"
                    passed_cases=$((passed_cases + 1))
                    type_passed=$((type_passed + 1))
                else
                    echo "    ❌ FAILED: $(basename "$folder")"
                    failed_cases=$((failed_cases + 1))
                    type_failed=$((type_failed + 1))
                fi
            else
                echo "    ⚠️  WARNING: No check_cuda.py found in $folder"
                failed_cases=$((failed_cases + 1))
                type_failed=$((type_failed + 1))
            fi
            
            # 返回原来的目录
            cd - > /dev/null
        else
            echo "    ❌ ERROR: Folder $folder does not exist"
            failed_cases=$((failed_cases + 1))
            type_failed=$((type_failed + 1))
        fi
    done
    
    echo "  📊 ${case_type} Summary: Passed ${type_passed}/${case_count}, Failed ${type_failed}/${case_count}"
done

echo ""
echo "========================================"
echo "🎉 All tests completed! Final statistics:"
echo "========================================"
echo "📊 Total case types tested: ${total_case_types}/${#case_type_list[@]}"
echo "📊 Total test cases: ${total_cases}"
echo "✅ Passed cases: ${passed_cases}"
echo "❌ Failed cases: ${failed_cases}"
if [ "$total_cases" -gt 0 ]; then
    echo "📈 Success rate: $(( passed_cases * 100 / total_cases ))%"
else
    echo "📈 Success rate: N/A (no test cases)"
fi
echo "========================================"

# CUDA_VISIBLE_DEVICES=7 PERF_FLAG='--no-perf' bash /data/apps/project/cu2tri/cu2til/cases/scripts/test_check_cuda.sh > expended_except_gqa_check.log 2>&1