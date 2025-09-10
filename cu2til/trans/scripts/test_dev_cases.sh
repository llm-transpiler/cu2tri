#!/bin/bash

# ============================================================================
# Dev Cases CUDA & Triton 测试脚本
# 用途: 遍历 dev_/ 目录下的所有case，依次运行 check_cuda.py 和 check_triton.py
# ============================================================================

# 基础目录设置
DEV_BASE_DIR="/workspace/cu2til/trans/dev_/gemini_2_5_pro/dev_"

# 控制选项 - 可以通过环境变量或参数控制
RUN_CUDA=${RUN_CUDA:-true}      # 是否运行CUDA测试
RUN_TRITON=${RUN_TRITON:-true}  # 是否运行Triton测试
PERF_FLAG=${PERF_FLAG:-"--no-perf"}  # 性能测试标志，默认禁用以加快速度
VERBOSE=${VERBOSE:-false}       # 是否显示详细输出

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-only)
            RUN_CUDA=true
            RUN_TRITON=false
            shift
            ;;
        --triton-only)
            RUN_CUDA=false
            RUN_TRITON=true
            shift
            ;;
        --with-perf)
            PERF_FLAG=""
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cuda-only      Only run CUDA tests"
            echo "  --triton-only    Only run Triton tests"
            echo "  --with-perf      Enable performance testing (slower)"
            echo "  --verbose, -v    Show detailed output"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 统计变量
total_directories=0
total_tests=0
passed_tests=0
failed_tests=0
cuda_passed=0
cuda_failed=0
triton_passed=0
triton_failed=0

# 开始时间
start_time=$(date +%s)

echo "========================================"
echo "🚀 Dev Cases Test Runner"
echo "========================================"
echo "📁 Base directory: $DEV_BASE_DIR"
echo "🔧 CUDA tests: $([ "$RUN_CUDA" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "🔧 Triton tests: $([ "$RUN_TRITON" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "⚡ Performance tests: $([ -z "$PERF_FLAG" ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "🕒 Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 检查基础目录是否存在
if [ ! -d "$DEV_BASE_DIR" ]; then
    echo "❌ ERROR: Base directory does not exist: $DEV_BASE_DIR"
    exit 1
fi

# 获取所有子目录并排序
mapfile -t directories < <(find "$DEV_BASE_DIR" -maxdepth 1 -type d -not -path "$DEV_BASE_DIR" | sort)

if [ ${#directories[@]} -eq 0 ]; then
    echo "❌ ERROR: No subdirectories found in $DEV_BASE_DIR"
    exit 1
fi

echo "📋 Found ${#directories[@]} directories to test"
echo ""

# 遍历每个目录
for dir in "${directories[@]}"; do
    dir_name=$(basename "$dir")
    total_directories=$((total_directories + 1))
    
    echo "📁 [$total_directories/${#directories[@]}] Testing: $dir_name"
    echo "----------------------------------------"
    
    # 进入目录
    if cd "$dir"; then
        dir_tests=0
        dir_passed=0
        dir_failed=0
        
        # 测试 CUDA
        if [ "$RUN_CUDA" = "true" ] && [ -f "check_cuda.py" ]; then
            dir_tests=$((dir_tests + 1))
            total_tests=$((total_tests + 1))
            echo "  🔄 Running CUDA test..."
            
            if [ "$VERBOSE" = "true" ]; then
                # 显示详细输出
                if python check_cuda.py $PERF_FLAG; then
                    echo "  ✅ CUDA: PASSED"
                    passed_tests=$((passed_tests + 1))
                    cuda_passed=$((cuda_passed + 1))
                    dir_passed=$((dir_passed + 1))
                else
                    echo "  ❌ CUDA: FAILED (exit code: $?)"
                    failed_tests=$((failed_tests + 1))
                    cuda_failed=$((cuda_failed + 1))
                    dir_failed=$((dir_failed + 1))
                fi
            else
                # 隐藏详细输出
                if python check_cuda.py $PERF_FLAG > /dev/null 2>&1; then
                    echo "  ✅ CUDA: PASSED"
                    passed_tests=$((passed_tests + 1))
                    cuda_passed=$((cuda_passed + 1))
                    dir_passed=$((dir_passed + 1))
                else
                    echo "  ❌ CUDA: FAILED (exit code: $?)"
                    failed_tests=$((failed_tests + 1))
                    cuda_failed=$((cuda_failed + 1))
                    dir_failed=$((dir_failed + 1))
                fi
            fi
        elif [ "$RUN_CUDA" = "true" ]; then
            echo "  ⚠️  CUDA: check_cuda.py not found"
        fi
        
        # 测试 Triton
        if [ "$RUN_TRITON" = "true" ] && [ -f "check_triton.py" ]; then
            dir_tests=$((dir_tests + 1))
            total_tests=$((total_tests + 1))
            echo "  🔄 Running Triton test..."
            
            if [ "$VERBOSE" = "true" ]; then
                # 显示详细输出
                if python check_triton.py $PERF_FLAG; then
                    echo "  ✅ Triton: PASSED"
                    passed_tests=$((passed_tests + 1))
                    triton_passed=$((triton_passed + 1))
                    dir_passed=$((dir_passed + 1))
                else
                    echo "  ❌ Triton: FAILED (exit code: $?)"
                    failed_tests=$((failed_tests + 1))
                    triton_failed=$((triton_failed + 1))
                    dir_failed=$((dir_failed + 1))
                fi
            else
                # 隐藏详细输出
                if python check_triton.py $PERF_FLAG > /dev/null 2>&1; then
                    echo "  ✅ Triton: PASSED"
                    passed_tests=$((passed_tests + 1))
                    triton_passed=$((triton_passed + 1))
                    dir_passed=$((dir_passed + 1))
                else
                    echo "  ❌ Triton: FAILED (exit code: $?)"
                    failed_tests=$((failed_tests + 1))
                    triton_failed=$((triton_failed + 1))
                    dir_failed=$((dir_failed + 1))
                fi
            fi
        elif [ "$RUN_TRITON" = "true" ]; then
            echo "  ⚠️  Triton: check_triton.py not found"
        fi
        
        # 目录总结
        if [ $dir_tests -gt 0 ]; then
            echo "  📊 $dir_name: Passed $dir_passed/$dir_tests, Failed $dir_failed/$dir_tests"
        else
            echo "  ⚠️  $dir_name: No test files found"
        fi
        
        # 返回基础目录
        cd "$DEV_BASE_DIR" || exit 1
    else
        echo "  ❌ ERROR: Cannot enter directory $dir"
        failed_tests=$((failed_tests + 1))
    fi
    
    echo ""
done

# 计算总时间
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

echo "========================================"
echo "🎉 All tests completed! Final Results:"
echo "========================================"
echo "📊 Directories tested: $total_directories"
echo "📊 Total tests run: $total_tests"
echo "✅ Total passed: $passed_tests"
echo "❌ Total failed: $failed_tests"

if [ $total_tests -gt 0 ]; then
    success_rate=$((passed_tests * 100 / total_tests))
    echo "📈 Overall success rate: $success_rate%"
else
    echo "📈 Overall success rate: N/A (no tests run)"
fi

echo ""
echo "🔧 Test type breakdown:"
if [ "$RUN_CUDA" = "true" ]; then
    cuda_total=$((cuda_passed + cuda_failed))
    if [ $cuda_total -gt 0 ]; then
        cuda_rate=$((cuda_passed * 100 / cuda_total))
        echo "  CUDA:   Passed $cuda_passed/$cuda_total ($cuda_rate%)"
    else
        echo "  CUDA:   No tests run"
    fi
fi

if [ "$RUN_TRITON" = "true" ]; then
    triton_total=$((triton_passed + triton_failed))
    if [ $triton_total -gt 0 ]; then
        triton_rate=$((triton_passed * 100 / triton_total))
        echo "  Triton: Passed $triton_passed/$triton_total ($triton_rate%)"
    else
        echo "  Triton: No tests run"
    fi
fi

echo ""
echo "⏱️  Total runtime: ${minutes}m ${seconds}s"
echo "🕒 End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 退出码：如果有失败的测试则返回1，否则返回0
if [ $failed_tests -gt 0 ]; then
    exit 1
else
    exit 0
fi
