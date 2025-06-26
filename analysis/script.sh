#!/bin/bash
echo "---------------no_err----------------"
echo "==01_single_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "'max_relative_error': 0.0, 'max_absolute_error': 0.0" {} \; # | wc -l
echo "==02_fused_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op/ -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "'max_relative_error': 0.0, 'max_absolute_error': 0.0" {} \; # | wc -l
echo "==03_network=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/03_network -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "'max_relative_error': 0.0, 'max_absolute_error': 0.0" {} \; # | wc -l

echo "---------------load_err----------------"
echo "==01_single_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "Failed to load CUDA extension:" {} \;
echo "==02_fused_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op/ -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "Failed to load CUDA extension:" {} \;
echo "==03_network=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/03_network -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "Failed to load CUDA extension:" {} \;

echo "---------------timeout----------------"
echo "==01_single_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "timeout" {} \;
echo "==02_fused_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "timeout" {} \;
echo "==03_network=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/03_network -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "timeout" {} \;

echo "---------------CUDA error----------------"
echo "==01_single_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "CUDA error" {} \;
echo "==02_fused_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "CUDA error" {} \;
echo "==03_network=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/03_network -name "eval.log" -path "*/logs/check_cuda/eval.log" -exec grep -l "CUDA error" {} \;


echo "---------------文件夹统计----------------"
echo "==01_single_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op -maxdepth 1 -type d | wc -l
echo "==02_fused_op=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op -maxdepth 1 -type d | wc -l
echo "==03_network=="
find /workspace/cu2tri/outputs/cu2tri/kernelbench_c/03_network -maxdepth 1 -type d | wc -l
