#!/usr/bin/env python3
"""
多GPU矩阵乘法任务示例

这个脚本展示了如何使用新添加的多GPU matmul任务函数来同时在所有可用GPU上执行计算。
"""

import asyncio
import logging
import json
from server.xpu.nvgpu.examples.example_tasks import concurrent_gpu_benchmark

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_concurrent_gpu_benchmark():
    """运行并发GPU基准测试"""
    print("=" * 60)
    print("运行并发GPU基准测试（不同矩阵大小）")
    print("=" * 60)
    
    # 配置参数
    matrix_sizes = [512, 1024, 2048]  # 不同的矩阵大小
    num_iterations = 3  # 每个GPU执行的迭代次数
    
    print(f"测试矩阵大小: {matrix_sizes}")
    print(f"每个大小的迭代次数: {num_iterations}")
    print()
    
    # 执行基准测试
    result = await concurrent_gpu_benchmark(matrix_sizes=matrix_sizes, num_iterations=num_iterations)
    
    # 打印结果
    if result["status"] == "success":
        print(f"✅ 基准测试成功完成!")
        print(f"总GPU数量: {result['total_gpus']}")
        print()
        
        # 为每个矩阵大小显示结果
        for size_result in result["detailed_results"]:
            matrix_size = size_result["matrix_size"]
            print(f"矩阵大小 {matrix_size}x{matrix_size} 的结果:")
            print("-" * 40)
            
            for gpu_result in size_result["gpu_results"]:
                gpu_data = gpu_result["result"]
                if gpu_data and gpu_data.get("status") == "success":
                    print(f"  GPU {gpu_data['gpu_id']}: {gpu_data['estimated_tflops']:.2f} TFLOPs, "
                          f"平均时间: {gpu_data['avg_computation_time']:.4f}s")
                else:
                    error_msg = gpu_data.get("error", "Unknown error") if gpu_data else "No result returned"
                    print(f"  GPU {gpu_result['gpu_id']}: 失败 - {error_msg}")
            print()
    else:
        print(f"❌ 基准测试失败: {result['error']}")
    
    return result


def save_results_to_file(results, filename="multi_gpu_results.json"):
    """将结果保存到JSON文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到 {filename}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


async def main():
    """主函数"""
    print("🚀 多GPU矩阵乘法任务示例")
    print("=" * 60)
    
    try:
        print("\n" + "="*60 + "\n")
        
        # 2. 运行并发基准测试
        benchmark_result = await run_concurrent_gpu_benchmark()
        
        print("\n🎉 所有任务完成!")
        
    except Exception as e:
        logger.error(f"运行示例时发生错误: {e}")
        print(f"❌ 运行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 