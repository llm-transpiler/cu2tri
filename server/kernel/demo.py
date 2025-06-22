#!/usr/bin/env python3
"""
内核开发服务器演示脚本
展示如何使用各种功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from server.kernel.models import (
    CompileRequest, TestRequest, PerfRequest, LLMRequest,
    KernelType, TestStage
)
from server.kernel.service import KernelDevelopmentService

async def demo_triton_kernel_development():
    """演示完整的Triton内核开发流程"""
    print("🚀 演示: 完整的Triton内核开发流程")
    print("="*60)
    
    service = KernelDevelopmentService()
    
    # 步骤1: 使用LLM生成内核代码
    print("\n📝 步骤1: LLM辅助代码生成")
    llm_req = LLMRequest(
        conversation_id="demo_development",
        prompt="""
请生成一个高效的Triton内核来实现向量元素求平方功能：
- 输入：浮点数向量 x
- 输出：向量 y，其中 y[i] = x[i] * x[i]
- 要求：优化内存访问，处理任意长度的向量
        """,
        context={
            "task": "vector_square",
            "optimization_target": "performance"
        }
    )
    
    try:
        llm_result = await service.llm_assist(llm_req)
        if llm_result.success:
            generated_code = llm_result.response
            print("✅ LLM代码生成成功")
            print(f"生成的代码长度: {len(generated_code)} 字符")
        else:
            print("❌ LLM代码生成失败，使用预定义代码")
            generated_code = """
import triton
import triton.language as tl

@triton.jit
def vector_square_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    tl.store(y_ptr + offsets, y, mask=mask)
            """
    except Exception as e:
        print(f"LLM服务异常: {e}")
        generated_code = """
import triton
import triton.language as tl

@triton.jit
def vector_square_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    tl.store(y_ptr + offsets, y, mask=mask)
        """
    
    # 步骤2: 编译内核
    print("\n🔧 步骤2: 编译内核")
    compile_req = CompileRequest(
        kernel_type=KernelType.TRITON,
        source_code=generated_code,
        compile_options={
            "optimize": True,
            "gpu_id": 5  # 使用GPU 5
        }
    )
    
    compile_result = await service.compile_kernel(compile_req)
    if compile_result.success:
        print("✅ 内核编译成功")
        print(f"编译时间: {compile_result.compile_time_ms}ms")
    else:
        print(f"❌ 内核编译失败: {compile_result.error_message}")
        return False
    
    # 步骤3: 功能测试
    print("\n🧪 步骤3: 功能测试")
    test_req = TestRequest(
        kernel_name="vector_square_kernel",
        test_stage=TestStage.FUNCTIONAL,
        test_inputs=[
            {"vector": [1.0, 2.0, 3.0, 4.0]},
            {"vector": [0.5, 1.5, 2.5]},
        ],
        expected_outputs=[
            [1.0, 4.0, 9.0, 16.0],
            [0.25, 2.25, 6.25]
        ]
    )
    
    test_result = await service.test_kernel(test_req)
    if test_result.success:
        print("✅ 功能测试通过")
        print(f"测试用例通过: {test_result.passed_tests}/{test_result.total_tests}")
    else:
        print(f"❌ 功能测试失败: {test_result.error_message}")
    
    # 步骤4: 性能测试
    print("\n⚡ 步骤4: 性能测试")
    perf_req = PerfRequest(
        kernel_name="vector_square_kernel",
        input_sizes=[
            (1024,),
            (4096,),
            (16384,),
            (65536,)
        ],
        num_runs=100,
        warmup_runs=10,
        gpu_id=5
    )
    
    perf_result = await service.benchmark_kernel(perf_req)
    if perf_result.success:
        print("✅ 性能测试完成")
        print(f"平均执行时间: {perf_result.avg_time_ms:.3f}ms")
        print(f"最佳执行时间: {perf_result.min_time_ms:.3f}ms")
        print(f"GPU利用率: {perf_result.gpu_utilization:.1f}%")
        print(f"内存带宽: {perf_result.memory_bandwidth_gb_s:.1f} GB/s")
    else:
        print(f"❌ 性能测试失败: {perf_result.error_message}")
    
    # 步骤5: 优化建议
    print("\n💡 步骤5: 获取优化建议")
    if perf_result.success:
        optimization_req = LLMRequest(
            conversation_id="demo_development",
            prompt=f"""
基于以下性能测试结果，请提供优化建议：

性能数据:
- 平均执行时间: {perf_result.avg_time_ms:.3f}ms
- GPU利用率: {perf_result.gpu_utilization:.1f}%
- 内存带宽: {perf_result.memory_bandwidth_gb_s:.1f} GB/s

请分析性能瓶颈并提供具体优化建议。
            """,
            context={
                "task": "performance_optimization",
                "kernel_type": "triton",
                "performance_data": perf_result.dict()
            }
        )
        
        try:
            optimization_result = await service.llm_assist(optimization_req)
            if optimization_result.success:
                print("✅ 优化建议生成完成")
                print("建议内容:")
                print(optimization_result.response[:300] + "..." if len(optimization_result.response) > 300 else optimization_result.response)
            else:
                print(f"❌ 优化建议生成失败: {optimization_result.error_message}")
        except Exception as e:
            print(f"优化建议服务异常: {e}")
    
    print("\n🎉 演示完成! 内核开发流程展示结束。")
    return True

async def demo_cuda_to_triton_conversion():
    """演示CUDA到Triton的转换"""
    print("\n🔄 演示: CUDA到Triton转换")
    print("="*60)
    
    service = KernelDevelopmentService()
    
    # CUDA内核代码
    cuda_code = """
__global__ void matrix_add_cuda(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}
    """
    
    print("原始CUDA内核:")
    print(cuda_code)
    
    # 使用LLM转换
    conversion_req = LLMRequest(
        conversation_id="cuda_to_triton_demo",
        prompt=f"""
请将以下CUDA内核转换为等效的Triton内核：

{cuda_code}

要求：
1. 保持相同的功能
2. 优化内存访问模式
3. 使用Triton的最佳实践
4. 添加适当的边界检查
        """,
        context={
            "task": "cuda_to_triton_conversion",
            "source_language": "cuda",
            "target_language": "triton"
        }
    )
    
    try:
        conversion_result = await service.llm_assist(conversion_req)
        if conversion_result.success:
            print("✅ CUDA到Triton转换成功")
            print("\n转换后的Triton内核:")
            print(conversion_result.response)
            
            # 尝试编译转换后的代码
            compile_req = CompileRequest(
                kernel_type=KernelType.TRITON,
                source_code=conversion_result.response,
                compile_options={"gpu_id": 5}
            )
            
            compile_result = await service.compile_kernel(compile_req)
            if compile_result.success:
                print("✅ 转换后的内核编译成功")
            else:
                print(f"⚠️  转换后的内核编译失败: {compile_result.error_message}")
        else:
            print(f"❌ CUDA到Triton转换失败: {conversion_result.error_message}")
    except Exception as e:
        print(f"转换服务异常: {e}")

async def demo_gpu_resource_management():
    """演示GPU资源管理"""
    print("\n🎮 演示: GPU资源管理")
    print("="*60)
    
    service = KernelDevelopmentService()
    gpu_manager = service.gpu_manager
    
    # 获取GPU状态
    print("当前GPU状态:")
    gpu_stats = await gpu_manager.get_gpu_stats()
    for gpu_id, stats in gpu_stats.items():
        print(f"  GPU {gpu_id}: {stats.get('name', 'Unknown')}")
        print(f"    内存: {stats.get('memory_used', 0)}MB / {stats.get('memory_total', 0)}MB")
        print(f"    利用率: {stats.get('utilization', 0)}%")
        print(f"    温度: {stats.get('temperature', 0)}°C")
    
    # 演示GPU分配
    print("\n演示GPU分配:")
    try:
        # 分配开发环境GPU
        dev_gpu = await gpu_manager.allocate_gpu(stage='development')
        if dev_gpu is not None:
            print(f"✅ 分配开发GPU: {dev_gpu}")
            
            # 等待一下
            await asyncio.sleep(1)
            
            # 释放GPU
            await gpu_manager.release_gpu(dev_gpu)
            print(f"✅ 释放开发GPU: {dev_gpu}")
        else:
            print("❌ 开发GPU分配失败")
        
        # 分配生产环境GPU (包括GPU 5)
        prod_gpu = await gpu_manager.allocate_gpu(stage='production', preferred_gpu_id=5)
        if prod_gpu is not None:
            print(f"✅ 分配生产GPU: {prod_gpu}")
            
            # 等待一下
            await asyncio.sleep(1)
            
            # 释放GPU
            await gpu_manager.release_gpu(prod_gpu)
            print(f"✅ 释放生产GPU: {prod_gpu}")
        else:
            print("❌ 生产GPU分配失败")
            
    except Exception as e:
        print(f"GPU管理演示异常: {e}")

async def main():
    """主演示函数"""
    print("🎬 内核开发服务器功能演示")
    print("="*80)
    
    demos = [
        ("Triton内核开发流程", demo_triton_kernel_development),
        ("CUDA到Triton转换", demo_cuda_to_triton_conversion),
        ("GPU资源管理", demo_gpu_resource_management)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ 演示 '{demo_name}' 异常: {e}")
        
        print("\n" + "-"*80)
    
    print("\n🎉 所有演示完成!")
    print("\n📚 更多功能请参考:")
    print("  - README.md: 详细文档")
    print("  - tests/: 测试用例")
    print("  - API文档: http://localhost:8000/docs (启动服务器后)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示异常: {e}")
        sys.exit(1) 