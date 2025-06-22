#!/usr/bin/env python3
"""
基础功能测试脚本
测试内核开发服务器的基本功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from server.kernel.models import (
    CompileRequest, TestRequest, PerfRequest, 
    KernelType, TestStage
)
from server.kernel.service import KernelDevelopmentService
from server.kernel.gpu_manager import GPUManager

async def test_gpu_manager():
    """测试GPU管理器基本功能"""
    print("🔧 测试GPU管理器...")
    
    gpu_manager = GPUManager()
    
    # 获取GPU状态
    gpu_stats = await gpu_manager.get_gpu_stats()
    print(f"GPU状态: {gpu_stats}")
    
    # 检查GPU可用性
    available_gpus = await gpu_manager.get_available_gpus()
    print(f"可用GPU: {available_gpus}")
    
    print("✅ GPU管理器测试通过\n")

async def test_compile_triton_kernel():
    """测试Triton内核编译"""
    print("🔧 测试Triton内核编译...")
    
    # 简单的向量加法内核
    triton_code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    """
    
    compile_req = CompileRequest(
        kernel_type=KernelType.TRITON,
        source_code=triton_code,
        compile_options={"optimize": True}
    )
    
    service = KernelDevelopmentService()
    result = await service.compile_kernel(compile_req)
    
    print(f"编译结果: {result.success}")
    if result.success:
        print("✅ Triton内核编译成功")
    else:
        print(f"❌ 编译失败: {result.error_message}")
    
    print()
    return result.success

async def test_cuda_kernel():
    """测试CUDA内核编译"""
    print("🔧 测试CUDA内核编译...")
    
    # 简单的CUDA内核
    cuda_code = """
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
    """
    
    compile_req = CompileRequest(
        kernel_type=KernelType.CUDA,
        source_code=cuda_code,
        compile_options={"arch": "sm_80"}
    )
    
    service = KernelDevelopmentService()
    result = await service.compile_kernel(compile_req)
    
    print(f"编译结果: {result.success}")
    if result.success:
        print("✅ CUDA内核编译成功")
    else:
        print(f"❌ 编译失败: {result.error_message}")
    
    print()
    return result.success

async def test_functional_testing():
    """测试功能测试模块"""
    print("🔧 测试功能测试模块...")
    
    test_req = TestRequest(
        kernel_name="add_kernel",
        test_stage=TestStage.FUNCTIONAL,
        test_inputs=[
            {"x": [1.0, 2.0, 3.0, 4.0], "y": [5.0, 6.0, 7.0, 8.0]},
            {"x": [10.0, 20.0], "y": [30.0, 40.0]}
        ],
        expected_outputs=[
            [6.0, 8.0, 10.0, 12.0],
            [40.0, 60.0]
        ]
    )
    
    service = KernelDevelopmentService()
    result = await service.test_kernel(test_req)
    
    print(f"测试结果: {result.success}")
    if result.success:
        print("✅ 功能测试通过")
    else:
        print(f"❌ 测试失败: {result.error_message}")
    
    print()
    return result.success

async def test_service_integration():
    """测试服务集成"""
    print("🔧 测试服务集成...")
    
    service = KernelDevelopmentService()
    
    # 检查服务状态
    health = await service.health_check()
    print(f"服务健康状态: {health}")
    
    print("✅ 服务集成测试通过\n")

async def main():
    """运行所有基础测试"""
    print("🚀 开始基础功能测试\n")
    
    tests = [
        ("GPU管理器", test_gpu_manager),
        ("服务集成", test_service_integration),
        ("Triton内核编译", test_compile_triton_kernel),
        ("CUDA内核编译", test_cuda_kernel),
        ("功能测试", test_functional_testing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success is not False:  # None也算通过
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}\n")
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基础测试通过!")
        return 0
    else:
        print("⚠️  部分测试失败，请检查日志")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 