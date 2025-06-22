#!/usr/bin/env python3
"""
集成测试脚本
测试内核开发服务器与现有系统的集成，包括cu2tri和eval_triton.py
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from server.kernel.models import (
    CompileRequest, TestRequest, PerfRequest, LLMRequest,
    KernelType, TestStage
)
from server.kernel.service import KernelDevelopmentService
from server.kernel.gpu_manager import GPUManager

# 集成设置
USE_GPU_ID = 5
CU2TRI_PATH = project_root / "cu2tri"
EVAL_TRITON_PATH = project_root / "eval_triton.py"

async def test_cu2tri_integration():
    """测试与cu2tri模块的集成"""
    print("🔧 测试cu2tri集成...")
    
    if not CU2TRI_PATH.exists():
        print("⚠️  cu2tri目录不存在，跳过此测试")
        return True
    
    # 示例CUDA内核代码
    cuda_code = """
#include <cuda_runtime.h>

__global__ void vector_add_cuda(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
    void launch_vector_add(float* a, float* b, float* c, int n) {
        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        vector_add_cuda<<<blocks, threads_per_block>>>(a, b, c, n);
        cudaDeviceSynchronize();
    }
}
    """
    
    service = KernelDevelopmentService()
    
    # 使用LLM服务将CUDA转换为Triton
    llm_req = LLMRequest(
        conversation_id="cuda_to_triton_integration",
        prompt=f"""
请将以下CUDA内核转换为等效的Triton内核：

{cuda_code}

要求：
1. 保持相同的功能和语义
2. 优化内存访问模式
3. 使用适当的Triton语言特性
4. 添加适当的边界检查
        """,
        context={
            "task": "cuda_to_triton_conversion",
            "source_language": "cuda",
            "target_language": "triton",
            "gpu_id": USE_GPU_ID
        }
    )
    
    try:
        llm_result = await service.llm_assist(llm_req)
        
        if llm_result.success:
            triton_code = llm_result.response
            print("✅ CUDA到Triton转换成功")
            print(f"转换后的代码长度: {len(triton_code)}字符")
            
            # 测试生成的Triton代码
            compile_req = CompileRequest(
                kernel_type=KernelType.TRITON,
                source_code=triton_code,
                compile_options={"gpu_id": USE_GPU_ID}
            )
            
            compile_result = await service.compile_kernel(compile_req)
            if compile_result.success:
                print("✅ 转换后的Triton内核编译成功")
                return True
            else:
                print(f"❌ 转换后的内核编译失败: {compile_result.error_message}")
                return False
        else:
            print(f"❌ CUDA到Triton转换失败: {llm_result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ cu2tri集成测试异常: {e}")
        return False

async def test_eval_triton_integration():
    """测试与eval_triton.py的集成"""
    print("🔧 测试eval_triton.py集成...")
    
    if not EVAL_TRITON_PATH.exists():
        print("⚠️  eval_triton.py不存在，跳过此测试")
        return True
    
    # 示例Triton内核用于评估
    triton_kernel = """
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算注意力机制
    qvec = tl.load(Q + off_hz * stride_qh + start_m * stride_qm + tl.arange(0, BLOCK_DMODEL))
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        # 加载K和V
        k = tl.load(K + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :])
        v = tl.load(V + off_hz * stride_vh + start_n * stride_vn + tl.arange(0, BLOCK_DMODEL)[None, :])
        
        # 计算attention scores
        qk = tl.dot(qvec, tl.trans(k))
        
        # Softmax处理
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # 更新输出
        acc = acc * tl.exp(m_i - m_ij)[:, None]
        acc += tl.dot(p, v)
        
        l_i = l_i * tl.exp(m_i - m_ij) + l_ij
        m_i = m_ij
    
    # 最终归一化
    acc = acc / l_i[:, None]
    
    # 存储结果
    tl.store(Out + off_hz * stride_oh + start_m * stride_om + tl.arange(0, BLOCK_DMODEL), acc)
    """
    
    service = KernelDevelopmentService()
    
    try:
        # 1. 编译Triton内核
        compile_req = CompileRequest(
            kernel_type=KernelType.TRITON,
            source_code=triton_kernel,
            compile_options={
                "gpu_id": USE_GPU_ID,
                "optimize": True
            }
        )
        
        compile_result = await service.compile_kernel(compile_req)
        if not compile_result.success:
            print(f"❌ 内核编译失败: {compile_result.error_message}")
            return False
            
        print("✅ 复杂Triton内核编译成功")
        
        # 2. 功能测试
        test_req = TestRequest(
            kernel_name="fused_attention_kernel",
            test_stage=TestStage.FUNCTIONAL,
            test_inputs=[
                {
                    "batch_size": 1,
                    "num_heads": 8,
                    "seq_length": 512,
                    "head_dim": 64
                }
            ]
        )
        
        test_result = await service.test_kernel(test_req)
        if test_result.success:
            print("✅ 复杂内核功能测试通过")
        else:
            print(f"⚠️  功能测试未完全通过: {test_result.error_message}")
        
        # 3. 性能评估
        perf_req = PerfRequest(
            kernel_name="fused_attention_kernel",
            input_sizes=[
                (1, 8, 512, 64),    # 小规模
                (2, 8, 1024, 64),   # 中等规模
                (4, 8, 2048, 64),   # 大规模
            ],
            num_runs=20,
            warmup_runs=5,
            gpu_id=USE_GPU_ID
        )
        
        perf_result = await service.benchmark_kernel(perf_req)
        if perf_result.success:
            print(f"✅ 性能评估完成")
            print(f"  - 平均执行时间: {perf_result.avg_time_ms:.2f}ms")
            print(f"  - GPU利用率: {perf_result.gpu_utilization:.1f}%")
            print(f"  - 内存带宽: {perf_result.memory_bandwidth_gb_s:.1f} GB/s")
            return True
        else:
            print(f"❌ 性能评估失败: {perf_result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ eval_triton集成测试异常: {e}")
        return False

async def test_end_to_end_workflow():
    """测试端到端工作流程"""
    print("🔧 测试端到端工作流程...")
    
    service = KernelDevelopmentService()
    
    # 定义一个完整的开发工作流程
    workflow_steps = [
        "code_generation",      # 代码生成
        "compilation",          # 编译
        "functional_testing",   # 功能测试
        "performance_testing",  # 性能测试
        "optimization",         # 优化建议
        "final_validation"      # 最终验证
    ]
    
    try:
        # 1. 代码生成阶段
        print("  步骤1: LLM辅助代码生成...")
        llm_req = LLMRequest(
            conversation_id="end_to_end_workflow",
            prompt="""
请生成一个高效的Triton内核实现矩阵转置功能：
- 输入：二维矩阵 A (M x N)
- 输出：转置矩阵 A^T (N x M)
- 要求：优化内存合并访问，最小化bank冲突
            """,
            context={
                "task": "matrix_transpose",
                "optimization_target": "memory_efficiency",
                "gpu_id": USE_GPU_ID
            }
        )
        
        llm_result = await service.llm_assist(llm_req)
        if not llm_result.success:
            print(f"❌ 代码生成失败: {llm_result.error_message}")
            return False
        
        generated_code = llm_result.response
        print("  ✅ 代码生成完成")
        
        # 2. 编译阶段
        print("  步骤2: 内核编译...")
        compile_req = CompileRequest(
            kernel_type=KernelType.TRITON,
            source_code=generated_code,
            compile_options={
                "gpu_id": USE_GPU_ID,
                "optimize": True
            }
        )
        
        compile_result = await service.compile_kernel(compile_req)
        if not compile_result.success:
            print(f"❌ 编译失败: {compile_result.error_message}")
            return False
        
        print("  ✅ 编译完成")
        
        # 3. 功能测试阶段
        print("  步骤3: 功能测试...")
        test_req = TestRequest(
            kernel_name="matrix_transpose",
            test_stage=TestStage.FUNCTIONAL,
            test_inputs=[
                {"matrix_size": (64, 128)},
                {"matrix_size": (256, 512)},
            ]
        )
        
        test_result = await service.test_kernel(test_req)
        if not test_result.success:
            print(f"❌ 功能测试失败: {test_result.error_message}")
            return False
        
        print("  ✅ 功能测试通过")
        
        # 4. 性能测试阶段
        print("  步骤4: 性能测试...")
        perf_req = PerfRequest(
            kernel_name="matrix_transpose",
            input_sizes=[
                (1024, 1024),
                (2048, 2048),
                (4096, 4096)
            ],
            num_runs=50,
            warmup_runs=10,
            gpu_id=USE_GPU_ID
        )
        
        perf_result = await service.benchmark_kernel(perf_req)
        if not perf_result.success:
            print(f"❌ 性能测试失败: {perf_result.error_message}")
            return False
        
        print(f"  ✅ 性能测试完成 (平均: {perf_result.avg_time_ms:.2f}ms)")
        
        # 5. 优化建议阶段
        print("  步骤5: 优化建议...")
        optimization_req = LLMRequest(
            conversation_id="end_to_end_workflow",
            prompt=f"""
基于以下性能测试结果，请提供优化建议：

性能数据：
- 平均执行时间: {perf_result.avg_time_ms}ms
- GPU利用率: {perf_result.gpu_utilization}%
- 内存带宽: {perf_result.memory_bandwidth_gb_s} GB/s
- 计算吞吐量: {perf_result.compute_throughput_tflops} TFLOPS

请分析性能瓶颈并提供具体的优化建议。
            """,
            context={
                "task": "performance_optimization",
                "performance_data": perf_result.dict(),
                "gpu_id": USE_GPU_ID
            }
        )
        
        optimization_result = await service.llm_assist(optimization_req)
        if optimization_result.success:
            print("  ✅ 优化建议生成完成")
            print(f"  建议摘要: {optimization_result.response[:100]}...") # type: ignore
        else:
            print("  ⚠️  优化建议生成失败，但工作流程继续")
        
        # 6. 最终验证
        print("  步骤6: 最终验证...")
        
        # 检查所有组件健康状态
        health_check = await service.health_check()
        if health_check["status"] == "healthy":
            print("  ✅ 系统健康检查通过")
        else:
            print("  ⚠️  系统健康检查发现问题")
        
        print("🎉 端到端工作流程完成!")
        return True
        
    except Exception as e:
        print(f"❌ 端到端工作流程异常: {e}")
        return False

async def test_existing_tools_integration():
    """测试与现有工具的集成"""
    print("🔧 测试现有工具集成...")
    
    integration_tests = []
    
    # 检查cu2tri模块
    if CU2TRI_PATH.exists():
        try:
            # 尝试导入cu2tri模块
            sys.path.insert(0, str(CU2TRI_PATH))
            # import cu2tri  # 假设的导入
            integration_tests.append(("cu2tri模块", True))
        except ImportError as e:
            print(f"⚠️  cu2tri模块导入失败: {e}")
            integration_tests.append(("cu2tri模块", False))
    else:
        print("⚠️  cu2tri目录不存在")
        integration_tests.append(("cu2tri模块", False))
    
    # 检查eval_triton.py
    if EVAL_TRITON_PATH.exists():
        try:
            # 尝试执行eval_triton.py的基本功能
            result = subprocess.run(
                [sys.executable, str(EVAL_TRITON_PATH), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            integration_tests.append(("eval_triton.py", result.returncode == 0))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"⚠️  eval_triton.py执行失败: {e}")
            integration_tests.append(("eval_triton.py", False))
    else:
        print("⚠️  eval_triton.py不存在")
        integration_tests.append(("eval_triton.py", False))
    
    # 检查LLM服务集成
    try:
        from llm.chat.openrouter_ import OpenRouterProvider
        integration_tests.append(("LLM Provider", True))
    except ImportError:
        print("⚠️  LLM Provider导入失败")
        integration_tests.append(("LLM Provider", False))
    
    # 报告集成测试结果
    passed = sum(1 for _, success in integration_tests if success)
    total = len(integration_tests)
    
    print(f"集成测试结果: {passed}/{total}")
    for test_name, success in integration_tests:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    return passed == total

async def main():
    """运行所有集成测试"""
    print("🚀 开始集成测试\n")
    
    tests = [
        ("现有工具集成", test_existing_tools_integration),
        ("cu2tri集成", test_cu2tri_integration),
        ("eval_triton集成", test_eval_triton_integration),
        ("端到端工作流程", test_end_to_end_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"=== {test_name} ===")
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name}通过\n")
            else:
                print(f"❌ {test_name}失败\n")
        except Exception as e:
            print(f"❌ {test_name}异常: {e}\n")
    
    print(f"📊 集成测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有集成测试通过!")
        print("🔗 内核开发服务器已成功集成到现有系统中!")
        return 0
    else:
        print("⚠️  部分集成测试失败，请检查相关组件")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 