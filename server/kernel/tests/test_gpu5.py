#!/usr/bin/env python3
"""
GPU 5 专项测试脚本
专门使用GPU ID 5进行性能和功能测试
"""

import asyncio
import sys
import os
import time
from pathlib import Path
import dotenv

dotenv.load_dotenv()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from server.kernel.models import (
    CompileRequest, TestRequest, PerfRequest, 
    KernelType, TestStage
)
from server.kernel.service import KernelDevelopmentService
from server.kernel.gpu_manager import GPUManager

# 强制使用GPU 5
TARGET_GPU_ID = 5

async def test_gpu5_availability():
    """测试GPU 5的可用性"""
    print(f"🔧 测试GPU {TARGET_GPU_ID}可用性...")
    
    gpu_manager = GPUManager()
    
    # 检查GPU 5状态
    gpu_stats = await gpu_manager.get_gpu_stats()
    if TARGET_GPU_ID in gpu_stats:
        gpu_info = gpu_stats[TARGET_GPU_ID] # type: ignore
        print(f"GPU {TARGET_GPU_ID} 状态:")
        print(f"  - 名称: {gpu_info.get('name', 'Unknown')}")
        print(f"  - 内存: {gpu_info.get('memory_used', 0)}MB / {gpu_info.get('memory_total', 0)}MB")
        print(f"  - 利用率: {gpu_info.get('utilization', 0)}%")
        print(f"  - 温度: {gpu_info.get('temperature', 0)}°C")
        
        # 检查是否空闲
        is_idle = gpu_info.get('utilization', 100) < 10
        print(f"  - 是否空闲: {'是' if is_idle else '否'}")
        
        if is_idle:
            print(f"✅ GPU {TARGET_GPU_ID} 可用且空闲")
            return True
        else:
            print(f"⚠️  GPU {TARGET_GPU_ID} 正在使用中")
            return False
    else:
        print(f"❌ GPU {TARGET_GPU_ID} 不存在或不可访问")
        return False

async def test_gpu5_allocation():
    """测试GPU 5的分配和释放"""
    print(f"🔧 测试GPU {TARGET_GPU_ID}分配...")
    
    gpu_manager = GPUManager()
    
    try:
        # 尝试分配GPU 5（生产环境）
        allocated_gpu = await gpu_manager.allocate_gpu(
            stage='production',
            preferred_gpu_id=TARGET_GPU_ID
        )
        
        if allocated_gpu == TARGET_GPU_ID:
            print(f"✅ 成功分配GPU {TARGET_GPU_ID}")
            
            # 等待一秒钟模拟使用
            await asyncio.sleep(1)
            
            # 释放GPU
            await gpu_manager.release_gpu(TARGET_GPU_ID)
            print(f"✅ 成功释放GPU {TARGET_GPU_ID}")
            return True
        else:
            print(f"⚠️  分配到GPU {allocated_gpu}，而非GPU {TARGET_GPU_ID}")
            if allocated_gpu is not None:
                await gpu_manager.release_gpu(allocated_gpu)
            return False
            
    except Exception as e:
        print(f"❌ GPU {TARGET_GPU_ID}分配失败: {e}")
        return False

async def test_triton_kernel_on_gpu5():
    """在GPU 5上测试Triton内核编译和执行"""
    print(f"🔧 在GPU {TARGET_GPU_ID}测试Triton内核...")
    
    # 矩阵乘法内核
    triton_code = """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    """
    
    compile_req = CompileRequest(
        kernel_type=KernelType.TRITON,
        source_code=triton_code,
        compile_options={
            "optimize": True,
            "gpu_id": TARGET_GPU_ID
        }
    )
    
    service = KernelDevelopmentService()
    
    # 设置使用特定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(TARGET_GPU_ID)
    
    try:
        result = await service.compile_kernel(compile_req)
        
        if result.success:
            print(f"✅ Triton内核在GPU {TARGET_GPU_ID}编译成功")
            print(f"编译时间: {result.compile_time_ms}ms")
            return True
        else:
            print(f"❌ 编译失败: {result.error_message}")
            return False
    except Exception as e:
        print(f"❌ 编译异常: {e}")
        return False
    finally:
        # 恢复环境变量
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

async def test_performance_on_gpu5():
    """在GPU 5上进行性能测试"""
    print(f"🔧 在GPU {TARGET_GPU_ID}进行性能测试...")
    
    perf_req = PerfRequest(
        kernel_name="matmul_kernel",
        input_sizes=[
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096)
        ],
        num_runs=10,
        warmup_runs=5,
        gpu_id=TARGET_GPU_ID
    )
    
    service = KernelDevelopmentService()
    
    # 设置使用特定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(TARGET_GPU_ID)
    
    try:
        result = await service.benchmark_kernel(perf_req)
        
        if result.success:
            print(f"✅ 性能测试在GPU {TARGET_GPU_ID}完成")
            print(f"平均执行时间: {result.avg_time_ms:.2f}ms")
            print(f"最佳执行时间: {result.min_time_ms:.2f}ms")
            print(f"GPU利用率: {result.gpu_utilization:.1f}%")
            print(f"内存带宽: {result.memory_bandwidth_gb_s:.1f} GB/s")
            print(f"计算吞吐量: {result.compute_throughput_tflops:.2f} TFLOPS")
            return True
        else:
            print(f"❌ 性能测试失败: {result.error_message}")
            return False
    except Exception as e:
        print(f"❌ 性能测试异常: {e}")
        return False
    finally:
        # 恢复环境变量
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

async def test_memory_stress_on_gpu5():
    """在GPU 5上进行内存压力测试"""
    print(f"🔧 在GPU {TARGET_GPU_ID}进行内存压力测试...")
    
    gpu_manager = GPUManager()
    
    try:
        # 分配GPU
        allocated_gpu = await gpu_manager.allocate_gpu(
            stage='production',
            preferred_gpu_id=TARGET_GPU_ID
        )
        
        if allocated_gpu != TARGET_GPU_ID:
            print(f"⚠️  未能分配到GPU {TARGET_GPU_ID}")
            if allocated_gpu is not None:
                await gpu_manager.release_gpu(allocated_gpu)
            return False
        
        # 监控内存使用
        initial_stats = await gpu_manager.get_gpu_stats()
        initial_memory = initial_stats[TARGET_GPU_ID]['memory_used'] # type: ignore
        
        print(f"初始内存使用: {initial_memory}MB")
        
        # 模拟内存密集型工作负载
        import torch
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{TARGET_GPU_ID}')
            
            # 分配大块内存
            tensors = []
            for i in range(5):
                size = 1024 * (i + 1)  # 逐渐增加大小
                tensor = torch.randn(size, size, device=device, dtype=torch.float32)
                tensors.append(tensor)
                
                # 检查内存使用
                current_stats = await gpu_manager.get_gpu_stats()
                current_memory = current_stats[TARGET_GPU_ID]['memory_used'] # type: ignore
                print(f"分配张量 {i+1}: {current_memory}MB (+{current_memory - initial_memory}MB)")
                
                await asyncio.sleep(0.5)  # 短暂等待
            
            # 清理内存
            del tensors
            torch.cuda.empty_cache()
            
            # 检查最终内存
            final_stats = await gpu_manager.get_gpu_stats()
            final_memory = final_stats[TARGET_GPU_ID]['memory_used'] # type: ignore
            print(f"清理后内存使用: {final_memory}MB")
            
            print(f"✅ GPU {TARGET_GPU_ID}内存压力测试完成")
            return True
        else:
            print("❌ CUDA不可用")
            return False
            
    except Exception as e:
        print(f"❌ 内存压力测试异常: {e}")
        return False
    finally:
        await gpu_manager.release_gpu(TARGET_GPU_ID)

async def main():
    """运行GPU 5专项测试"""
    print(f"🚀 开始GPU {TARGET_GPU_ID}专项测试\n")
    
    tests = [
        ("GPU可用性检查", test_gpu5_availability),
        ("GPU分配测试", test_gpu5_allocation),
        ("Triton内核测试", test_triton_kernel_on_gpu5),
        ("性能基准测试", test_performance_on_gpu5),
        ("内存压力测试", test_memory_stress_on_gpu5),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            start_time = time.time()
            success = await test_func()
            end_time = time.time()
            
            print(f"耗时: {end_time - start_time:.2f}秒")
            
            if success:
                passed += 1
                print(f"✅ {test_name}通过\n")
            else:
                print(f"❌ {test_name}失败\n")
                
        except Exception as e:
            print(f"❌ {test_name}异常: {e}\n")
    
    print(f"📊 GPU {TARGET_GPU_ID}测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print(f"🎉 GPU {TARGET_GPU_ID}所有测试通过!")
        return 0
    else:
        print(f"⚠️  GPU {TARGET_GPU_ID}部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 