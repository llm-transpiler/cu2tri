#!/usr/bin/env python3
"""
Level 1 CUDA代码批量编译示例
演示如何使用kernel server的compile queue批量编译所有Level 1的CUDA代码
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import dotenv
dotenv.load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from server.kernel.service import KernelService
from server.kernel.models import CompileRequest, TaskStatus, TaskType
from server.kernel.queue_manager import KernelQueueManager
from server.kernel.gpu_manager import GPUManager

class Level1CUDACompiler:
    """Level 1 CUDA代码批量编译器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 初始化服务
        self.kernel_service = None
        self.gpu_manager = None
        self.queue_manager = None
        
        # Level 1 CUDA目录
        self.level1_dir = Path("/workspace/monocases/cu2tri/outputs/level1")
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 编译统计
        self.compilation_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None,
            "detailed_results": []
        }
    
    async def initialize_services(self):
        """初始化服务"""
        try:
            self.logger.info("Initializing kernel services...")
            
            # 初始化GPU管理器
            self.gpu_manager = GPUManager()
            await self.gpu_manager.initialize()
            
            # 初始化队列管理器
            self.queue_manager = KernelQueueManager(self.gpu_manager)
            await self.queue_manager.start()
            
            # 初始化内核服务
            self.kernel_service = KernelService(
                gpu_manager=self.gpu_manager,
                queue_manager=self.queue_manager
            )
            
            self.logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def discover_cuda_files(self) -> List[Dict[str, Any]]:
        """发现所有Level 1 CUDA文件"""
        cuda_files = []
        
        if not self.level1_dir.exists():
            self.logger.error(f"Level 1 directory not found: {self.level1_dir}")
            return []
        
        self.logger.info(f"Scanning for CUDA files in: {self.level1_dir}")
        
        # 遍历所有Level 1子目录
        for problem_dir in self.level1_dir.iterdir():
            if not problem_dir.is_dir():
                continue
            
            problem_name = problem_dir.name
            self.logger.info(f"Scanning problem: {problem_name}")
            
            # 查找CUDA文件
            cuda_file_patterns = ["*.cu", "cuda_ref.cu", "*_cuda.cu"]
            
            for pattern in cuda_file_patterns:
                for cuda_file in problem_dir.glob(pattern):
                    try:
                        # 读取CUDA文件内容
                        with open(cuda_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        cuda_files.append({
                            "problem_name": problem_name,
                            "file_path": str(cuda_file),
                            "file_name": cuda_file.name,
                            "content": content,
                            "size": len(content),
                            "complexity": self._estimate_complexity(content)
                        })
                        
                        self.logger.info(f"  Found: {cuda_file.name} ({len(content)} chars)")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to read {cuda_file}: {e}")
        
        self.logger.info(f"✅ Discovered {len(cuda_files)} CUDA files")
        return cuda_files
    
    def _estimate_complexity(self, code: str) -> str:
        """估算代码复杂度"""
        lines = len(code.split('\n'))
        
        # 计算复杂度指标
        kernel_count = code.count('__global__')
        device_func_count = code.count('__device__')
        shared_mem_count = code.count('__shared__')
        sync_count = code.count('__syncthreads')
        atomic_count = len([line for line in code.split('\n') if 'atomic' in line.lower()])
        
        complexity_score = (
            lines * 0.1 +
            kernel_count * 10 +
            device_func_count * 5 +
            shared_mem_count * 3 +
            sync_count * 2 +
            atomic_count * 2
        )
        
        if complexity_score < 50:
            return "simple"
        elif complexity_score < 150:
            return "medium"
        else:
            return "complex"
    
    async def compile_cuda_files(self, cuda_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量编译CUDA文件"""
        self.compilation_stats["total_files"] = len(cuda_files)
        self.compilation_stats["start_time"] = time.time()
        
        self.logger.info(f"🚀 Starting batch compilation of {len(cuda_files)} CUDA files")
        
        # 按复杂度排序（简单的先编译）
        cuda_files.sort(key=lambda x: {"simple": 0, "medium": 1, "complex": 2}[x["complexity"]])
        
        # 分批处理
        batch_size = 5  # 同时处理5个文件
        results = []
        
        for i in range(0, len(cuda_files), batch_size):
            batch = cuda_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(cuda_files) + batch_size - 1) // batch_size
            
            self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            # 并行处理当前批次
            batch_tasks = []
            for cuda_file in batch:
                task = self._compile_single_file(cuda_file)
                batch_tasks.append(task)
            
            # 等待批次完成
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理批次结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch compilation error: {result}")
                    self.compilation_stats["failed"] += 1
                    results.append({
                        "file_info": batch[j],
                        "success": False,
                        "error": str(result),
                        "compilation_time": 0.0
                    })
                else:
                    results.append(result)
                    if result["success"]:
                        self.compilation_stats["successful"] += 1
                    else:
                        self.compilation_stats["failed"] += 1
            
            # 批次间短暂等待，避免过载
            if i + batch_size < len(cuda_files):
                await asyncio.sleep(1)
        
        self.compilation_stats["end_time"] = time.time()
        self.compilation_stats["detailed_results"] = results
        
        return self.compilation_stats
    
    async def _compile_single_file(self, cuda_file: Dict[str, Any]) -> Dict[str, Any]:
        """编译单个CUDA文件"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔧 Compiling: {cuda_file['problem_name']}/{cuda_file['file_name']}")
            
            # 创建编译请求
            compile_request = CompileRequest(
                request_id=f"compile_{cuda_file['problem_name']}_{int(time.time() * 1000)}",
                conversation_id="level1_batch_compile",
                source_code=cuda_file['content'],
                language="cuda",
                optimization_level="O2",
                target_arch="sm_80",  # 为H100优化
                additional_flags=[
                    "-std=c++17",
                    "-use_fast_math",
                    "--expt-relaxed-constexpr"
                ],
                metadata={
                    "problem_name": cuda_file['problem_name'],
                    "file_name": cuda_file['file_name'],
                    "complexity": cuda_file['complexity'],
                    "batch_compile": True
                }
            )
            
            # 提交编译任务
            response = await self.kernel_service.compile_kernel(compile_request)
            
            compilation_time = time.time() - start_time
            
            if response.status == TaskStatus.COMPLETED:
                self.logger.info(f"✅ Compiled successfully: {cuda_file['file_name']} ({compilation_time:.2f}s)")
                
                # 保存编译结果
                await self._save_compilation_result(cuda_file, response, compilation_time)
                
                return {
                    "file_info": cuda_file,
                    "success": True,
                    "compilation_time": compilation_time,
                    "binary_size": len(response.result.binary_code) if response.result and response.result.binary_code else 0,
                    "warnings": len(response.result.warnings) if response.result and response.result.warnings else 0,
                    "gpu_used": response.result.gpu_id if response.result else None
                }
            else:
                self.logger.error(f"❌ Compilation failed: {cuda_file['file_name']} - {response.result.error_message if response.result else 'Unknown error'}")
                
                return {
                    "file_info": cuda_file,
                    "success": False,
                    "compilation_time": compilation_time,
                    "error": response.result.error_message if response.result else "Unknown compilation error",
                    "error_details": response.result.error_details if response.result else None
                }
                
        except Exception as e:
            compilation_time = time.time() - start_time
            self.logger.error(f"💥 Exception during compilation of {cuda_file['file_name']}: {e}")
            
            return {
                "file_info": cuda_file,
                "success": False,
                "compilation_time": compilation_time,
                "error": str(e),
                "exception": True
            }
    
    async def _save_compilation_result(self, cuda_file: Dict[str, Any], response, compilation_time: float):
        """保存编译结果"""
        try:
            result_file = self.results_dir / f"{cuda_file['problem_name']}_compilation_result.json"
            
            result_data = {
                "problem_name": cuda_file['problem_name'],
                "file_name": cuda_file['file_name'],
                "compilation_time": compilation_time,
                "success": True,
                "binary_size": len(response.result.binary_code) if response.result.binary_code else 0,
                "warnings_count": len(response.result.warnings) if response.result.warnings else 0,
                "warnings": response.result.warnings if response.result.warnings else [],
                "gpu_used": response.result.gpu_id if response.result else None,
                "timestamp": time.time()
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save compilation result: {e}")
    
    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        stats = self.compilation_stats
        
        if stats["start_time"] and stats["end_time"]:
            total_time = stats["end_time"] - stats["start_time"]
        else:
            total_time = 0
        
        success_rate = (stats["successful"] / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
        
        report_lines = [
            "="*60,
            "🎯 Level 1 CUDA批量编译报告",
            "="*60,
            f"📊 总体统计:",
            f"  • 总文件数: {stats['total_files']}",
            f"  • 编译成功: {stats['successful']} ({success_rate:.1f}%)",
            f"  • 编译失败: {stats['failed']}",
            f"  • 跳过文件: {stats['skipped']}",
            f"  • 总耗时: {total_time:.2f}秒",
            f"  • 平均编译时间: {total_time/stats['total_files']:.2f}秒/文件" if stats['total_files'] > 0 else "  • 平均编译时间: N/A",
            ""
        ]
        
        # 按复杂度统计
        complexity_stats = {}
        for result in stats["detailed_results"]:
            complexity = result["file_info"]["complexity"]
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {"total": 0, "success": 0}
            complexity_stats[complexity]["total"] += 1
            if result["success"]:
                complexity_stats[complexity]["success"] += 1
        
        if complexity_stats:
            report_lines.extend([
                "📈 按复杂度统计:",
                ""
            ])
            for complexity, data in complexity_stats.items():
                success_rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
                report_lines.append(f"  • {complexity.title()}: {data['success']}/{data['total']} ({success_rate:.1f}%)")
        
        # 失败案例
        failed_cases = [r for r in stats["detailed_results"] if not r["success"]]
        if failed_cases:
            report_lines.extend([
                "",
                "❌ 失败案例:",
                ""
            ])
            for case in failed_cases[:10]:  # 只显示前10个
                file_info = case["file_info"]
                error = case.get("error", "Unknown error")[:100]
                report_lines.append(f"  • {file_info['problem_name']}/{file_info['file_name']}: {error}")
            
            if len(failed_cases) > 10:
                report_lines.append(f"  ... 还有 {len(failed_cases) - 10} 个失败案例")
        
        # 性能最佳案例
        successful_cases = [r for r in stats["detailed_results"] if r["success"]]
        if successful_cases:
            successful_cases.sort(key=lambda x: x["compilation_time"])
            report_lines.extend([
                "",
                "🏆 编译最快的案例:",
                ""
            ])
            for case in successful_cases[:5]:
                file_info = case["file_info"]
                time_taken = case["compilation_time"]
                report_lines.append(f"  • {file_info['problem_name']}/{file_info['file_name']}: {time_taken:.2f}s")
        
        report_lines.extend([
            "",
            "="*60,
            f"📁 详细结果保存在: {self.results_dir}",
            "="*60
        ])
        
        return "\n".join(report_lines)
    
    async def run_batch_compilation(self):
        """运行批量编译"""
        try:
            self.logger.info("🚀 Starting Level 1 CUDA batch compilation")
            
            # 1. 初始化服务
            await self.initialize_services()
            
            # 2. 发现CUDA文件
            cuda_files = await self.discover_cuda_files()
            
            if not cuda_files:
                self.logger.warning("No CUDA files found!")
                return
            
            # 3. 执行批量编译
            await self.compile_cuda_files(cuda_files)
            
            # 4. 生成报告
            report = self.generate_summary_report()
            print(report)
            
            # 5. 保存报告
            report_file = self.results_dir / f"batch_compilation_report_{int(time.time())}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"📊 Report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Batch compilation failed: {e}")
            raise
        finally:
            # 清理资源
            if self.queue_manager:
                await self.queue_manager.stop()
            if self.gpu_manager:
                await self.gpu_manager.cleanup()

# 使用示例和命令行接口
async def main():
    """主函数"""
    compiler = Level1CUDACompiler()
    await compiler.run_batch_compilation()

if __name__ == "__main__":
    print("🎯 Level 1 CUDA批量编译器")
    print("=" * 50)
    print("此示例将编译cu2tri/outputs/level1/目录下的所有CUDA文件")
    print("使用kernel server的compile queue进行并行编译")
    print("=" * 50)
    
    # 运行批量编译
    asyncio.run(main()) 