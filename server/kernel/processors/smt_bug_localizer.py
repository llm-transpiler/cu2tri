#!/usr/bin/env python3
"""
SMT Bug Localizer
基于SMT求解器的GPU内核bug定位和形式化验证系统
"""

import ast
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not available. Install with: pip install z3-solver")

class BugType(Enum):
    """Bug类型枚举"""
    MEMORY_OUT_OF_BOUNDS = "memory_out_of_bounds"
    DATA_RACE = "data_race"
    DEADLOCK = "deadlock"
    TYPE_MISMATCH = "type_mismatch"
    DIVISION_BY_ZERO = "division_by_zero"
    UNINITIALIZED_VARIABLE = "uninitialized_variable"
    BUFFER_OVERFLOW = "buffer_overflow"
    SHARED_MEMORY_BANK_CONFLICT = "shared_memory_bank_conflict"
    WARP_DIVERGENCE = "warp_divergence"
    ATOMIC_RACE = "atomic_race"

@dataclass
class BugLocation:
    """Bug位置信息"""
    file_name: str
    line_number: int
    column: int
    function_name: str
    bug_type: BugType
    severity: str  # "critical", "high", "medium", "low"
    description: str
    suggested_fix: str
    confidence: float  # 0.0 to 1.0

@dataclass
class SMTVerificationResult:
    """SMT验证结果"""
    is_safe: bool
    bugs_found: List[BugLocation]
    verification_time: float
    model_constraints: int
    solver_stats: Dict[str, Any]

class SMTBugLocalizer:
    """SMT bug定位器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.z3_available = Z3_AVAILABLE
        
        if not self.z3_available:
            self.logger.warning("Z3 solver not available. Some features will be disabled.")
        
        # 初始化SMT求解器
        if self.z3_available:
            self.solver = z3.Solver()
            self.solver.set("timeout", 30000)  # 30秒超时
    
    def analyze_kernel(self, source_code: str, kernel_type: str = "triton") -> SMTVerificationResult:
        """分析内核代码，查找bug"""
        start_time = self._get_time()
        bugs_found = []
        
        try:
            # 1. 语法分析和AST构建
            if kernel_type.lower() == "triton":
                bugs_found.extend(self._analyze_triton_kernel(source_code))
            elif kernel_type.lower() == "cuda":
                bugs_found.extend(self._analyze_cuda_kernel(source_code))
            else:
                self.logger.warning(f"Unsupported kernel type: {kernel_type}")
                
            # 2. 形式化验证（如果Z3可用）
            if self.z3_available:
                formal_bugs = self._formal_verification(source_code, kernel_type)
                bugs_found.extend(formal_bugs)
            
            # 3. 静态分析
            static_bugs = self._static_analysis(source_code, kernel_type)
            bugs_found.extend(static_bugs)
            
            # 4. 去重和排序
            bugs_found = self._deduplicate_bugs(bugs_found)
            bugs_found.sort(key=lambda x: (x.severity == "critical", x.confidence), reverse=True)
            
            verification_time = self._get_time() - start_time
            
            return SMTVerificationResult(
                is_safe=len([b for b in bugs_found if b.severity in ["critical", "high"]]) == 0,
                bugs_found=bugs_found,
                verification_time=verification_time,
                model_constraints=len(self.solver.assertions()) if self.z3_available else 0,
                solver_stats=self._get_solver_stats() if self.z3_available else {}
            )
            
        except Exception as e:
            self.logger.error(f"Kernel analysis failed: {e}")
            return SMTVerificationResult(
                is_safe=False,
                bugs_found=[],
                verification_time=self._get_time() - start_time,
                model_constraints=0,
                solver_stats={}
            )
    
    def _analyze_triton_kernel(self, source_code: str) -> List[BugLocation]:
        """分析Triton内核"""
        bugs = []
        lines = source_code.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # 检查边界检查
            if 'tl.load' in line and 'mask=' not in line:
                bugs.append(BugLocation(
                    file_name="kernel.py",
                    line_number=line_num,
                    column=line.find('tl.load'),
                    function_name=self._extract_function_name(lines, i),
                    bug_type=BugType.MEMORY_OUT_OF_BOUNDS,
                    severity="high",
                    description="tl.load without mask可能导致越界访问",
                    suggested_fix="添加mask参数: tl.load(ptr + offsets, mask=mask)",
                    confidence=0.9
                ))
            
            # 检查存储操作
            if 'tl.store' in line and 'mask=' not in line:
                bugs.append(BugLocation(
                    file_name="kernel.py",
                    line_number=line_num,
                    column=line.find('tl.store'),
                    function_name=self._extract_function_name(lines, i),
                    bug_type=BugType.MEMORY_OUT_OF_BOUNDS,
                    severity="high",
                    description="tl.store without mask可能导致越界写入",
                    suggested_fix="添加mask参数: tl.store(ptr + offsets, data, mask=mask)",
                    confidence=0.9
                ))
            
            # 检查块大小
            if 'BLOCK_SIZE' in line and ':' in line and 'tl.constexpr' not in line:
                bugs.append(BugLocation(
                    file_name="kernel.py",
                    line_number=line_num,
                    column=line.find('BLOCK_SIZE'),
                    function_name=self._extract_function_name(lines, i),
                    bug_type=BugType.TYPE_MISMATCH,
                    severity="medium",
                    description="BLOCK_SIZE应该声明为tl.constexpr",
                    suggested_fix="使用 BLOCK_SIZE: tl.constexpr",
                    confidence=0.8
                ))
            
            # 检查类型转换
            if re.search(r'tl\.(float|int)\d+', line):
                if not re.search(r'\.to\(tl\.(float|int)\d+\)', line):
                    bugs.append(BugLocation(
                        file_name="kernel.py",
                        line_number=line_num,
                        column=0,
                        function_name=self._extract_function_name(lines, i),
                        bug_type=BugType.TYPE_MISMATCH,
                        severity="medium",
                        description="可能存在隐式类型转换问题",
                        suggested_fix="使用显式类型转换 .to(tl.float32)",
                        confidence=0.6
                    ))
        
        return bugs
    
    def _analyze_cuda_kernel(self, source_code: str) -> List[BugLocation]:
        """分析CUDA内核"""
        bugs = []
        lines = source_code.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # 检查边界检查
            if 'threadIdx' in line or 'blockIdx' in line:
                if not any(check in line for check in ['if', '<', '>', '<=', '>=']):
                    # 检查下一几行是否有边界检查
                    has_bounds_check = False
                    for j in range(i+1, min(i+5, len(lines))):
                        if any(check in lines[j] for check in ['if', '<', '>', '<=', '>=']):
                            has_bounds_check = True
                            break
                    
                    if not has_bounds_check:
                        bugs.append(BugLocation(
                            file_name="kernel.cu",
                            line_number=line_num,
                            column=0,
                            function_name=self._extract_cuda_function_name(lines, i),
                            bug_type=BugType.MEMORY_OUT_OF_BOUNDS,
                            severity="high",
                            description="缺少边界检查，可能导致越界访问",
                            suggested_fix="添加边界检查: if (idx < N)",
                            confidence=0.7
                        ))
            
            # 检查共享内存声明
            if '__shared__' in line:
                if '[' in line and ']' in line:
                    # 检查是否使用了动态大小
                    shared_decl = line[line.find('__shared__'):line.find(';')]
                    if 'BLOCK_SIZE' in shared_decl or 'blockDim' in shared_decl:
                        bugs.append(BugLocation(
                            file_name="kernel.cu",
                            line_number=line_num,
                            column=line.find('__shared__'),
                            function_name=self._extract_cuda_function_name(lines, i),
                            bug_type=BugType.SHARED_MEMORY_BANK_CONFLICT,
                            severity="medium",
                            description="共享内存大小可能导致bank冲突",
                            suggested_fix="确保共享内存大小避免bank冲突",
                            confidence=0.6
                        ))
            
            # 检查同步原语
            if '__syncthreads()' in line:
                # 检查是否在条件分支中
                indent_level = len(line) - len(line.lstrip())
                for j in range(i-1, max(0, i-10), -1):
                    if 'if' in lines[j] and len(lines[j]) - len(lines[j].lstrip()) < indent_level:
                        bugs.append(BugLocation(
                            file_name="kernel.cu",
                            line_number=line_num,
                            column=line.find('__syncthreads'),
                            function_name=self._extract_cuda_function_name(lines, i),
                            bug_type=BugType.DEADLOCK,
                            severity="critical",
                            description="__syncthreads()在条件分支中可能导致死锁",
                            suggested_fix="确保所有线程都执行__syncthreads()",
                            confidence=0.8
                        ))
                        break
            
            # 检查原子操作
            if any(atomic in line for atomic in ['atomicAdd', 'atomicMax', 'atomicMin', 'atomicCAS']):
                bugs.append(BugLocation(
                    file_name="kernel.cu",
                    line_number=line_num,
                    column=0,
                    function_name=self._extract_cuda_function_name(lines, i),
                    bug_type=BugType.ATOMIC_RACE,
                    severity="medium",
                    description="原子操作可能存在竞争条件",
                    suggested_fix="检查原子操作的正确性和性能影响",
                    confidence=0.5
                ))
        
        return bugs
    
    def _formal_verification(self, source_code: str, kernel_type: str) -> List[BugLocation]:
        """形式化验证"""
        if not self.z3_available:
            return []
        
        bugs = []
        
        try:
            # 清空之前的约束
            self.solver.reset()
            
            # 创建符号变量
            if kernel_type.lower() == "triton":
                bugs.extend(self._verify_triton_properties(source_code))
            elif kernel_type.lower() == "cuda":
                bugs.extend(self._verify_cuda_properties(source_code))
            
        except Exception as e:
            self.logger.error(f"Formal verification failed: {e}")
        
        return bugs
    
    def _verify_triton_properties(self, source_code: str) -> List[BugLocation]:
        """验证Triton内核属性"""
        bugs = []
        
        # 提取关键信息
        block_size_match = re.search(r'BLOCK_SIZE.*?(\d+)', source_code)
        if block_size_match:
            block_size = int(block_size_match.group(1))
            
            # 创建符号变量
            pid = z3.Int('pid')
            n_elements = z3.Int('n_elements')
            
            # 添加约束
            self.solver.add(pid >= 0)
            self.solver.add(n_elements > 0)
            
            # 检查边界条件
            block_start = pid * block_size
            max_offset = block_start + block_size - 1
            
            # 验证是否可能越界
            self.solver.push()
            self.solver.add(max_offset >= n_elements)
            
            if self.solver.check() == z3.sat:
                bugs.append(BugLocation(
                    file_name="kernel.py",
                    line_number=1,
                    column=0,
                    function_name="triton_kernel",
                    bug_type=BugType.MEMORY_OUT_OF_BOUNDS,
                    severity="high",
                    description=f"在某些条件下可能发生越界访问 (BLOCK_SIZE={block_size})",
                    suggested_fix="添加适当的边界检查和mask",
                    confidence=0.85
                ))
            
            self.solver.pop()
        
        return bugs
    
    def _verify_cuda_properties(self, source_code: str) -> List[BugLocation]:
        """验证CUDA内核属性"""
        bugs = []
        
        # 提取线程配置信息
        thread_patterns = [
            r'blockDim\.x\s*\*\s*blockIdx\.x\s*\+\s*threadIdx\.x',
            r'threadIdx\.x',
            r'blockIdx\.x'
        ]
        
        if any(re.search(pattern, source_code) for pattern in thread_patterns):
            # 创建符号变量
            threadIdx_x = z3.Int('threadIdx_x')
            blockIdx_x = z3.Int('blockIdx_x')
            blockDim_x = z3.Int('blockDim_x')
            N = z3.Int('N')
            
            # 添加约束
            self.solver.add(threadIdx_x >= 0)
            self.solver.add(threadIdx_x < blockDim_x)
            self.solver.add(blockIdx_x >= 0)
            self.solver.add(blockDim_x > 0)
            self.solver.add(N > 0)
            
            # 计算全局索引
            global_idx = blockDim_x * blockIdx_x + threadIdx_x
            
            # 验证是否可能越界
            self.solver.push()
            self.solver.add(global_idx >= N)
            
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                bugs.append(BugLocation(
                    file_name="kernel.cu",
                    line_number=1,
                    column=0,
                    function_name="cuda_kernel",
                    bug_type=BugType.MEMORY_OUT_OF_BOUNDS,
                    severity="high",
                    description=f"全局索引可能越界: {model}",
                    suggested_fix="添加边界检查: if (idx < N)",
                    confidence=0.9
                ))
            
            self.solver.pop()
        
        return bugs
    
    def _static_analysis(self, source_code: str, kernel_type: str) -> List[BugLocation]:
        """静态分析"""
        bugs = []
        lines = source_code.split('\n')
        
        # 检查常见的编程错误
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # 检查除零错误
            if '/' in line and 'if' not in line:
                bugs.append(BugLocation(
                    file_name=f"kernel.{kernel_type}",
                    line_number=line_num,
                    column=line.find('/'),
                    function_name=self._extract_function_name(lines, i),
                    bug_type=BugType.DIVISION_BY_ZERO,
                    severity="medium",
                    description="可能存在除零错误",
                    suggested_fix="在除法前添加零检查",
                    confidence=0.4
                ))
            
            # 检查未初始化变量（简单检查）
            var_declarations = re.findall(r'(\w+)\s*[=;]', line)
            for var in var_declarations:
                if var not in ['int', 'float', 'double', 'if', 'for', 'while']:
                    # 检查变量是否在使用前初始化（简化版）
                    for j in range(i+1, min(i+10, len(lines))):
                        if var in lines[j] and '=' not in lines[j].split(var)[0]:
                            bugs.append(BugLocation(
                                file_name=f"kernel.{kernel_type}",
                                line_number=j+1,
                                column=lines[j].find(var),
                                function_name=self._extract_function_name(lines, j),
                                bug_type=BugType.UNINITIALIZED_VARIABLE,
                                severity="low",
                                description=f"变量 '{var}' 可能未初始化",
                                suggested_fix=f"在使用前初始化变量 '{var}'",
                                confidence=0.3
                            ))
                            break
        
        return bugs
    
    def _extract_function_name(self, lines: List[str], line_idx: int) -> str:
        """提取函数名（Triton）"""
        for i in range(line_idx, max(0, line_idx-20), -1):
            if 'def ' in lines[i]:
                match = re.search(r'def\s+(\w+)', lines[i])
                return match.group(1) if match else "unknown"
        return "unknown"
    
    def _extract_cuda_function_name(self, lines: List[str], line_idx: int) -> str:
        """提取函数名（CUDA）"""
        for i in range(line_idx, max(0, line_idx-20), -1):
            if '__global__' in lines[i] or '__device__' in lines[i]:
                # 查找下一行的函数定义
                for j in range(i, min(i+3, len(lines))):
                    match = re.search(r'(\w+)\s*\(', lines[j])
                    if match:
                        return match.group(1)
        return "unknown"
    
    def _deduplicate_bugs(self, bugs: List[BugLocation]) -> List[BugLocation]:
        """去重bug报告"""
        unique_bugs = []
        seen_bugs = set()
        
        for bug in bugs:
            bug_key = (bug.line_number, bug.bug_type, bug.description[:50])
            if bug_key not in seen_bugs:
                seen_bugs.add(bug_key)
                unique_bugs.append(bug)
        
        return unique_bugs
    
    def _get_time(self) -> float:
        """获取当前时间"""
        import time
        return time.time()
    
    def _get_solver_stats(self) -> Dict[str, Any]:
        """获取求解器统计信息"""
        if not self.z3_available:
            return {}
        
        stats = self.solver.statistics()
        return {
            'decisions': stats.get_key_value('decisions'),
            'conflicts': stats.get_key_value('conflicts'),
            'propagations': stats.get_key_value('propagations'),
            'memory': stats.get_key_value('memory')
        }
    
    def generate_bug_report(self, result: SMTVerificationResult) -> str:
        """生成bug报告"""
        if not result.bugs_found:
            return "✅ 未发现安全问题。内核代码通过所有检查。"
        
        report_parts = []
        report_parts.append(f"🔍 SMT验证报告 (耗时: {result.verification_time:.2f}s)")
        report_parts.append(f"安全状态: {'❌ 不安全' if not result.is_safe else '⚠️ 发现潜在问题'}")
        report_parts.append(f"发现 {len(result.bugs_found)} 个问题:")
        
        # 按严重程度分组
        critical_bugs = [b for b in result.bugs_found if b.severity == "critical"]
        high_bugs = [b for b in result.bugs_found if b.severity == "high"]
        medium_bugs = [b for b in result.bugs_found if b.severity == "medium"]
        low_bugs = [b for b in result.bugs_found if b.severity == "low"]
        
        if critical_bugs:
            report_parts.append("\n🚨 **严重问题:**")
            for bug in critical_bugs:
                report_parts.append(f"  行 {bug.line_number}: {bug.description}")
                report_parts.append(f"    建议: {bug.suggested_fix}")
                report_parts.append(f"    置信度: {bug.confidence:.1%}")
        
        if high_bugs:
            report_parts.append("\n⚠️  **高危问题:**")
            for bug in high_bugs:
                report_parts.append(f"  行 {bug.line_number}: {bug.description}")
                report_parts.append(f"    建议: {bug.suggested_fix}")
        
        if medium_bugs:
            report_parts.append("\n📝 **中等问题:**")
            for bug in medium_bugs[:3]:  # 限制显示数量
                report_parts.append(f"  行 {bug.line_number}: {bug.description}")
        
        if low_bugs:
            report_parts.append(f"\n💡 **其他建议:** {len(low_bugs)} 个优化建议")
        
        if result.solver_stats:
            report_parts.append(f"\n📊 求解器统计: {result.model_constraints} 个约束")
        
        return "\n".join(report_parts)

# 使用示例
if __name__ == "__main__":
    # 测试Triton内核
    triton_code = '''
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # 缺少mask
    y = tl.load(y_ptr + offsets)  # 缺少mask
    output = x + y
    tl.store(output_ptr + offsets, output)  # 缺少mask
    '''
    
    # 测试CUDA内核
    cuda_code = '''
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];  // 缺少边界检查
    __syncthreads();  // 可能在条件分支中
}
    '''
    
    localizer = SMTBugLocalizer()
    
    print("=== Triton内核分析 ===")
    triton_result = localizer.analyze_kernel(triton_code, "triton")
    print(localizer.generate_bug_report(triton_result))
    
    print("\n=== CUDA内核分析 ===")
    cuda_result = localizer.analyze_kernel(cuda_code, "cuda")
    print(localizer.generate_bug_report(cuda_result)) 