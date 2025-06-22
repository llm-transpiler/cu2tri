#!/usr/bin/env python3
"""
SMT (Satisfiability Modulo Theories) Processor
集成SMT Bug Localizer的形式化验证处理器
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import time
import traceback

from .base import BaseProcessor
from .smt_bug_localizer import SMTBugLocalizer, SMTVerificationResult, BugLocation
from ..models import SMTRequest, SMTResult

class SMTProcessor(BaseProcessor):
    """SMT处理器 - 集成bug定位的形式化验证"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 初始化SMT Bug Localizer
        try:
            self.bug_localizer = SMTBugLocalizer()
            self.logger.info("SMT Bug Localizer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SMT Bug Localizer: {e}")
            self.bug_localizer = None
        
    async def process(self, request: SMTRequest) -> SMTResult:
        """处理SMT验证请求"""
        try:
            self.logger.info(f"Processing SMT request: {request.property_type}")
            
            if not self.bug_localizer:
                return SMTResult(
                    success=False,
                    error_message="SMT Bug Localizer not available",
                    is_verified=False,
                    verification_time=0.0
                )
            
            # 执行形式化验证和bug定位
            start_time = time.time()
            
            # 分析内核代码
            verification_result = await self._analyze_with_smt(request)
            
            verification_time = time.time() - start_time
            
            # 生成详细报告
            bug_report = self.bug_localizer.generate_bug_report(verification_result)
            
            # 构建结果
            return SMTResult(
                success=True,
                is_verified=verification_result.is_safe,
                verification_time=verification_time,
                counterexample=self._extract_counterexamples(verification_result),
                proof_trace=bug_report,
                bugs_found=len(verification_result.bugs_found),
                critical_bugs=len([b for b in verification_result.bugs_found if b.severity == "critical"]),
                solver_stats=verification_result.solver_stats
            )
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"SMT processing failed: {error_msg}\n{error_traceback}")
            
            return SMTResult(
                success=False,
                error_message=error_msg,
                is_verified=False,
                verification_time=0.0
            )
    
    async def _analyze_with_smt(self, request: SMTRequest) -> SMTVerificationResult:
        """使用SMT进行分析"""
        # 确定内核类型
        kernel_type = self._detect_kernel_type(request.code)
        
        # 执行分析
        result = self.bug_localizer.analyze_kernel(request.code, kernel_type)
        
        return result
    
    def _detect_kernel_type(self, code: str) -> str:
        """检测内核类型"""
        if any(pattern in code for pattern in ['@triton.jit', 'tl.load', 'tl.store', 'triton.language']):
            return "triton"
        elif any(pattern in code for pattern in ['__global__', '__device__', 'threadIdx', 'blockIdx']):
            return "cuda"
        else:
            # 默认为triton
            return "triton"
    
    def _extract_counterexamples(self, result: SMTVerificationResult) -> Optional[str]:
        """提取反例信息"""
        if not result.bugs_found:
            return None
        
        # 提取最严重的bugs作为反例
        critical_bugs = [b for b in result.bugs_found if b.severity in ["critical", "high"]]
        
        if not critical_bugs:
            return None
        
        counterexamples = []
        for bug in critical_bugs[:3]:  # 限制反例数量
            counterexamples.append(
                f"Line {bug.line_number}: {bug.description} "
                f"(Confidence: {bug.confidence:.1%})"
            )
        
        return "\n".join(counterexamples)
    
    async def analyze_specific_property(self, code: str, property_type: str) -> Dict[str, Any]:
        """分析特定属性"""
        if not self.bug_localizer:
            return {"error": "SMT Bug Localizer not available"}
        
        kernel_type = self._detect_kernel_type(code)
        result = self.bug_localizer.analyze_kernel(code, kernel_type)
        
        # 根据属性类型过滤结果
        filtered_bugs = []
        
        if property_type == "memory_safety":
            filtered_bugs = [b for b in result.bugs_found 
                           if b.bug_type.value in ["memory_out_of_bounds", "buffer_overflow"]]
        elif property_type == "race_conditions":
            filtered_bugs = [b for b in result.bugs_found 
                           if b.bug_type.value in ["data_race", "atomic_race"]]
        elif property_type == "deadlock":
            filtered_bugs = [b for b in result.bugs_found 
                           if b.bug_type.value == "deadlock"]
        elif property_type == "type_safety":
            filtered_bugs = [b for b in result.bugs_found 
                           if b.bug_type.value == "type_mismatch"]
        else:
            filtered_bugs = result.bugs_found
        
        return {
            "property_type": property_type,
            "is_safe": len([b for b in filtered_bugs if b.severity in ["critical", "high"]]) == 0,
            "bugs_found": len(filtered_bugs),
            "critical_issues": len([b for b in filtered_bugs if b.severity == "critical"]),
            "high_issues": len([b for b in filtered_bugs if b.severity == "high"]),
            "verification_time": result.verification_time,
            "details": [
                {
                    "line": bug.line_number,
                    "type": bug.bug_type.value,
                    "severity": bug.severity,
                    "description": bug.description,
                    "fix": bug.suggested_fix,
                    "confidence": bug.confidence
                }
                for bug in filtered_bugs
            ]
        }
    
    async def batch_analyze_properties(self, code: str, properties: List[str]) -> Dict[str, Any]:
        """批量分析多个属性"""
        results = {}
        
        for prop in properties:
            try:
                results[prop] = await self.analyze_specific_property(code, prop)
            except Exception as e:
                results[prop] = {"error": str(e)}
        
        # 计算总体安全状态
        overall_safe = all(
            result.get("is_safe", False) for result in results.values() 
            if "error" not in result
        )
        
        total_bugs = sum(
            result.get("bugs_found", 0) for result in results.values()
            if "error" not in result
        )
        
        return {
            "overall_safe": overall_safe,
            "total_bugs_found": total_bugs,
            "properties_analyzed": len(properties),
            "properties_passed": len([r for r in results.values() if r.get("is_safe", False)]),
            "detailed_results": results
        }
    
    def get_bug_statistics(self, result: SMTVerificationResult) -> Dict[str, Any]:
        """获取bug统计信息"""
        if not result.bugs_found:
            return {"total": 0, "by_severity": {}, "by_type": {}}
        
        # 按严重程度统计
        by_severity = {}
        for bug in result.bugs_found:
            by_severity[bug.severity] = by_severity.get(bug.severity, 0) + 1
        
        # 按类型统计
        by_type = {}
        for bug in result.bugs_found:
            bug_type = bug.bug_type.value
            by_type[bug_type] = by_type.get(bug_type, 0) + 1
        
        # 置信度统计
        confidences = [bug.confidence for bug in result.bugs_found]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total": len(result.bugs_found),
            "by_severity": by_severity,
            "by_type": by_type,
            "average_confidence": avg_confidence,
            "high_confidence_bugs": len([b for b in result.bugs_found if b.confidence > 0.8])
        } 