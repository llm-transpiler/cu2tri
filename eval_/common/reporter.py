from pydantic import BaseModel
import torch
import traceback

from .config import EvalConfig


class CorrectnessResult(BaseModel):
    shape_match: bool = False
    values_match: bool = False
    overall_match: bool = False
    max_relative_error: float = float('inf')
    max_absolute_error: float = float('inf')
    error: str = ""
    traceback: str = ""



class PerformanceResult(BaseModel):
    perf_time_ms: float = float('inf')
    error: str = ""
    traceback: str = ""


def _compare_tensor_results(triton_result: torch.Tensor, torch_result: torch.Tensor, config: EvalConfig) -> CorrectnessResult:
    """比较两个结果的匹配度"""
    try:
        # 形状比较
        shape_match = triton_result.shape == torch_result.shape
        
        # 数值比较
        if shape_match:
            # 使用allclose进行数值比较
            values_match = torch.allclose(
                triton_result, 
                torch_result, 
                rtol=config.rtol, 
                atol=config.atol
            )
            
            # 计算相对误差
            diff = torch.abs(triton_result - torch_result)
            max_rel_error = torch.max(diff / (torch.abs(torch_result) + 1e-10))
            max_abs_error = torch.max(diff)
            
        else:
            values_match = False
            max_rel_error = float('inf')
            max_abs_error = float('inf')
        
        return CorrectnessResult(
            shape_match=shape_match,
            values_match=values_match,
            overall_match=shape_match and values_match,
            max_relative_error=float(max_rel_error),
            max_absolute_error=float(max_abs_error),
        )
        
    except Exception as e:
        return CorrectnessResult(
            error=str(e),
            traceback=traceback.format_exc()
        )

