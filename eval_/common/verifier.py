from pydantic import BaseModel
import traceback
import copy


class CompareResult(BaseModel):
    comp_exec_success: bool = True
    dtype_match: bool = False
    shape_match: bool = False
    values_match: bool = False
    overall_match: bool = False
    max_relative_error: float = float('inf')
    max_absolute_error: float = float('inf')
    error: str = ""
    traceback: str = ""
    output_capture: str = ""


class PerformanceResult(BaseModel):
    perf_exec_success: bool = True
    perf_time_ms: float = float('inf')
    error: str = ""
    traceback: str = ""
    output_capture: str = ""


def _safe_copy_output_item(item):
    """安全地复制单个项目，使用多层fallback策略"""
    import torch
    
    try:
        # 优先处理tensor
        if isinstance(item, torch.Tensor):
            return item.cpu().detach().clone()
    except:
        pass
    
    # 多层fallback复制策略
    try:
        return item.detach().clone()
    except:
        try:
            return copy.deepcopy(item)
        except:
            try:
                return copy.copy(item)
            except:
                return item

def _safe_copy_input_item(item):
    """安全地复制单个项目，使用多层fallback策略"""
    import torch
    
    try:
        # 优先处理tensor
        if isinstance(item, torch.Tensor):
            return item.cuda().detach().clone()
    except:
        pass
    
    # 多层fallback复制策略
    try:
        return item.detach().clone()
    except:
        try:
            return copy.deepcopy(item)
        except:
            try:
                return copy.copy(item)
            except:
                return item


def _normalize_input(inputs):
    """规范化输入为统一格式，使用鲁棒的复制策略"""
    # 确保输入为list格式
    if not isinstance(inputs, (list, tuple)) or isinstance(inputs, (str, bytes)):
        inputs = [inputs]
    
    # 安全复制所有项目
    normalized = []
    for item in inputs:
        normalized.append(_safe_copy_input_item(item))
    
    return normalized


def _normalize_output(outputs):
    """规范化输出为统一格式，使用鲁棒的复制策略"""
    # 确保输出为list格式
    if not isinstance(outputs, (list, tuple)) or isinstance(outputs, (str, bytes)):
        outputs = [outputs]
    
    # 安全复制所有项目
    normalized = []
    for item in outputs:
        normalized.append(_safe_copy_output_item(item))
    
    return normalized


def _compare_single_tensor(tensor1, tensor2, rtol: float, atol: float):
    """比较单个tensor的匹配度"""
    import torch
    
    # 形状比较
    shape_match = tensor1.shape == tensor2.shape
    
    # dtype比较
    dtype_match = tensor1.dtype == tensor2.dtype
    
    # 数值比较
    if shape_match:
        try:
            # 处理dtype不匹配的情况
            if not dtype_match:
                # 尝试转换到相同dtype进行比较
                if tensor1.dtype.is_floating_point and tensor2.dtype.is_floating_point:
                    common_dtype = torch.promote_types(tensor1.dtype, tensor2.dtype)
                    tensor1_cmp = tensor1.to(common_dtype)
                    tensor2_cmp = tensor2.to(common_dtype)
                else:
                    tensor1_cmp, tensor2_cmp = tensor1, tensor2
            else:
                tensor1_cmp, tensor2_cmp = tensor1, tensor2
                
            values_match = torch.allclose(
                tensor1_cmp, 
                tensor2_cmp, 
                rtol=rtol, 
                atol=atol
            )
            
            # 计算误差
            diff = torch.abs(tensor1_cmp - tensor2_cmp)
            max_rel_error = torch.max(diff / (torch.abs(tensor2_cmp) + 1e-10))
            max_abs_error = torch.max(diff)
            
        except Exception:
            values_match = False
            max_rel_error = float('inf')
            max_abs_error = float('inf')
    else:
        values_match = False
        max_rel_error = float('inf')
        max_abs_error = float('inf')
    
    return shape_match, dtype_match, values_match, float(max_rel_error), float(max_abs_error)


def _compare_tensor_results(real_outs, ref_outs, rtol: float, atol: float) -> CompareResult:
    """比较两个结果的匹配度 - 支持单输出和多输出"""
    
    try:
        with torch.no_grad():
            import torch
            # 规范化输出格式
            real_outs = _normalize_output(real_outs)
            ref_outs = _normalize_output(ref_outs)
            
            # 检查输出数量匹配
            if len(real_outs) != len(ref_outs):
                return CompareResult(
                    error=f"Output count mismatch: {len(real_outs)} vs {len(ref_outs)}"
                )
            
            # 初始化结果
            overall_shape_match = True
            overall_dtype_match = True
            overall_values_match = True
            max_rel_error = 0.0
            max_abs_error = 0.0
            
            # 逐个比较输出
            for i, (real_out, ref_out) in enumerate(zip(real_outs, ref_outs)):
                if isinstance(real_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
                    shape_match, dtype_match, values_match, rel_err, abs_err = _compare_single_tensor(
                        real_out, ref_out, rtol, atol
                    )
                    
                    overall_shape_match &= shape_match
                    overall_dtype_match &= dtype_match
                    overall_values_match &= values_match
                    max_rel_error = max(max_rel_error, rel_err)
                    max_abs_error = max(max_abs_error, abs_err)
                    
                else:
                    # 非tensor输出的比较
                    shape_match = True  # 非tensor没有shape概念
                    dtype_match = type(real_out) == type(ref_out)
                    try:
                        if hasattr(real_out, '__len__') and hasattr(ref_out, '__len__'):
                            values_match = len(real_out) == len(ref_out) and all(
                                abs(a - b) <= atol + rtol * abs(b) 
                                for a, b in zip(real_out, ref_out)
                            )
                        else:
                            values_match = abs(real_out - ref_out) <= atol + rtol * abs(ref_out)
                    except:
                        values_match = real_out == ref_out
                    
                    overall_shape_match &= shape_match
                    overall_dtype_match &= dtype_match  
                    overall_values_match &= values_match
        
        return CompareResult(
            comp_exec_success=True,
            shape_match=overall_shape_match,
            dtype_match=overall_dtype_match,
            values_match=overall_values_match,
            overall_match=overall_shape_match and overall_dtype_match and overall_values_match,
            max_relative_error=max_rel_error,
            max_absolute_error=max_abs_error,
        )
        
    except Exception as e:
        return CompareResult(
            comp_exec_success=False,
            error=str(e),
            traceback=traceback.format_exc()
        )

