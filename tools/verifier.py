from abc import ABC, abstractmethod
import torch
from utils.config_utils import TestResult
from utils.io_utils import capture_output
import traceback
import logging
from typing import Tuple
import copy
from collections.abc import Iterable
class KernelVerifier(ABC):
    @staticmethod
    def safe_first_run(kernel_func: callable, args: tuple, kernel_name: str, logger: logging.Logger = None) -> TestResult:
        """Safe execution of first run with error capture"""
        result = TestResult(kernel_name=kernel_name, compile_success=True)
        
        logger.info(f"  Execute First Run Test...")
        
        try:
            with capture_output() as (stdout_capture, stderr_capture):
                kernel_func(*args)
            
            result.running_success = True
            result.running_message = f"{kernel_name} First Run Success"
            logger.info(f"  ✓ {kernel_name} First Run Success")
            
        except Exception as e:
            result.running_error = str(e)
            result.running_message = f"{kernel_name} First Run Failed: {str(e)}"
            logger.info(f"  ✗ {result.running_message}")
            logger.info(f"  Error Details: {traceback.format_exc()}")
        
        return result
    
    @staticmethod
    def test_correctness(kernel_func: callable, args: tuple, reference: torch.Tensor,
                        kernel_name: str, rtol: float, atol: float, logger: logging.Logger = None) -> bool:
        """Test correctness with consistency and accuracy checks"""
        try:
            assert args is not None
            # Tensor is Iterable
            if not isinstance(args, (list, tuple, dict)) or isinstance(args, (str, bytes)):
                args = (args,)
            with torch.no_grad():
                # Multiple runs for consistency check
                outputs = []
                for _ in range(3):
                    output = kernel_func(*args)
                    output_clone = []
                    if not isinstance(output, (list, tuple, dict)) or isinstance(output, (str, bytes)):
                        output = [output]
                    for out in output:
                        try: # if isinstance(out, torch.Tensor):
                            output_clone.append(out.detach().clone())
                        except:
                            try:
                                output_clone.append(copy.deepcopy(out))
                            except:
                                try:
                                    output_clone.append(copy.copy(out))
                                except:
                                    output_clone.append(out)
                    outputs.append(output_clone)
                torch.cuda.synchronize()
                
                # Helper function to compare tensors (single or multiple)
                def compare_outputs(out1, out2):
                    """Compare two outputs (single tensor or list of tensors)"""
                    if isinstance(out1, list) and isinstance(out2, list):
                        if len(out1) != len(out2):
                            logger.error(f"  Output length mismatch: {len(out1)} != {len(out2)}")
                            return float('inf')
                        diffs = []
                        for o1, o2 in zip(out1, out2):
                            if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                                diffs.append(torch.max(torch.abs(o1 - o2)).item())
                        return max(diffs) if diffs else 0.0
                    elif isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
                        return torch.max(torch.abs(out1 - out2)).item()
                    else:
                        # For non-tensor outputs, try direct comparison
                        try:
                            return abs(out1 - out2) if out1 != out2 else 0.0
                        except:
                            return 0.0 if out1 == out2 else float('inf')
                
                def check_accuracy(output, ref, rtol, atol):
                    """Check accuracy between output and reference"""
                    if isinstance(output, list) and isinstance(ref, list):
                        if len(output) != len(ref):
                            logger.error(f"  Output length mismatch: {len(output)} != {len(ref)}")
                            return False, float('inf')
                        
                        all_passed = True
                        max_rel_err = 0.0
                        
                        for out, r in zip(output, ref):
                            if isinstance(out, torch.Tensor) and isinstance(r, torch.Tensor):
                                passed = torch.allclose(out, r, rtol=rtol, atol=atol)
                                rel_err = torch.max(torch.abs((out - r) / (r + 1e-8))).item()
                                all_passed = all_passed and passed
                                max_rel_err = max(max_rel_err, rel_err)
                            else:
                                # For non-tensor outputs
                                try:
                                    passed = abs(out - r) <= atol + rtol * abs(r)
                                    rel_err = abs(out - r) / (abs(r) + 1e-8)
                                    all_passed = all_passed and passed
                                    max_rel_err = max(max_rel_err, rel_err)
                                except:
                                    all_passed = all_passed and (out == r)
                        return all_passed, max_rel_err
                    
                    elif isinstance(output, torch.Tensor) and isinstance(ref, torch.Tensor):
                        passed = torch.allclose(output, ref, rtol=rtol, atol=atol)
                        rel_err = torch.max(torch.abs((output - ref) / (ref + 1e-8))).item()
                        return passed, rel_err
                    else:
                        # For non-tensor single outputs
                        try:
                            passed = abs(output - ref) <= atol + rtol * abs(ref)
                            rel_err = abs(output - ref) / (abs(ref) + 1e-8)
                            return passed, rel_err
                        except:
                            if output == ref:
                                return True, 0.0
                            else:
                                return False, float('inf')
                
                # Consistency check
                max_diff = 0.0
                for i in range(1, len(outputs)):
                    diff = compare_outputs(outputs[0], outputs[i])
                    max_diff = max(max_diff, diff)
                if not isinstance(reference, (list, tuple, dict)) or isinstance(reference, (str, bytes)):
                    reference = [reference]
                # Accuracy check
                accuracy_passed, rel_err = check_accuracy(outputs[0], reference, rtol, atol)
                
                logger.info(f"  Consistency: {'PASS' if max_diff < atol else 'FAIL'} (max diff: {max_diff:.2e})")
                logger.info(f"  Accuracy: {'PASS' if accuracy_passed else 'FAIL'} (rel_err: {rel_err:.2e})")
                
                if accuracy_passed and max_diff < atol:
                    return True, "Success"
                return False, f"Consistency: {'PASS' if max_diff < atol else 'FAIL'} (max diff: {max_diff:.2e}), Accuracy: {'PASS' if accuracy_passed else 'FAIL'} (rel_err: {rel_err:.2e})"
                
        except Exception as e:
            logger.error(f"  Error Details: {traceback.format_exc()}")
            logger.info(f"  ✗ {kernel_name} Correctness Test Exception: {e}")
            return False, f"{kernel_name} Correctness Test Exception: {e}, {traceback.format_exc()}"
