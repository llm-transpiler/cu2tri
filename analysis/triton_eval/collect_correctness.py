#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd

def find_eval_log_files(base_dir: str, date: str = "20250626_122406") -> List[str]:
    """查找所有的 eval.log 文件"""
    eval_logs = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"错误：基础目录 {base_dir} 不存在")
        return []
    
    for kernel_dir in base_path.iterdir():
        if kernel_dir.is_dir():
            eval_log = kernel_dir / "logs" / "gemini_2_5_pro" / date / "eval.log"
            if eval_log.exists():
                eval_logs.append(str(eval_log))
    
    return sorted(eval_logs)

def extract_kernel_name(log_path: str) -> str:
    """从日志路径提取内核名称"""
    return Path(log_path).parent.parent.parent.parent.name

def parse_triton_cuda_result(log_content: str) -> Dict:
    """解析 Triton vs CUDA 测试的完整结果信息"""
    result = {
        "has_triton_cuda_test": False,
        "comp_exec_success": None,
        "dtype_match": None,
        "shape_match": None,
        "values_match": None,
        "overall_match": None,
        "max_relative_error": None,
        "max_absolute_error": None,
        "error_message": None,
        "traceback": None,
        "output_capture": None,
        "full_result_dict": None
    }
    
    # 首先检查是否有 Triton vs CUDA 测试
    if "Triton vs CUDA" not in log_content:
        return result
    
    result["has_triton_cuda_test"] = True
    
    # 分步解析，找到 Triton vs CUDA 相关的日志行
    lines = log_content.split('\n')
    triton_cuda_lines = []
    
    for i, line in enumerate(lines):
        if "Triton vs CUDA" in line and ("Task completed" in line or "Task result" in line):
            # 找到相关行后，取接下来的几行确保完整信息
            triton_cuda_lines.extend(lines[i:min(i+5, len(lines))])
    
    if not triton_cuda_lines:
        return result
    
    # 在相关行中提取信息
    triton_cuda_text = ' '.join(triton_cuda_lines)
    
    # 尝试提取完整的结果字典
    dict_pattern = r"\{'comp_exec_success': [^}]+\}"
    dict_match = re.search(dict_pattern, triton_cuda_text)
    if dict_match:
        result["full_result_dict"] = dict_match.group(0)
        
        # 尝试使用更安全的方式解析字典
        dict_str = dict_match.group(0)
        try:
            # 使用 ast.literal_eval 更安全，但如果失败则回退到正则表达式
            import ast
            parsed_dict = ast.literal_eval(dict_str)
            result["comp_exec_success"] = parsed_dict.get('comp_exec_success')
            result["dtype_match"] = parsed_dict.get('dtype_match')
            result["shape_match"] = parsed_dict.get('shape_match')
            result["values_match"] = parsed_dict.get('values_match')
            result["overall_match"] = parsed_dict.get('overall_match')
            result["max_relative_error"] = parsed_dict.get('max_relative_error')
            result["max_absolute_error"] = parsed_dict.get('max_absolute_error')
            result["error_message"] = parsed_dict.get('error', '')
            result["traceback"] = parsed_dict.get('traceback', '')
            result["output_capture"] = parsed_dict.get('output_capture', '')
        except (ValueError, SyntaxError) as e:
            print(f"ast.literal_eval 解析失败，尝试使用正则表达式: {e}")
            # 如果 ast.literal_eval 失败，使用正则表达式单独提取各个字段
            pass
    
    # 如果上面的解析失败，尝试单独提取各个字段
    if result["comp_exec_success"] is None:
        for field in ['comp_exec_success', 'dtype_match', 'shape_match', 'values_match', 'overall_match']:
            pattern = f"'{field}': (True|False)"
            match = re.search(pattern, triton_cuda_text)
            if match:
                result[field] = match.group(1) == 'True'
        
        # 提取数值字段
        for field in ['max_relative_error', 'max_absolute_error']:
            pattern = f"'{field}': ([\\d.e-]+)"
            match = re.search(pattern, triton_cuda_text)
            if match:
                try:
                    result[field] = float(match.group(1))
                except ValueError:
                    pass
        
        # 提取字符串字段 - 更保守的方式
        for field in ['error', 'traceback', 'output_capture']:
            # 尝试匹配空字符串
            empty_pattern = f"'{field}': ''"
            if re.search(empty_pattern, triton_cuda_text):
                if field == 'error':
                    result["error_message"] = ''
                else:
                    result[field] = ''
                continue
            
            # 尝试匹配非空字符串，但只取安全的部分
            pattern = f"'{field}': '([^']*?)'"
            match = re.search(pattern, triton_cuda_text)
            if match:
                value = match.group(1)
                if field == 'error':
                    result["error_message"] = value
                else:
                    result[field] = value
    
    return result

def extract_triton_language_errors(error_message: str, traceback: str) -> Dict:
    """提取 triton.language 相关的错误信息"""
    error_info = {
        "error_type": "unknown",
        "missing_attribute": None,
        "module_path": None,
        "is_triton_language_error": False
    }
    
    if not error_message:
        return error_info
    
    # 检查是否为 triton.language 相关错误
    triton_lang_pattern = r"module '(triton\.language[^']*?)' has no attribute '([^']+)'"
    match = re.search(triton_lang_pattern, error_message)
    
    if match:
        error_info["is_triton_language_error"] = True
        error_info["module_path"] = match.group(1)
        error_info["missing_attribute"] = match.group(2)
        error_info["error_type"] = "missing_attribute"
    
    return error_info

def analyze_eval_logs(base_dir: str, date: str = "20250626_122406") -> Dict:
    """分析所有 eval.log 文件"""
    eval_logs = find_eval_log_files(base_dir, date)
    print(f"找到 {len(eval_logs)} 个 eval.log 文件")
    
    results = []
    failed_tests = []
    triton_language_errors = defaultdict(list)
    error_summary = defaultdict(int)
    match_failure_summary = defaultdict(int)
    
    for idx, log_path in enumerate(eval_logs):
        kernel_name = extract_kernel_name(log_path)
        print(f"正在处理 {idx+1}/{len(eval_logs)}: {kernel_name}")
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析 Triton vs CUDA 结果
            triton_cuda_result = parse_triton_cuda_result(content)
            
            result = {
                "kernel_name": kernel_name,
                "log_path": log_path,
                **triton_cuda_result
            }
            
            # 检查是否有测试失败
            has_failure = False
            failure_reasons = []
            
            if triton_cuda_result["has_triton_cuda_test"]:
                # 检查各种匹配失败
                if triton_cuda_result["comp_exec_success"] is False:
                    has_failure = True
                    failure_reasons.append("comp_exec_success")
                    match_failure_summary["comp_exec_success"] += 1
                
                if triton_cuda_result["dtype_match"] is False:
                    has_failure = True
                    failure_reasons.append("dtype_match")
                    match_failure_summary["dtype_match"] += 1
                
                if triton_cuda_result["shape_match"] is False:
                    has_failure = True
                    failure_reasons.append("shape_match")
                    match_failure_summary["shape_match"] += 1
                
                if triton_cuda_result["values_match"] is False:
                    has_failure = True
                    failure_reasons.append("values_match")
                    match_failure_summary["values_match"] += 1
                
                if triton_cuda_result["overall_match"] is False:
                    has_failure = True
                    failure_reasons.append("overall_match")
                    match_failure_summary["overall_match"] += 1
                
                result["has_failure"] = has_failure
                result["failure_reasons"] = failure_reasons
                
                # 如果有失败，记录详细信息
                if has_failure:
                    failed_tests.append({
                        "kernel_name": kernel_name,
                        "kernel_dir": str(Path(log_path).parent.parent.parent.parent),
                        "failure_reasons": failure_reasons,
                        "full_result": triton_cuda_result["full_result_dict"] or "无法解析完整结果",
                        **{k: v for k, v in triton_cuda_result.items() if k.endswith('_match') or k in ['comp_exec_success', 'max_relative_error', 'max_absolute_error', 'error_message']}
                    })
                    print(f"  发现失败: {', '.join(failure_reasons)}")
            
            # 如果有错误信息，进一步分析 triton.language 错误
            if triton_cuda_result["error_message"]:
                error_details = extract_triton_language_errors(
                    triton_cuda_result["error_message"],
                    triton_cuda_result["traceback"] or ""
                )
                result.update(error_details)
                
                # 统计 triton.language 错误类型
                if error_details["is_triton_language_error"]:
                    error_key = f"{error_details['module_path']}.{error_details['missing_attribute']}"
                    triton_language_errors[error_key].append(kernel_name)
                    error_summary[error_details["missing_attribute"]] += 1
            
            results.append(result)
            
        except Exception as e:
            print(f"解析 {log_path} 时出错: {str(e)}")
            results.append({
                "kernel_name": kernel_name,
                "log_path": log_path,
                "parsing_error": str(e)
            })
    
    # 按内核名称数字排序
    def get_sort_key(result):
        kernel_name = result["kernel_name"]
        try:
            num_match = re.match(r'(\d+)', kernel_name)
            return int(num_match.group(1)) if num_match else 999
        except:
            return 999
    
    results.sort(key=get_sort_key)
    failed_tests.sort(key=lambda x: get_sort_key(x))
    
    # 统计信息
    total_kernels = len(results)
    has_triton_cuda_test = sum(1 for r in results if r.get("has_triton_cuda_test", False))
    failed_test_count = sum(1 for r in results if r.get("has_failure", False))
    triton_language_error_count = sum(1 for r in results if r.get("is_triton_language_error", False))
    
    return {
        "base_dir": base_dir,
        "date": date,
        "total_kernels": total_kernels,
        "has_triton_cuda_test": has_triton_cuda_test,
        "failed_test_count": failed_test_count,
        "triton_language_error_count": triton_language_error_count,
        "results": results,
        "failed_tests": failed_tests,
        "triton_language_errors": dict(triton_language_errors),
        "error_summary": dict(error_summary),
        "match_failure_summary": dict(match_failure_summary)
    }

def create_error_reports(analysis_results: Dict, output_dir: str = ""):
    """创建错误报告"""
    if not output_dir:
        output_dir = "."
    
    # 1. 创建详细的失败测试信息表
    if analysis_results["failed_tests"]:
        failed_details = []
        for test in analysis_results["failed_tests"]:
            # 安全地处理可能为None的错误信息
            error_msg = test.get("error_message", "") or ""
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            failed_details.append({
                "内核名称": test["kernel_name"],
                "内核目录": test["kernel_dir"],
                "失败原因": ", ".join(test["failure_reasons"]),
                "编译执行成功": test.get("comp_exec_success", "N/A"),
                "数据类型匹配": test.get("dtype_match", "N/A"),
                "形状匹配": test.get("shape_match", "N/A"),
                "数值匹配": test.get("values_match", "N/A"),
                "整体匹配": test.get("overall_match", "N/A"),
                "最大相对误差": test.get("max_relative_error", "N/A"),
                "最大绝对误差": test.get("max_absolute_error", "N/A"),
                "错误信息": error_msg,
                "完整结果": test["full_result"]
            })
        
        df_failed = pd.DataFrame(failed_details)
        df_failed.to_csv(f"{output_dir}/failed_tests.csv", index=False, encoding='utf-8-sig')
    
    # 2. 创建匹配失败统计表
    if analysis_results["match_failure_summary"]:
        match_stats = []
        for failure_type, count in analysis_results["match_failure_summary"].items():
            match_stats.append({
                "失败类型": failure_type,
                "出现次数": count
            })
        
        df_match_stats = pd.DataFrame(match_stats)
        df_match_stats = df_match_stats.sort_values("出现次数", ascending=False)
        df_match_stats.to_csv(f"{output_dir}/match_failure_statistics.csv", index=False, encoding='utf-8-sig')
    
    # 3. 创建详细的 triton.language 错误信息表
    if analysis_results["error_summary"]:
        error_details = []
        for result in analysis_results["results"]:
            if result.get("is_triton_language_error", False):
                # 安全地处理可能为None的错误信息
                error_msg = result.get("error_message", "") or ""
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                
                error_details.append({
                    "内核名称": result["kernel_name"],
                    "模块路径": result.get("module_path", ""),
                    "缺失属性": result.get("missing_attribute", ""),
                    "错误信息": error_msg,
                })
        
        if error_details:
            df_errors = pd.DataFrame(error_details)
            df_errors.to_csv(f"{output_dir}/triton_language_errors.csv", index=False, encoding='utf-8-sig')
    
    # 4. 保存完整的分析结果
    with open(f"{output_dir}/triton_cuda_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

def print_summary(analysis_results: Dict):
    """打印分析摘要"""
    print("\n" + "="*80)
    print("Triton vs CUDA 测试结果分析摘要")
    print("="*80)
    print(f"总内核数量: {analysis_results['total_kernels']}")
    print(f"有 Triton vs CUDA 测试的内核: {analysis_results['has_triton_cuda_test']}")
    print(f"测试失败的内核: {analysis_results['failed_test_count']}")
    print(f"出现 triton.language 错误的内核: {analysis_results['triton_language_error_count']}")
    
    if analysis_results["match_failure_summary"]:
        print("\n匹配失败统计:")
        for failure_type, count in sorted(analysis_results["match_failure_summary"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {failure_type}: {count} 次")
    
    if analysis_results["error_summary"]:
        print("\n缺失属性统计:")
        for attr, count in sorted(analysis_results["error_summary"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {attr}: {count} 次")
    
    # 打印失败测试的详细信息 - 优化格式
    if analysis_results["failed_tests"]:
        print(f"\n失败测试详细信息 ({len(analysis_results['failed_tests'])} 个):")
        print("-" * 80)
        for i, test in enumerate(analysis_results["failed_tests"], 1):
            print(f"\n{i}. 内核: {test['kernel_name']}")
            print(f"   目录: {test['kernel_dir']}")
            print(f"   失败原因: {', '.join(test['failure_reasons'])}")
            
            # 显示各项匹配状态
            status_items = []
            for key in ['comp_exec_success', 'dtype_match', 'shape_match', 'values_match', 'overall_match']:
                value = test.get(key, 'N/A')
                if value is False:
                    status_items.append(f"{key}: ❌")
                elif value is True:
                    status_items.append(f"{key}: ✅")
                else:
                    status_items.append(f"{key}: {value}")
            print(f"   状态: {' | '.join(status_items)}")
            
            # 显示错误信息（如果有的话，限制长度）
            if test.get('error_message'):
                error_msg = test['error_message']
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"   错误: {error_msg}")
            
            # 显示数值错误（如果有的话）
            if test.get('max_relative_error') and test.get('max_absolute_error'):
                rel_err = test['max_relative_error']
                abs_err = test['max_absolute_error']
                if rel_err != 1000000000.0:  # 不是默认错误值
                    print(f"   误差: 相对={rel_err:.2e}, 绝对={abs_err:.2e}")
        
        print("-" * 80)
        print(f"\n💡 提示: 详细信息已保存到 CSV 和 JSON 文件中")

def main():
    """主函数"""
    set_name = "02_fused_op"
    set_name = "01_single_op"
    base_dir = f"/workspace/cu2tri/outputs/cu2tri/kernelbench_c/{set_name}"
    date = "20250626_122406"
    date = "20250625_040552"
    date = "20250627_200217"
    # date = "20250627_191551"
    output_dir = f"/workspace/analysis/triton_eval/{set_name}"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 分析所有日志文件
    analysis_results = analyze_eval_logs(base_dir, date)
    
    # 创建报告
    create_error_reports(analysis_results, output_dir)
    
    # 打印摘要
    print_summary(analysis_results)
    
    print(f"\n生成的文件 (在 {output_dir} 目录):")
    print("- failed_tests.csv: 所有失败测试的详细信息")
    print("- match_failure_statistics.csv: 匹配失败统计")
    print("- triton_language_errors.csv: triton.language 错误详情")
    print("- triton_cuda_analysis.json: 完整分析结果")

if __name__ == "__main__":
    main()
