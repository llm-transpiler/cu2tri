#!/usr/bin/env python3
"""
CUDA Check Results Analyzer - Analyze results.json files in check_cuda folders
Similar to performance_analyzer.py, but specifically for CUDA vs PyTorch comparison results

Features:
1. Count match results from all results.json files in check_cuda folders
2. List absolute and relative errors
3. Judge pass/fail based on 1e-3 threshold
4. Only analyze performance for kernels that pass match tests
5. Plot performance with CUDA as 1.0 baseline showing PyTorch performance
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# # Set matplotlib font to avoid Chinese font warnings
# plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False

def find_cuda_check_results(base_dir: str) -> List[str]:
    """Find all results.json files in check_cuda directories"""
    results_files = []
    base_path = Path(base_dir)
    
    # Traverse all kernel directories
    for kernel_dir in base_path.iterdir():
        if kernel_dir.is_dir():
            # Look for logs/check_cuda/results.json
            results_file = kernel_dir / "logs" / "check_cuda" / "results.json"
            if results_file.exists():
                results_files.append(str(results_file))
    
    return sorted(results_files)

def extract_kernel_name_from_path(results_path: str) -> str:
    """Extract kernel name from results.json file path"""
    parts = Path(results_path).parts
    
    # Find directory containing kernel name
    for i, part in enumerate(parts):
        if part == 'logs' and i > 0:
            kernel_name = parts[i-1]
            # Remove trailing underscore
            if kernel_name.endswith('_'):
                return kernel_name.rstrip('_')
            else:
                return kernel_name
    
    return "Unknown"

def parse_cuda_check_result(results_path: str) -> Dict:
    """Parse single CUDA check results.json file"""
    result = {
        "kernel_name": extract_kernel_name_from_path(results_path),
        "results_path": results_path,
        "success": False,
        "correctness": {
            "overall_match": None,
            "values_match": None,
            "dtype_match": None,
            "shape_match": None,
            "max_relative_error": None,
            "max_absolute_error": None,
        },
        "performance": {
            "cuda_time_ms": None,
            "torch_time_ms": None,
            "pytorch_speedup": None,  # PyTorch time / CUDA time 
            "cuda_baseline": 1.0      # CUDA as baseline
        },
        "error_threshold_pass": {
            "relative_1e3": None,    # relative error < 1e-3
            "absolute_1e3": None,    # absolute error < 1e-3
            "combined_1e3": None     # both pass
        },
        "parsing_errors": []
    }
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse basic information
        result["success"] = data.get("success", False)
        
        # Parse correctness results
        if "correctness" in data:
            correctness = data["correctness"]
            result["correctness"]["overall_match"] = correctness.get("overall_match")
            result["correctness"]["values_match"] = correctness.get("values_match")
            result["correctness"]["dtype_match"] = correctness.get("dtype_match")
            result["correctness"]["shape_match"] = correctness.get("shape_match")
            result["correctness"]["max_relative_error"] = correctness.get("max_relative_error")
            result["correctness"]["max_absolute_error"] = correctness.get("max_absolute_error")
            
            # Judge error threshold pass status
            max_rel_err = result["correctness"]["max_relative_error"]
            max_abs_err = result["correctness"]["max_absolute_error"]
            
            if max_rel_err is not None:
                result["error_threshold_pass"]["relative_1e3"] = max_rel_err < 1e-3
            if max_abs_err is not None:
                result["error_threshold_pass"]["absolute_1e3"] = max_abs_err < 1e-3
            
            # Combined judgment: both relative and absolute errors must be < 1e-3 to pass
            if (result["error_threshold_pass"]["relative_1e3"] is not None and 
                result["error_threshold_pass"]["absolute_1e3"] is not None):
                result["error_threshold_pass"]["combined_1e3"] = (
                    result["error_threshold_pass"]["relative_1e3"] and 
                    result["error_threshold_pass"]["absolute_1e3"]
                )
        
        # Parse performance results
        if "performance" in data:
            performance = data["performance"]
            
            # Extract from detailed format
            if "summary" in performance:
                summary = performance["summary"]
                result["performance"]["cuda_time_ms"] = summary.get("cuda_time_ms")
                result["performance"]["torch_time_ms"] = summary.get("torch_time_ms")
                result["performance"]["pytorch_speedup"] = summary.get("pytorch_speedup")
            elif "cuda" in performance and "torch" in performance:
                # Extract from detailed cuda/torch sections
                result["performance"]["cuda_time_ms"] = performance["cuda"].get("perf_time_ms")
                result["performance"]["torch_time_ms"] = performance["torch"].get("perf_time_ms")
                
                # Calculate PyTorch performance relative to CUDA
                cuda_time = result["performance"]["cuda_time_ms"]
                torch_time = result["performance"]["torch_time_ms"]
                if cuda_time and torch_time and cuda_time > 0:
                    result["performance"]["pytorch_speedup"] = torch_time / cuda_time
                    
    except Exception as e:
        result["parsing_errors"].append(f"Error parsing {results_path}: {str(e)}")
        
    return result

def categorize_kernel(kernel_name: str) -> str:
    """Categorize kernel by type"""
    kernel_lower = kernel_name.lower()
    
    if any(x in kernel_lower for x in ['matmul', 'matrix_multiplication', 'tensor_matrix']):
        return "Matrix Operations"
    elif any(x in kernel_lower for x in ['conv']):
        return "Convolution"
    elif any(x in kernel_lower for x in ['relu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'swish', 'elu', 'leakyrelu', 'hardtanh', 'softsign', 'softplus', 'selu', 'hardsigmoid']):
        return "Activation Functions"
    elif any(x in kernel_lower for x in ['norm']):
        return "Normalization"
    elif any(x in kernel_lower for x in ['pooling']):
        return "Pooling"
    elif any(x in kernel_lower for x in ['reduction', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin']):
        return "Reduction"
    elif any(x in kernel_lower for x in ['loss']):
        return "Loss Functions"
    elif any(x in kernel_lower for x in ['cumsum', 'cumprod']):
        return "Cumulative Operations"
    else:
        return "Other"

def analyze_cuda_check_results(base_dir: str, analysis_dir: str = "") -> Dict:
    """Analyze all CUDA check results"""
    results_files = find_cuda_check_results(base_dir)
    print(f"Found {len(results_files)} CUDA check result files")
    
    results = []
    parsing_errors = []
    
    for results_file in results_files:
        result = parse_cuda_check_result(results_file)
        results.append(result)
        if result["parsing_errors"]:
            parsing_errors.extend(result["parsing_errors"])
    
    # Sort by kernel name (numerical order)
    def get_sort_key(result):
        kernel_name = result["kernel_name"]
        try:
            num_match = re.match(r'(\d+)', kernel_name)
            return int(num_match.group(1)) if num_match else 999
        except:
            return 999
    
    results.sort(key=get_sort_key)
    
    # Statistical analysis
    total_kernels = len(results)
    
    # Correctness statistics
    correctness_stats = {
        "total_tested": 0,
        "overall_match_pass": 0,
        "values_match_pass": 0,
        "dtype_match_pass": 0,
        "shape_match_pass": 0,
    }
    
    # Error threshold statistics
    threshold_stats = {
        "relative_1e3_pass": 0,
        "absolute_1e3_pass": 0,
        "combined_1e3_pass": 0,
        "total_with_errors": 0
    }
    
    # Performance statistics
    performance_stats = {
        "total_with_performance": 0,
        "match_and_performance": 0,  # Only count matched kernels with performance data
    }
    
    # Collect mismatch cases
    mismatch_cases = []
    high_error_cases = []
    
    # Group statistics by kernel type
    kernel_type_stats = defaultdict(lambda: {
        "total": 0,
        "match_pass": 0,
        "threshold_pass": 0,
        "performance_available": 0
    })
    
    for result in results:
        kernel_type = categorize_kernel(result["kernel_name"])
        kernel_type_stats[kernel_type]["total"] += 1
        
        # Correctness statistics
        if result["correctness"]["overall_match"] is not None:
            correctness_stats["total_tested"] += 1
            if result["correctness"]["overall_match"]:
                correctness_stats["overall_match_pass"] += 1
                kernel_type_stats[kernel_type]["match_pass"] += 1
            else:
                # Collect mismatch cases
                mismatch_cases.append({
                    "kernel_name": result["kernel_name"],
                    "max_rel_err": result["correctness"]["max_relative_error"],
                    "max_abs_err": result["correctness"]["max_absolute_error"],
                    "values_match": result["correctness"]["values_match"],
                    "dtype_match": result["correctness"]["dtype_match"],
                    "shape_match": result["correctness"]["shape_match"]
                })
                
        if result["correctness"]["values_match"]:
            correctness_stats["values_match_pass"] += 1
        if result["correctness"]["dtype_match"]:
            correctness_stats["dtype_match_pass"] += 1  
        if result["correctness"]["shape_match"]:
            correctness_stats["shape_match_pass"] += 1
            
        # Error threshold statistics
        if (result["correctness"]["max_relative_error"] is not None or 
            result["correctness"]["max_absolute_error"] is not None):
            threshold_stats["total_with_errors"] += 1
            
            if result["error_threshold_pass"]["relative_1e3"]:
                threshold_stats["relative_1e3_pass"] += 1
            if result["error_threshold_pass"]["absolute_1e3"]:
                threshold_stats["absolute_1e3_pass"] += 1
            if result["error_threshold_pass"]["combined_1e3"]:
                threshold_stats["combined_1e3_pass"] += 1
                kernel_type_stats[kernel_type]["threshold_pass"] += 1
            else:
                # Collect high error cases
                high_error_cases.append({
                    "kernel_name": result["kernel_name"],
                    "max_rel_err": result["correctness"]["max_relative_error"],
                    "max_abs_err": result["correctness"]["max_absolute_error"],
                    "relative_1e3_pass": result["error_threshold_pass"]["relative_1e3"],
                    "absolute_1e3_pass": result["error_threshold_pass"]["absolute_1e3"]
                })
        
        # Performance statistics  
        if (result["performance"]["cuda_time_ms"] is not None and 
            result["performance"]["torch_time_ms"] is not None):
            performance_stats["total_with_performance"] += 1
            kernel_type_stats[kernel_type]["performance_available"] += 1
            
            # Only analyze performance for matched kernels
            if result["correctness"]["overall_match"]:
                performance_stats["match_and_performance"] += 1
    
    return {
        "timestamp": datetime.now().isoformat(),
        "base_dir": base_dir,
        "total_kernels": total_kernels,
        "results": results,
        "correctness_stats": correctness_stats,
        "threshold_stats": threshold_stats,
        "performance_stats": performance_stats,
        "mismatch_cases": mismatch_cases,
        "high_error_cases": high_error_cases,
        "kernel_type_stats": dict(kernel_type_stats),
        "parsing_errors": parsing_errors
    }

def create_performance_plots(analysis_results: Dict, analysis_dir: str = ""):
    """Create performance comparison plots with CUDA as 1.0 baseline"""
    results = analysis_results["results"]
    
    # Only include matched results with performance data
    valid_results = []
    for result in results:
        if (result["correctness"]["overall_match"] and 
            result["performance"]["cuda_time_ms"] is not None and 
            result["performance"]["torch_time_ms"] is not None):
            
            # Extract number for sorting
            kernel_name = result["kernel_name"]
            try:
                num_match = re.match(r'(\d+)', kernel_name)
                sort_key = int(num_match.group(1)) if num_match else 999
            except:
                sort_key = 999
            valid_results.append((sort_key, result))
    
    valid_results.sort(key=lambda x: x[0])
    
    if not valid_results:
        print("No valid performance data available for plotting")
        return
    
    print(f"Found {len(valid_results)} kernels that pass matching and have performance data")
    
    # Group processing (20 per group)
    group_size = 20
    num_groups = (len(valid_results) + group_size - 1) // group_size
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, len(valid_results))
        group_results = [x[1] for x in valid_results[start_idx:end_idx]]
        
        # Create chart
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        kernel_names = []
        cuda_times = []      # CUDA as baseline = 1.0
        pytorch_ratios = []  # PyTorch relative to CUDA ratio
        
        for result in group_results:
            # Truncate kernel name for display
            kernel_name = result["kernel_name"]
            if len(kernel_name) > 25:
                display_name = kernel_name[:22] + "..."
            else:
                display_name = kernel_name
            kernel_names.append(display_name)
            
            # CUDA as baseline 1.0
            cuda_times.append(1.0)
            
            # Calculate PyTorch relative to CUDA ratio
            cuda_time = result["performance"]["cuda_time_ms"]
            torch_time = result["performance"]["torch_time_ms"]
            
            if cuda_time and cuda_time > 0:
                pytorch_ratio = torch_time / cuda_time
                pytorch_ratios.append(pytorch_ratio)
            else:
                pytorch_ratios.append(0)
        
        # Create bar chart
        x = np.arange(len(kernel_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cuda_times, width, label='CUDA (Baseline)', alpha=0.8, color='green')
        bars2 = ax.bar(x + width/2, pytorch_ratios, width, label='PyTorch (Relative)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Kernels', fontsize=12)
        ax.set_ylabel('Relative Time (CUDA = 1.0)', fontsize=12)
        ax.set_title(f'CUDA vs PyTorch Performance Comparison (Group {group_idx + 1})\nOnly includes kernels that pass correctness tests\nValues < 1.0 = Faster than CUDA', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_names, rotation=60, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # CUDA baseline label
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
                   '1.00', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # PyTorch ratio label
            if pytorch_ratios[i] > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.02,
                       f'{pytorch_ratios[i]:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Ensure analysis_dir has trailing slash
        if analysis_dir and not analysis_dir.endswith('/'):
            analysis_dir += '/'
        
        plot_file = f'{analysis_dir}cuda_pytorch_performance_group_{group_idx + 1}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {plot_file}")

def create_detailed_reports(analysis_results: Dict, analysis_dir: str = ""):
    """Create detailed CSV reports"""
    
    # Define helper function at the top to be available throughout the function
    def get_kernel_sort_key(kernel_name):
        try:
            num_match = re.match(r'(\d+)', kernel_name)
            return int(num_match.group(1)) if num_match else 999
        except:
            return 999
    
    # Ensure analysis_dir has trailing slash
    if analysis_dir and not analysis_dir.endswith('/'):
        analysis_dir += '/'
    
    # 1. Detailed test status table
    test_status_data = []
    for result in analysis_results["results"]:
        # Extract number for sorting
        kernel_name = result["kernel_name"]
        try:
            num_match = re.match(r'(\d+)', kernel_name)
            sort_key = int(num_match.group(1)) if num_match else 999
        except:
            sort_key = 999
            
        row = {
            "Sort_Key": sort_key,
            "Kernel_Name": result["kernel_name"],
            "Kernel_Type": categorize_kernel(result["kernel_name"]),
            "Success": "Yes" if result["success"] else "No",
            "Overall_Match": "Pass" if result["correctness"]["overall_match"] else 
                           ("Fail" if result["correctness"]["overall_match"] is False else "N/A"),
            "Values_Match": "Pass" if result["correctness"]["values_match"] else 
                          ("Fail" if result["correctness"]["values_match"] is False else "N/A"),
            "Dtype_Match": "Pass" if result["correctness"]["dtype_match"] else 
                         ("Fail" if result["correctness"]["dtype_match"] is False else "N/A"),
            "Shape_Match": "Pass" if result["correctness"]["shape_match"] else 
                         ("Fail" if result["correctness"]["shape_match"] is False else "N/A"),
            "Max_Relative_Error": f"{result['correctness']['max_relative_error']:.6e}" if result["correctness"]["max_relative_error"] is not None else "N/A",
            "Max_Absolute_Error": f"{result['correctness']['max_absolute_error']:.6e}" if result["correctness"]["max_absolute_error"] is not None else "N/A",
            "Relative_1e3_Pass": "Pass" if result["error_threshold_pass"]["relative_1e3"] else 
                               ("Fail" if result["error_threshold_pass"]["relative_1e3"] is False else "N/A"),
            "Absolute_1e3_Pass": "Pass" if result["error_threshold_pass"]["absolute_1e3"] else 
                               ("Fail" if result["error_threshold_pass"]["absolute_1e3"] is False else "N/A"),
            "Combined_1e3_Pass": "Pass" if result["error_threshold_pass"]["combined_1e3"] else 
                               ("Fail" if result["error_threshold_pass"]["combined_1e3"] is False else "N/A"),
            "CUDA_Time_ms": f"{result['performance']['cuda_time_ms']:.4f}" if result["performance"]["cuda_time_ms"] is not None else "N/A",
            "PyTorch_Time_ms": f"{result['performance']['torch_time_ms']:.4f}" if result["performance"]["torch_time_ms"] is not None else "N/A",
            "PyTorch_CUDA_Ratio": f"{result['performance']['pytorch_speedup']:.4f}" if result["performance"]["pytorch_speedup"] is not None else "N/A"
        }
        test_status_data.append(row)
    
    # Sort by numerical order
    test_status_data.sort(key=lambda x: x["Sort_Key"])
    
    # Remove sort key
    for row in test_status_data:
        del row["Sort_Key"]
    
    test_status_df = pd.DataFrame(test_status_data)
    test_status_df.to_csv(f'{analysis_dir}cuda_check_detailed_status.csv', index=False, encoding='utf-8-sig')
    
    # 2. Mismatch case details
    if analysis_results["mismatch_cases"]:
        mismatch_df = pd.DataFrame(analysis_results["mismatch_cases"])
        # Sort by kernel name using helper function
        mismatch_df['sort_key'] = mismatch_df['kernel_name'].apply(get_kernel_sort_key)
        mismatch_df = mismatch_df.sort_values('sort_key')
        mismatch_df = mismatch_df.drop('sort_key', axis=1)
        mismatch_df.to_csv(f'{analysis_dir}cuda_pytorch_mismatches.csv', index=False, encoding='utf-8-sig')
    else:
        # Create empty file
        pd.DataFrame(columns=['kernel_name', 'max_rel_err', 'max_abs_err', 'values_match', 'dtype_match', 'shape_match']).to_csv(f'{analysis_dir}cuda_pytorch_mismatches.csv', index=False, encoding='utf-8-sig')
    
    # 3. High error cases (failed 1e-3 threshold)
    if analysis_results["high_error_cases"]:
        high_error_df = pd.DataFrame(analysis_results["high_error_cases"])
        high_error_df['sort_key'] = high_error_df['kernel_name'].apply(get_kernel_sort_key)  
        high_error_df = high_error_df.sort_values(['sort_key', 'max_rel_err'], ascending=[True, False])
        high_error_df = high_error_df.drop('sort_key', axis=1)
        high_error_df.to_csv(f'{analysis_dir}high_error_cases_1e3.csv', index=False, encoding='utf-8-sig')
    else:
        # Create empty file
        pd.DataFrame(columns=['kernel_name', 'max_rel_err', 'max_abs_err', 'relative_1e3_pass', 'absolute_1e3_pass']).to_csv(f'{analysis_dir}high_error_cases_1e3.csv', index=False, encoding='utf-8-sig')
    
    # 4. Statistics by kernel type
    kernel_type_data = []
    for kernel_type, stats in analysis_results["kernel_type_stats"].items():
        kernel_type_data.append({
            "Kernel_Type": kernel_type,
            "Total": stats["total"],
            "Match_Pass": stats["match_pass"],
            "Threshold_Pass": stats["threshold_pass"],
            "Performance_Available": stats["performance_available"],
            "Match_Pass_Rate_%": f"{(stats['match_pass']/stats['total']*100):.1f}" if stats["total"] > 0 else "0.0",
            "Threshold_Pass_Rate_%": f"{(stats['threshold_pass']/stats['total']*100):.1f}" if stats["total"] > 0 else "0.0"
        })
    
    kernel_type_df = pd.DataFrame(kernel_type_data)
    kernel_type_df.to_csv(f'{analysis_dir}cuda_check_kernel_type_stats.csv', index=False, encoding='utf-8-sig')
    
    print("Created detailed reports:")
    print(f"- {analysis_dir}cuda_check_detailed_status.csv")
    print(f"- {analysis_dir}cuda_pytorch_mismatches.csv")
    print(f"- {analysis_dir}high_error_cases_1e3.csv")
    print(f"- {analysis_dir}cuda_check_kernel_type_stats.csv")

def main(analysis_dir: str = ""):
    """Main function with configurable analysis directory"""
    if not analysis_dir:
        # Default to /workspace/analysis/check_cuda/
        analysis_dir = "/workspace/analysis/check_cuda/"
    
    # Ensure analysis directory exists and has trailing slash
    analysis_path = Path(analysis_dir)
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    if not analysis_dir.endswith('/'):
        analysis_dir += '/'
    
    # Use hardcoded base directory for now, can be made configurable later
    base_dir = "/workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op"
    
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist")
        print("Please check if the path is correct")
        return
    
    print("Analyzing CUDA check result files...")
    analysis_results = analyze_cuda_check_results(base_dir, analysis_dir)
    
    # Save complete analysis results
    with open(f'{analysis_dir}cuda_check_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # Create performance comparison charts
    create_performance_plots(analysis_results, analysis_dir=analysis_dir)
    
    # Create detailed reports
    create_detailed_reports(analysis_results, analysis_dir=analysis_dir)
    
    # Print summary information
    print("\n" + "="*80)
    print("CUDA CHECK ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total kernels: {analysis_results['total_kernels']}")
    print(f"Successfully tested kernels: {analysis_results['correctness_stats']['total_tested']}")
    
    if analysis_results['correctness_stats']['total_tested'] > 0:
        match_rate = analysis_results['correctness_stats']['overall_match_pass'] / analysis_results['correctness_stats']['total_tested'] * 100
        print(f"Overall match pass: {analysis_results['correctness_stats']['overall_match_pass']}/{analysis_results['correctness_stats']['total_tested']} ({match_rate:.1f}%)")
    
    if analysis_results['threshold_stats']['total_with_errors'] > 0:
        threshold_rate = analysis_results['threshold_stats']['combined_1e3_pass'] / analysis_results['threshold_stats']['total_with_errors'] * 100
        print(f"1e-3 threshold pass: {analysis_results['threshold_stats']['combined_1e3_pass']}/{analysis_results['threshold_stats']['total_with_errors']} ({threshold_rate:.1f}%)")
    
    print(f"Kernels with performance data: {analysis_results['performance_stats']['total_with_performance']}")
    print(f"Matched kernels with performance data: {analysis_results['performance_stats']['match_and_performance']}")
    print(f"Mismatch cases: {len(analysis_results['mismatch_cases'])}")
    print(f"High error cases: {len(analysis_results['high_error_cases'])}")
    print(f"Parsing errors: {len(analysis_results['parsing_errors'])}")
    
    print(f"\nGenerated files (in {analysis_dir} directory):")
    print("- cuda_check_analysis.json")
    print("- cuda_pytorch_performance_group_*.png")
    print("- cuda_check_detailed_status.csv")
    print("- cuda_pytorch_mismatches.csv")
    print("- high_error_cases_1e3.csv")
    print("- cuda_check_kernel_type_stats.csv")

if __name__ == "__main__":
    # Use configurable analysis directory
    analysis_dir = "/workspace/analysis/check_cuda/"
    main(analysis_dir)