#!/usr/bin/env python3
"""
检查level1目录下每个case的gemini_2_5_pro性能测试结果
并生成以CUDA为基准(1.0)的相对性能柱状图
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model_dir_name = "gemini_2_5_pro"
# model_dir_name = "deepseek_deepseek_r1_0528_free"
os.makedirs(model_dir_name, exist_ok=True)

def parse_performance_log(log_path):
    """解析性能测试日志文件，提取Triton、CUDA、PyTorch的执行时间"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找性能测试结果部分
        pattern = r'📈 性能测试结果:\s*\n\s*Triton:\s*([\d.]+)\s*ms\s*\n\s*CUDA:\s*([\d.]+)\s*ms\s*\n\s*PyTorch:\s*([\d.]+)\s*ms'
        match = re.search(pattern, content)
        
        if match:
            triton_time = float(match.group(1))
            cuda_time = float(match.group(2))
            pytorch_time = float(match.group(3))
            return triton_time, cuda_time, pytorch_time
        
        return None
    except Exception as e:
        print(f"解析日志文件失败 {log_path}: {e}")
        return None

def check_cases():
    """检查所有case的性能测试结果"""
    level1_dir = Path("outputs/level1")
    
    if not level1_dir.exists():
        print(f"目录不存在: {level1_dir}")
        return [], {}
    
    results = []
    stats = {
        'total_cases': 0,
        'has_logs': 0,
        'has_target_model': 0,  
        'has_log_files': 0,
        'successful_parse': 0,
        'parse_error': 0,
        'no_logs': 0,
        'no_target_model': 0,
        'no_log_files': 0
    }
    
    # 遍历所有case目录
    for case_dir in sorted(level1_dir.iterdir()):
        if not case_dir.is_dir():
            continue
            
        case_name = case_dir.name
        print(f"\n检查案例: {case_name}")
        stats['total_cases'] += 1
        
        # 检查是否有logs目录
        logs_dir = case_dir / "logs"
        if not logs_dir.exists():
            print(f"  ❌ 没有找到logs目录")
            stats['no_logs'] += 1
            continue
        
        stats['has_logs'] += 1
        
        # 检查是否有目标模型目录
        target_model_dir = logs_dir / model_dir_name
        if not target_model_dir.exists():
            print(f"  ❌ 没有找到{model_dir_name}目录 (但有logs目录)")
            stats['no_target_model'] += 1
            continue
        
        print(f"  ✅ 找到{model_dir_name}目录")
        stats['has_target_model'] += 1
        
        # 查找eval_latest.log或最新的eval.log
        eval_log = target_model_dir / "eval_latest.log"
        if not eval_log.exists():
            # 查找时间戳目录下的eval.log
            timestamp_dirs = [d for d in target_model_dir.iterdir() if d.is_dir()]
            if timestamp_dirs:
                # 选择最新的时间戳目录
                latest_dir = max(timestamp_dirs, key=lambda x: x.name)
                eval_log = latest_dir / "eval.log"
        
        if not eval_log.exists():
            print(f"  ❌ 没有找到性能测试日志文件")
            stats['no_log_files'] += 1
            continue
        
        print(f"  📄 找到日志文件: {eval_log.relative_to(level1_dir)}")
        stats['has_log_files'] += 1
        
        # 解析性能数据
        perf_data = parse_performance_log(eval_log)
        if perf_data is None:
            print(f"  ❌ 无法解析性能数据")
            stats['parse_error'] += 1
            continue
        
        stats['successful_parse'] += 1
        
        triton_time, cuda_time, pytorch_time = perf_data
        print(f"  📊 性能数据 - Triton: {triton_time}ms, CUDA: {cuda_time}ms, PyTorch: {pytorch_time}ms")
        
        # 计算相对于CUDA的加速比
        if cuda_time > 0:
            triton_speedup = cuda_time / triton_time
            cuda_speedup = 1.0
            pytorch_speedup = cuda_time / pytorch_time
            
            results.append({
                'case': case_name,
                'triton_time': triton_time,
                'cuda_time': cuda_time, 
                'pytorch_time': pytorch_time,
                'triton_speedup': triton_speedup,
                'cuda_speedup': cuda_speedup,
                'pytorch_speedup': pytorch_speedup
            })
            
            print(f"  🚀 加速比 - Triton: {triton_speedup:.2f}x, CUDA: {cuda_speedup:.2f}x, PyTorch: {pytorch_speedup:.2f}x")
        else:
            print(f"  ❌ CUDA时间为0，无法计算比值")
    
    return results, stats

def plot_performance_comparison(results):
    """绘制性能对比柱状图"""
    if not results:
        print("No valid performance data for plotting")
        return
    
    # 设置字体为英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    case_names = [r['case'] for r in results]
    triton_speedups = [r['triton_speedup'] for r in results]
    cuda_speedups = [r['cuda_speedup'] for r in results]
    pytorch_speedups = [r['pytorch_speedup'] for r in results]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.8), 8))
    
    x = np.arange(len(case_names))
    width = 0.25
    
    # 绘制柱状图 - CUDA基准放在左边
    bars1 = ax.bar(x - width, cuda_speedups, width, label='CUDA (Baseline)', color='#FF6347', alpha=0.8)
    bars2 = ax.bar(x, triton_speedups, width, label='Triton', color='#2E8B57', alpha=0.8)
    bars3 = ax.bar(x + width, pytorch_speedups, width, label='PyTorch', color='#4169E1', alpha=0.8)
    
    # 设置图表属性
    ax.set_xlabel('Test Cases', fontsize=12)
    ax.set_ylabel('Speedup Ratio (CUDA = 1.0)', fontsize=12)
    ax.set_title('Level1 Performance Comparison - Speedup vs CUDA Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = f"{model_dir_name}/performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Performance comparison chart saved to: {output_path}")
    
    # 显示统计信息
    print(f"\n📈 Statistics:")
    print(f"Total cases analyzed: {len(results)}")
    print(f"Triton average speedup: {np.mean(triton_speedups):.2f}x")
    print(f"PyTorch average speedup: {np.mean(pytorch_speedups):.2f}x")
    
    # 找出最佳和最差性能
    best_triton_idx = np.argmax(triton_speedups)
    worst_triton_idx = np.argmin(triton_speedups)
    
    print(f"\nTriton best speedup case: {case_names[best_triton_idx]} ({triton_speedups[best_triton_idx]:.2f}x)")
    print(f"Triton worst speedup case: {case_names[worst_triton_idx]} ({triton_speedups[worst_triton_idx]:.2f}x)")
    
    plt.show()

def print_statistics(stats):
    """打印详细的统计信息"""
    print(f"\n" + "="*60)
    print(f"📊 LOGS目录分析统计")
    print(f"="*60)
    
    total = stats['total_cases']
    print(f"🔍 总案例数: {total}")
    
    if total == 0:
        return
    
    # 基础分类
    print(f"\n📁 目录存在情况:")
    print(f"  有logs目录: {stats['has_logs']} ({stats['has_logs']/total*100:.1f}%)")
    print(f"  无logs目录: {stats['no_logs']} ({stats['no_logs']/total*100:.1f}%)")
    
    if stats['has_logs'] > 0:
        print(f"\n🎯 在有logs的案例中:")
        print(f"  有{model_dir_name}目录: {stats['has_target_model']} ({stats['has_target_model']/stats['has_logs']*100:.1f}%)")
        print(f"  无{model_dir_name}目录: {stats['no_target_model']} ({stats['no_target_model']/stats['has_logs']*100:.1f}%)")
        
        if stats['has_target_model'] > 0:
            print(f"\n📄 在有{model_dir_name}目录的案例中:")
            print(f"  有日志文件: {stats['has_log_files']} ({stats['has_log_files']/stats['has_target_model']*100:.1f}%)")
            print(f"  无日志文件: {stats['no_log_files']} ({stats['no_log_files']/stats['has_target_model']*100:.1f}%)")
            
            if stats['has_log_files'] > 0:
                print(f"\n✅ 在有日志文件的案例中:")
                print(f"  解析成功: {stats['successful_parse']} ({stats['successful_parse']/stats['has_log_files']*100:.1f}%)")
                print(f"  解析失败: {stats['parse_error']} ({stats['parse_error']/stats['has_log_files']*100:.1f}%)")
    
    # 关键比例分析
    print(f"\n🎯 关键比例分析:")
    if stats['has_logs'] > 0:
        llm_success_ratio = stats['no_target_model'] / stats['has_logs'] * 100
        print(f"  LLM调用成功但无后续结果: {stats['no_target_model']}/{stats['has_logs']} = {llm_success_ratio:.1f}%")
    
    if stats['has_target_model'] > 0:
        error_ratio = (stats['parse_error'] + stats['no_log_files']) / stats['has_target_model'] * 100
        print(f"  有目录但出错的案例: {stats['parse_error'] + stats['no_log_files']}/{stats['has_target_model']} = {error_ratio:.1f}%")
    
    success_ratio = stats['successful_parse'] / total * 100
    print(f"  总体成功率: {stats['successful_parse']}/{total} = {success_ratio:.1f}%")
    
    print(f"="*60)

def main():
    """主函数"""
    print("🔍 Starting Level1 performance test analysis...")
    
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 检查所有案例
    results, stats = check_cases()
    
    # 输出统计信息
    print_statistics(stats)
    
    if results:
        print(f"\n✅ Successfully collected performance data from {len(results)} cases")
        
        # 绘制性能对比图
        plot_performance_comparison(results)
        
        # 输出详细结果到CSV
        import csv
        csv_path = f"{model_dir_name}/performance_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['case', 'triton_time', 'cuda_time', 'pytorch_time', 
                         'triton_speedup', 'cuda_speedup', 'pytorch_speedup']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"📄 Detailed results saved to: {csv_path}")
        
    else:
        print("\n❌ No valid performance test results found")

if __name__ == "__main__":
    main() 