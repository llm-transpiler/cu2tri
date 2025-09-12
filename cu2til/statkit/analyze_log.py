#!/usr/bin/env python3
"""
Log analyzer script for CUDA to Triton translation testing.
Extracts correct statistics from old log files, ignoring the potentially incorrect final summary.
"""

import re
import json
import yaml
from pathlib import Path
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime
import sys
import io

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    missing_packages = str(e)


def extract_timestamp_from_path(file_path):
    """Extract timestamp from file path. Format: YYYYMMDD_HHMMSS"""
    path_str = str(file_path)
    
    # Look for timestamp pattern in the path
    timestamp_pattern = r'(\d{8}_\d{6})'
    matches = re.findall(timestamp_pattern, path_str)
    
    if matches:
        # Return the last timestamp found in the path
        return matches[-1]
    else:
        raise ValueError(f"No timestamp found in path: {file_path}. Expected format: YYYYMMDD_HHMMSS")


def get_default_statistics_dir(log_file_path):
    """Get default statistics directory based on log file location."""
    log_path = Path(log_file_path).absolute()
    
    # Find dev_ directory in the path
    for parent in log_path.parents:
        if parent.name == 'dev_':
            return parent / "statistics_results"
    
    # If no dev_ found, look for trans/dev_ pattern
    path_parts = log_path.parts
    for i, part in enumerate(path_parts):
        if part == 'dev_' and i > 0 and path_parts[i-1] == 'trans':
            dev_path = Path(*path_parts[:i+1])
            return dev_path / "statistics_results"
    
    # Fallback to current directory
    return Path("statistics_results")


def detect_log_format(content):
    """
    Automatically detect the log format style.
    Returns 'comprehensive' for new format with case names, 'compact' for old format.
    """
    # Check for new format pattern (case_type/case_name with round info)
    comprehensive_pattern = r'🔹+ (\w+/[\w_]+): ([✅❌]) (SUCCESS|FAILED) \(Round \d+\) 🔹+'
    if re.search(comprehensive_pattern, content):
        return 'comprehensive'
    
    # Check for old format pattern (case_type only)
    compact_pattern = r'🔹+ (\w+): ([✅❌]) (SUCCESS|FAILED) 🔹+'
    if re.search(compact_pattern, content):
        return 'compact'
    
    # Default to compact for backward compatibility
    return 'compact'

def parse_log_file(log_file_path, log_format='auto'):
    """
    Parse the log file and extract test case results.
    Returns a detailed analysis ignoring the final summary.
    
    Args:
        log_file_path: Path to the log file
        log_format: 'auto', 'compact', or 'comprehensive'
    """
    print(f"📖 Analyzing log file: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Auto-detect format if requested
    if log_format == 'auto':
        log_format = detect_log_format(content)
        print(f"🔍 Auto-detected log format: {log_format}")
    else:
        print(f"📝 Using specified log format: {log_format}")
    
    # Extract model name from first line
    model_match = re.search(r'Using (\w+) model: (.+)', content)
    model_info = {
        "model_type": model_match.group(1) if model_match else "unknown",
        "model_name": model_match.group(2) if model_match else "unknown",
        "log_format": log_format
    }
    
    # Find all test case starts and results
    case_pattern = r'🎯 Testing case type: (\w+)\s*\n📁 Case name: ([^\n]+)'
    test_round_pattern = r'✅ Test round (\d+) PASSED! Triton kernel is working correctly\.'
    max_rounds_pattern = r'❌ Maximum rounds \((\d+)\) reached\. Manual intervention required\.'
    
    # Choose pattern based on format
    if log_format == 'comprehensive':
        case_result_pattern = r'🔹+ ([\w/]+): ([✅❌]) (SUCCESS|FAILED) \(Round (\d+)\) 🔹+'
    else:  # compact format
        case_result_pattern = r'🔹+ (\w+): ([✅❌]) (SUCCESS|FAILED) 🔹+'
    
    # Find all cases
    case_matches = re.findall(case_pattern, content)
    case_results = re.findall(case_result_pattern, content)
    
    print(f"📊 Found {len(case_matches)} individual test cases")
    
    # Build detailed results
    detailed_results = defaultdict(list)
    all_case_results = []
    
    # Split content by case boundaries for detailed analysis  
    if log_format == 'comprehensive':
        # Comprehensive format includes case names in section headers
        case_sections = re.split(r'🔹{20} Starting [\w/]+ 🔹{20}', content)
    else:
        # Compact format only has case types in section headers
        case_sections = re.split(r'🔹{20} Starting \w+ 🔹{20}', content)
    
    case_idx = 0
    for section in case_sections[1:]:  # Skip first empty section
        if case_idx >= len(case_matches):
            break
            
        case_type, case_name = case_matches[case_idx]
        
        # Find successful round
        success_matches = re.findall(test_round_pattern, section)
        max_rounds_matches = re.findall(max_rounds_pattern, section)
        
        success_round = None
        
        # Try to get round info from case result pattern first (more accurate for comprehensive format)
        case_result_match = None
        if log_format == 'comprehensive' and case_idx < len(case_results):
            # For comprehensive format, extract round info from the result line
            result_match = case_results[case_idx]
            if len(result_match) >= 4:  # (case_type/case_name, emoji, status, round)
                case_result_match = result_match
                _, _, status, round_str = result_match[:4]
                success = status == "SUCCESS"
                rounds = int(round_str) if round_str.isdigit() else None
        
        # Fallback to section analysis if no result match found
        if case_result_match is None:
            if success_matches:
                # Case succeeded
                success_round = int(success_matches[-1])  # Take the last success round
                success = True
                rounds = success_round
            elif max_rounds_matches:
                # Case failed after max rounds
                max_rounds = int(max_rounds_matches[0])
                success = False
                rounds = max_rounds
            else:
                # Fallback: check for any test round failures
                failure_matches = re.findall(r'❌ Test round (\d+) FAILED\.', section)
                if failure_matches:
                    success = False
                    rounds = int(failure_matches[-1])
                else:
                    # Unknown case - mark as error
                    success = False
                    rounds = None
        
        case_result = {
            "case_type": case_type,
            "case_name": case_name,
            "success": success,
            "rounds": rounds
        }
        
        detailed_results[case_type].append(case_result)
        all_case_results.append(case_result)
        
        case_idx += 1
    
    return model_info, detailed_results, all_case_results


def generate_statistics(detailed_results, all_case_results):
    """Generate comprehensive statistics from the parsed results."""
    
    total_cases = len(all_case_results)
    successful_cases = [r for r in all_case_results if r["success"]]
    total_success = len(successful_cases)
    
    # Case type level statistics
    case_type_stats = {}
    for case_type, case_results in detailed_results.items():
        case_type_success = sum(1 for result in case_results if result["success"])
        case_type_total = len(case_results)
        case_type_stats[case_type] = {
            "total": case_type_total,
            "success": case_type_success,
            "failed": case_type_total - case_type_success,
            "success_rate": case_type_success / case_type_total if case_type_total > 0 else 0,
            "status": "✅" if case_type_success == case_type_total else "❌" if case_type_success == 0 else "⚠️"
        }
    
    # Rounds distribution for successful cases
    rounds_distribution = {}
    for result in successful_cases:
        round_num = result["rounds"]
        if round_num is not None:
            rounds_distribution[round_num] = rounds_distribution.get(round_num, 0) + 1
    
    avg_rounds = sum(result["rounds"] for result in successful_cases if result["rounds"] is not None) / len([r for r in successful_cases if r["rounds"] is not None]) if successful_cases else 0
    
    # Overall statistics
    case_types_fully_passed = sum(1 for stats in case_type_stats.values() if stats["status"] == "✅")
    
    return {
        "overall": {
            "total_cases": total_cases,
            "successful_cases": total_success,
            "failed_cases": total_cases - total_success,
            "success_rate": total_success / total_cases if total_cases > 0 else 0,
            "case_types_total": len(detailed_results),
            "case_types_fully_passed": case_types_fully_passed,
            "avg_rounds_for_success": round(avg_rounds, 2)
        },
        "case_type_stats": case_type_stats,
        "rounds_distribution": rounds_distribution
    }


def print_detailed_summary(model_info, detailed_results, statistics):
    """Print a detailed summary in the new format."""
    
    print(f"\n{'='*80}")
    print(f"📊 DETAILED BATCH TESTING SUMMARY")
    print(f"🤖 Model: {model_info['model_type']} - {model_info['model_name']}")
    print(f"{'='*80}")
    
    # Case type level summary with individual cases
    for case_type, case_results in detailed_results.items():
        stats = statistics["case_type_stats"][case_type]
        print(f"{stats['status']} {case_type:<15} ({stats['success']}/{stats['total']})")
        
        # Individual case details
        for result in case_results:
            if result["success"]:
                status = f"  ✅ (Round {result['rounds']})"
            else:
                rounds_info = f"Round {result['rounds']}" if result.get('rounds') is not None else "Error"
                status = f"  ❌ ({rounds_info})"
            
            print(f"{status} {result['case_name']}")
        print()
    
    # Rounds distribution
    print(f"📊 Success rounds distribution:")
    for round_num in sorted(statistics["rounds_distribution"].keys()):
        count = statistics["rounds_distribution"][round_num]
        print(f"   Round {round_num}: {count} cases")
    
    print(f"📈 Average rounds for successful cases: {statistics['overall']['avg_rounds_for_success']}")
    
    print(f"{'='*80}")
    print(f"🏆 OVERALL TOTAL: {statistics['overall']['successful_cases']}/{statistics['overall']['total_cases']} individual cases succeeded")
    print(f"📊 Case type success rate: {statistics['overall']['case_types_fully_passed']}/{statistics['overall']['case_types_total']} case types fully passed")
    print(f"📈 Overall success rate: {statistics['overall']['success_rate']:.1%}")
    print(f"{'='*80}")


def save_results(output_path, model_info, detailed_results, all_case_results, statistics, format='json'):
    """Save results to JSON or YAML file."""
    
    # Prepare data for serialization
    export_data = {
        "metadata": {
            "model_info": model_info,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_cases": len(all_case_results)
        },
        "statistics": statistics,
        "detailed_results": dict(detailed_results),
        "all_case_results": all_case_results
    }
    
    if format.lower() == 'yaml':
        output_file = output_path.with_suffix('.yaml')
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True, indent=2)
    else:
        output_file = output_path.with_suffix('.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    return output_file


def generate_summary_table(statistics):
    """Generate a summary table of overall statistics."""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    overall = statistics["overall"]
    
    # Prepare data for table
    summary_data = [
        ["Total Test Cases", overall["total_cases"]],
        ["Successful Cases", overall["successful_cases"]],
        ["Failed Cases", overall["failed_cases"]],
        ["Overall Success Rate", f"{overall['success_rate']:.1%}"],
        ["Total Case Types", overall["case_types_total"]],
        ["Fully Successful Case Types", overall["case_types_fully_passed"]],
        ["Average Success Rounds", overall["avg_rounds_for_success"]]
    ]
    
    table = tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid")
    return table


def generate_case_type_table(statistics):
    """Generate a detailed table of case type statistics."""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    case_type_stats = statistics["case_type_stats"]
    
    # Prepare data for table
    table_data = []
    for case_type, stats in case_type_stats.items():
        table_data.append([
            stats["status"],
            case_type,
            stats["success"],
            stats["total"],
            f"{stats['success_rate']:.1%}",
            stats["failed"]
        ])
    
    # Sort by success rate descending
    table_data.sort(key=lambda x: float(x[4].rstrip('%')), reverse=True)
    
    headers = ["Status", "Case Type", "Success", "Total", "Success Rate", "Failed"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    return table


def generate_charts(model_info, statistics, all_case_results, output_dir):
    """Generate various charts for visualization."""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Visualization packages not available. Install matplotlib, seaborn, pandas, and tabulate to enable charts.")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    generated_files = []
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Complete Rounds Distribution Chart (including failed cases)
    rounds_dist = statistics["rounds_distribution"]
    if rounds_dist:
        plt.figure(figsize=(10, 6))
        rounds = sorted(rounds_dist.keys())
        counts = [rounds_dist[r] for r in rounds]
        
        # Add failed cases as a separate bar
        failed_cases = statistics["overall"]["failed_cases"]
        labels = [f'Round {r}' for r in rounds]  # Convert all to string labels
        
        if failed_cases > 0:
            counts = list(counts) + [failed_cases]
            labels = labels + ['Failed']
            # Color scheme: viridis for success rounds, gray for failed
            success_colors = sns.color_palette("viridis", len(sorted(rounds_dist.keys())))
            colors = list(success_colors) + ['#808080']  # Gray for failed
        else:
            colors = sns.color_palette("viridis", len(rounds))
        
        # Use position indices for x-axis
        x_pos = range(len(labels))
        bars = plt.bar(x_pos, counts, alpha=0.8, color=colors)
        plt.xlabel('Round')
        plt.ylabel('Cases')
        plt.title(f'Complete Distribution by Round - {model_info["model_name"]}')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(x_pos, labels, rotation=45 if failed_cases > 0 else 0)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        chart_file = output_dir / "rounds_distribution.png"
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(chart_file)
        print(f"📊 Rounds distribution chart saved to: {chart_file}")
    
    # 2. Case Type Success Rate Chart
    case_type_stats = statistics["case_type_stats"]
    df = pd.DataFrame.from_dict(case_type_stats, orient='index')
    df = df.sort_index(ascending=False)  # Sort by case type name in reverse order for correct display
    
    plt.figure(figsize=(12, 8))
    colors = ['#ff4444' if rate < 0.5 else '#ffaa44' if rate < 1.0 else '#44ff44' 
              for rate in df['success_rate']]
    
    bars = plt.barh(df.index, df['success_rate'], color=colors, alpha=0.8)
    plt.xlabel('Success Rate')
    plt.ylabel('Case Type')
    plt.title(f'Success Rate by Case Type - {model_info["model_name"]}')
    plt.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, df['success_rate'])):
        plt.text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1%}', va='center', ha='left')
    
    plt.xlim(0, 1.1)
    chart_file = output_dir / "case_type_success_rates.png"
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(chart_file)
    print(f"📊 Case type success rates chart saved to: {chart_file}")
    
    # 3. Overall Statistics Multi-Chart Layout  
    overall = statistics["overall"]
    fig = plt.figure(figsize=(20, 12))
    
    # First row: Main pie charts
    # Success/Failure pie chart
    sizes = [overall["successful_cases"], overall["failed_cases"]]
    labels = ['Success', 'Failed']
    colors = ['#44ff44', '#ff4444']
    explode = (0.05, 0)  # explode the success slice
    
    plt.subplot(2, 4, 1)
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=explode)
    plt.title('Overall Success/Failure Distribution', fontsize=12)
    
    # Case types pie chart
    case_types_success = overall["case_types_fully_passed"]
    case_types_partial = len([stats for stats in case_type_stats.values() 
                             if 0 < stats["success_rate"] < 1.0])
    case_types_failed = len([stats for stats in case_type_stats.values() 
                            if stats["success_rate"] == 0])
    
    plt.subplot(2, 4, 2)
    sizes2 = [case_types_success, case_types_partial, case_types_failed]
    labels2 = ['Fully Successful', 'Partially Successful', 'Failed']
    colors2 = ['#44ff44', '#ffaa44', '#ff4444']
    
    plt.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.0f', 
            startangle=90)
    plt.title('Case Type Status Distribution', fontsize=12)
    
    # Second row: Detailed round analysis
    # Rounds distribution as donut chart (including failed cases)
    plt.subplot(2, 4, 5)
    if rounds_dist:
        # Include failed cases in the distribution
        sorted_rounds = sorted(rounds_dist.keys())  # Sort rounds numerically
        sorted_values = [rounds_dist[r] for r in sorted_rounds]  # Get values in sorted order
        
        # Add failed cases as a separate segment
        failed_cases = statistics["overall"]["failed_cases"]
        if failed_cases > 0:
            sorted_values.append(failed_cases)
            all_labels = [f'Round {r}' for r in sorted_rounds] + ['Failed']
        else:
            all_labels = [f'Round {r}' for r in sorted_rounds]
        
        # Create color scheme: green to red for success rounds, gray for failed
        if failed_cases > 0:
            success_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.7, len(sorted_rounds)))
            colors_rounds = list(success_colors) + ['#808080']  # Gray for failed
        else:
            colors_rounds = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rounds_dist)))
        
        wedges, texts, autotexts = plt.pie(sorted_values, 
                                           autopct='%1.1f%%', 
                                           startangle=90, 
                                           pctdistance=0.75,
                                           radius=0.8,
                                           colors=colors_rounds,
                                           wedgeprops=dict(width=0.5))  # Donut style
        
        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('black')
            autotext.set_weight('bold')
        
        # Add a legend (sorted by round number, failed at end)
        plt.legend(all_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
        plt.title('Complete Round Distribution\n(Donut Chart)', fontsize=11)
    
    # Rounds distribution as a bar chart (including failed cases)
    plt.subplot(2, 4, 6)
    if rounds_dist:
        rounds = sorted(rounds_dist.keys())  # Sort rounds numerically
        counts = [rounds_dist[r] for r in rounds]  # Get counts in sorted order
        
        # Add failed cases as a separate bar
        failed_cases = statistics["overall"]["failed_cases"]
        labels = [f'R{r}' for r in rounds]  # Use short labels for compact display
        
        if failed_cases > 0:
            counts = list(counts) + [failed_cases]
            labels = labels + ['Failed']
            # Color scheme: green to red for success rounds, gray for failed
            success_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.7, len(sorted(rounds_dist.keys()))))
            colors_rounds = list(success_colors) + ['#808080']  # Gray for failed
        else:
            colors_rounds = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rounds)))
        
        # Use position indices for x-axis
        x_pos = range(len(labels))
        bars = plt.bar(x_pos, counts, color=colors_rounds, alpha=0.8)
        plt.xlabel('Round', fontsize=10)
        plt.ylabel('Cases', fontsize=10)
        plt.title('Complete Round Distribution\n(Bar Chart)', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(x_pos, labels, rotation=45 if failed_cases > 0 else 0, fontsize=9)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontsize=9)
    
    # Summary stats as text - right bottom
    plt.subplot(2, 4, (7, 8))  # Span across bottom right two cells
    plt.axis('off')
    summary_text = f"""Model: {model_info["model_name"]}
    
Overall Statistics:
• Total Cases: {overall["total_cases"]}
• Success: {overall["successful_cases"]} ({overall['success_rate']:.1%})
• Failed: {overall["failed_cases"]}
• Avg Rounds: {overall["avg_rounds_for_success"]:.2f}

Case Type Statistics:
• Total: {overall["case_types_total"]}
• Fully Successful: {case_types_success}
• Partially Successful: {case_types_partial}
• Failed: {case_types_failed}"""
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Test Results Overview - {model_info["model_name"]}', fontsize=16)
    chart_file = output_dir / "overall_statistics.png"
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(chart_file)
    print(f"📊 Overall statistics chart saved to: {chart_file}")
    
    # 4. Round Distribution by Case Type Stacked Bar Chart
    plt.figure(figsize=(14, 10))
    
    # Calculate round distribution for each case type
    case_round_data = {}
    
    for case_type in case_type_stats.keys():
        # Get all cases for this case type
        case_type_results = [r for r in all_case_results if r["case_type"] == case_type]
        total_cases = len(case_type_results)
        
        if total_cases == 0:
            continue
            
        # Count successful cases by round
        round_counts = {}
        for result in case_type_results:
            if result["success"]:
                round_num = result.get("rounds", 1)
                round_counts[round_num] = round_counts.get(round_num, 0) + 1
        
        # Convert to percentages
        round_percentages = {}
        for round_num, count in round_counts.items():
            round_percentages[round_num] = (count / total_cases) * 100
            
        case_round_data[case_type] = round_percentages
    
    # Prepare data for stacked bar chart
    case_types = sorted(case_round_data.keys(), reverse=True)  # Sort for consistent display
    all_rounds = []
    for data in case_round_data.values():
        if data:
            all_rounds.extend(data.keys())
    max_rounds = max(all_rounds) if all_rounds else 5  # Default to 5 if no successful cases
    
    # Create color map for rounds
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, max_rounds))
    
    # Create stacked horizontal bar chart
    y_pos = np.arange(len(case_types))
    
    # Track cumulative widths for stacking
    cumulative_widths = np.zeros(len(case_types))
    
    for round_num in range(1, max_rounds + 1):
        widths = []
        for case_type in case_types:
            width = case_round_data[case_type].get(round_num, 0)
            widths.append(width)
        
        plt.barh(y_pos, widths, left=cumulative_widths, 
                color=colors[round_num-1], alpha=0.8, 
                label=f'Round {round_num}')
        
        # Add percentage labels for bars wider than 5%
        for i, (width, cumulative) in enumerate(zip(widths, cumulative_widths)):
            if width > 5:  # Only show label if segment is > 5%
                plt.text(cumulative + width/2, i, f'R{round_num}:  {width:.1f}%', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        cumulative_widths += widths
    
    # Customize the chart
    plt.yticks(y_pos, case_types)
    plt.xlabel('Percentage of Cases', fontsize=12)
    plt.ylabel('Case Type', fontsize=12)
    plt.title(f'Success Round Distribution by Case Type - {model_info["model_name"]}', fontsize=14)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add grid
    plt.grid(axis='x', alpha=0.3)
    
    # Set x-axis limit
    plt.xlim(0, 100)
    
    chart_file = output_dir / "round_distribution_by_case.png"
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    generated_files.append(chart_file)
    print(f"📊 Round distribution by case type chart saved to: {chart_file}")
    
    return generated_files


def generate_detailed_case_table(all_case_results):
    """Generate a detailed table showing individual case results."""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    # Prepare data for detailed case table
    table_data = []
    for result in all_case_results:
        case_type = result["case_type"]
        case_name = result["case_name"]
        success = result["success"]
        rounds = result.get("rounds", "N/A")
        
        # Determine pass status for different rounds
        pass_round_1 = "✅" if success and rounds == 1 else "❌"
        pass_round_5 = "✅" if success and rounds <= 5 else "❌"
        
        table_data.append([
            case_type,
            case_name,
            pass_round_1,
            pass_round_5,
            rounds if rounds is not None else "Error",
            "✅ Success" if success else "❌ Failed"
        ])
    
    # Sort by case type, then by case name
    table_data.sort(key=lambda x: (x[0], x[1]))
    
    headers = ["Case Type", "Case Name", "Pass Round 1", "Pass ≤ Round 5", "Actual Rounds", "Final Status"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    return table


def generate_round_analysis_table(all_case_results):
    """Generate a table analyzing success by round."""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    # Count successes by round
    round_stats = {}
    for i in range(1, 6):  # Rounds 1-5
        round_stats[i] = {
            "pass_this_round": 0,
            "pass_by_this_round": 0
        }
    
    for result in all_case_results:
        if result["success"] and result.get("rounds") is not None:
            actual_round = result["rounds"]
            if 1 <= actual_round <= 5:
                # This case passed in this specific round
                round_stats[actual_round]["pass_this_round"] += 1
                
                # This case passed by this round (cumulative)
                for r in range(actual_round, 6):
                    round_stats[r]["pass_by_this_round"] += 1
    
    table_data = []
    for round_num in range(1, 6):
        stats = round_stats[round_num]
        table_data.append([
            f"Round {round_num}",
            stats["pass_this_round"],
            stats["pass_by_this_round"],
            f"{stats['pass_by_this_round']/len(all_case_results)*100:.1f}%"
        ])
    
    headers = ["Round", "Cases Passed in This Round", "Cumulative Cases Passed", "Cumulative Success Rate"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    return table


def generate_markdown_report(model_info, statistics, detailed_results, all_case_results, output_path, console_output=""):
    """Generate a simplified markdown report with model info and console output."""
    
    markdown_content = f"""# Test Results Analysis Report

## Model Information
- **Model**: {model_info["model_name"]}
- **Model Type**: {model_info["model_type"]}
- **Analysis Timestamp**: {output_path.parent.name}
- **Log File**: {model_info.get("log_file", "N/A")}

## Console Output

```
{console_output.strip()}
```

---
*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""

    # Save markdown report
    markdown_file = output_path.with_suffix('.md')
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"📝 Markdown report saved to: {markdown_file}")
    return markdown_file


def generate_markdown_summary(model_info, statistics, detailed_results, all_case_results, output_path):
    """Generate a concise markdown summary with native markdown tables."""
    
    overall = statistics["overall"]
    case_type_stats = statistics["case_type_stats"]
    rounds_dist = statistics["rounds_distribution"]
    
    # Calculate additional metrics
    round_1_success = len([r for r in all_case_results if r["success"] and r.get("rounds") == 1])
    round_5_success = len([r for r in all_case_results if r["success"] and r.get("rounds", 6) <= 5])
    
    markdown_content = f"""# Test Results Summary

## 📊 Model: {model_info["model_name"]}

**Analysis Timestamp:** {output_path.parent.name}  
**Total Cases:** {overall["total_cases"]} | **Success:** {overall["successful_cases"]} ({overall['success_rate']:.1%}) | **Failed:** {overall["failed_cases"]}

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | {round_1_success}/{overall['total_cases']} | {round_1_success/overall['total_cases']*100:.1f}% |
| **Round ≤5 Success** | {round_5_success}/{overall['total_cases']} | {round_5_success/overall['total_cases']*100:.1f}% |
| **Overall Success** | {overall["successful_cases"]}/{overall["total_cases"]} | {overall['success_rate']:.1%} |
| **Case Types Fully Passed** | {overall["case_types_fully_passed"]}/{overall["case_types_total"]} | {overall["case_types_fully_passed"]/overall["case_types_total"]*100:.1f}% |
| **Avg Rounds for Success** | {overall["avg_rounds_for_success"]:.2f} | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|"""

    cumulative = 0
    for round_num in sorted(rounds_dist.keys()):
        count = rounds_dist[round_num]
        cumulative += count
        rate = cumulative / overall["total_cases"] * 100
        markdown_content += f"\n| Round {round_num} | {count} | {cumulative} | {rate:.1f}% |"

    markdown_content += f"""

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|"""

    # Sort case types by success rate for better readability
    sorted_case_types = sorted(case_type_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True)
    
    for case_type, stats in sorted_case_types:
        if stats["success_rate"] == 1.0:
            status = "✅"
            performance = "Excellent"
        elif stats["success_rate"] >= 0.8:
            status = "🟢"
            performance = "Good"
        elif stats["success_rate"] >= 0.5:
            status = "🟡"
            performance = "Moderate"
        elif stats["success_rate"] > 0:
            status = "⚠️"
            performance = "Poor"
        else:
            status = "❌"
            performance = "Failed"
            
        markdown_content += f"\n| {status} | `{case_type}` | {stats['success_rate']:.1%} | {stats['success']}/{stats['total']} | {performance} |"

    markdown_content += f"""

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
"""
    top_performers = [ct for ct, stats in sorted_case_types if stats['success_rate'] == 1.0]
    if top_performers:
        for ct in top_performers:
            stats = case_type_stats[ct]
            markdown_content += f"- **{ct}**: {stats['success']}/{stats['total']} cases\n"
    else:
        markdown_content += "- *No case types achieved 100% success rate*\n"

    markdown_content += f"""
### Areas for Improvement
"""
    poor_performers = [ct for ct, stats in sorted_case_types if stats['success_rate'] < 0.5]
    if poor_performers:
        for ct in poor_performers:
            stats = case_type_stats[ct]
            markdown_content += f"- **{ct}**: {stats['success']}/{stats['total']} cases ({stats['success_rate']:.1%})\n"
    else:
        markdown_content += "- *All case types achieved ≥50% success rate*\n"

    markdown_content += f"""
### Round Analysis Insights
- **{round_1_success}** cases ({round_1_success/overall['total_cases']*100:.1f}%) succeeded on first attempt
- **{round_5_success - round_1_success}** additional cases succeeded within 5 rounds
- **{overall['failed_cases']}** cases ({overall['failed_cases']/overall['total_cases']*100:.1f}%) failed after maximum rounds

---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | {len([s for s in case_type_stats.values() if s['success_rate'] == 1.0])} | {len([s for s in case_type_stats.values() if s['success_rate'] == 1.0])/len(case_type_stats)*100:.1f}% |
| **Good Case Types** (80-99%) | {len([s for s in case_type_stats.values() if 0.8 <= s['success_rate'] < 1.0])} | {len([s for s in case_type_stats.values() if 0.8 <= s['success_rate'] < 1.0])/len(case_type_stats)*100:.1f}% |
| **Moderate Case Types** (50-79%) | {len([s for s in case_type_stats.values() if 0.5 <= s['success_rate'] < 0.8])} | {len([s for s in case_type_stats.values() if 0.5 <= s['success_rate'] < 0.8])/len(case_type_stats)*100:.1f}% |
| **Poor Case Types** (<50%) | {len([s for s in case_type_stats.values() if s['success_rate'] < 0.5])} | {len([s for s in case_type_stats.values() if s['success_rate'] < 0.5])/len(case_type_stats)*100:.1f}% |

---

*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""

    # Save markdown summary
    summary_file = output_path.parent / f"{output_path.stem}_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"📝 Markdown summary saved to: {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(description='Analyze CUDA to Triton translation test logs')
    parser.add_argument('log_file', help='Path to the log file to analyze')
    parser.add_argument('--output', '-o', help='Output file path (without extension)')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', 
                       help='Output format (default: json)')
    parser.add_argument('--no-summary', action='store_true', help='Skip printing detailed summary')
    parser.add_argument('--charts', action='store_true', help='Generate visualization charts (requires matplotlib, seaborn, pandas)')
    parser.add_argument('--tables', action='store_true', help='Print formatted tables (requires tabulate)')
    parser.add_argument('--detailed-tables', action='store_true', help='Generate detailed case-by-case tables')
    parser.add_argument('--markdown', action='store_true', help='Generate markdown report')
    parser.add_argument('--statistics-dir', help='Base directory for statistics results (default: dev_/statistics_results)')
    parser.add_argument('--all-viz', action='store_true', help='Enable charts, tables, detailed tables, and markdown report')
    parser.add_argument('--timestamp', help='Custom timestamp for output directory (default: extracted from log path)')
    parser.add_argument('--log-format', choices=['auto', 'compact', 'comprehensive'], default='auto',
                       help='Log format style: "compact" (case_type only), "comprehensive" (case_type/case_name with rounds), "auto" (auto-detect)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        return 1
    
    # Parse log file
    try:
        model_info, detailed_results, all_case_results = parse_log_file(log_path, args.log_format)
        # Add log file info to model_info
        model_info["log_file"] = str(log_path)
    except Exception as e:
        print(f"❌ Error parsing log file: {e}")
        return 1
    
    # Generate statistics
    statistics = generate_statistics(detailed_results, all_case_results)
    
    # Handle visualization options
    enable_charts = args.charts or args.all_viz
    enable_tables = args.tables or args.all_viz
    enable_detailed_tables = args.detailed_tables or args.all_viz
    enable_markdown = args.markdown or args.all_viz
    
    # Check if visualization packages are available when needed
    if (enable_charts or enable_tables or enable_detailed_tables or enable_markdown) and not VISUALIZATION_AVAILABLE:
        print(f"⚠️ Warning: Visualization packages not available: {missing_packages}")
        print("📦 Install required packages: pip install matplotlib seaborn pandas tabulate")
        enable_charts = False
        enable_tables = False
        enable_detailed_tables = False
        enable_markdown = False
    
    # Get timestamp from log path or use provided one
    try:
        timestamp = args.timestamp or extract_timestamp_from_path(log_path)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Please provide a timestamp using --timestamp YYYYMMDD_HHMMSS")
        return 1
    
    # Clean model name for directory usage
    model_name_clean = "".join(c if c.isalnum() else "_" for c in model_info["model_name"].split("/")[-1].lower())
    
    # Determine base directory
    if args.output:
        output_path = Path(args.output)
        base_dir = output_path.parent
    else:
        # Get default statistics directory
        if args.statistics_dir:
            base_statistics_dir = Path(args.statistics_dir)
        else:
            base_statistics_dir = get_default_statistics_dir(log_path)
        
        base_dir = base_statistics_dir / model_name_clean / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{log_path.stem}_analysis"
        print(f"📁 Results will be saved to: {base_dir}")
    
    # Capture console output
    console_output = io.StringIO()
    original_stdout = sys.stdout
    
    try:
        # Start capturing output for tables and detailed output (excluding summary)
        if enable_tables or enable_detailed_tables or enable_markdown:
            sys.stdout = console_output
        
        # Generate and display tables
        if enable_tables:
            print(f"\n{'='*80}")
            print(f"📊 Overall Statistics Table")
            print(f"{'='*80}")
            summary_table = generate_summary_table(statistics)
            if summary_table:
                print(summary_table)
            
            print(f"\n{'='*80}")
            print(f"📊 Case Type Statistics Table")
            print(f"{'='*80}")
            case_type_table = generate_case_type_table(statistics)
            if case_type_table:
                print(case_type_table)
        
        # Generate and display detailed tables
        if enable_detailed_tables:
            print(f"\n{'='*80}")
            print(f"📋 Detailed Case Results Table")
            print(f"{'='*80}")
            detailed_table = generate_detailed_case_table(all_case_results)
            if detailed_table:
                print(detailed_table)
            
            print(f"\n{'='*80}")
            print(f"📈 Round Analysis Table")
            print(f"{'='*80}")
            round_table = generate_round_analysis_table(all_case_results)
            if round_table:
                print(round_table)
    
    finally:
        # Restore stdout
        sys.stdout = original_stdout
    
    # Print summary unless disabled (this goes to console only, not captured)
    if not args.no_summary:
        print_detailed_summary(model_info, detailed_results, statistics)
    
    # Display what was captured to console as well
    captured_content = console_output.getvalue()
    if captured_content.strip():
        print(captured_content)
    
    # Generate charts
    generated_chart_files = []
    if enable_charts:
        charts_dir = base_dir / "charts"
        print(f"\n📊 Generating visualization charts...")
        generated_chart_files = generate_charts(model_info, statistics, all_case_results, charts_dir)
    
    # Generate markdown report with captured output
    markdown_file = None
    summary_file = None
    if enable_markdown:
        print(f"\n📝 Generating markdown reports...")
        # Combine summary and captured output for markdown
        summary_output = io.StringIO()
        original_stdout_temp = sys.stdout
        sys.stdout = summary_output
        try:
            if not args.no_summary:
                print_detailed_summary(model_info, detailed_results, statistics)
        finally:
            sys.stdout = original_stdout_temp
        
        full_console_output = summary_output.getvalue() + captured_content
        markdown_file = generate_markdown_report(model_info, statistics, detailed_results, all_case_results, output_path, full_console_output)
        
        # Generate native markdown summary
        summary_file = generate_markdown_summary(model_info, statistics, detailed_results, all_case_results, output_path)
    
    # Save JSON/YAML results
    try:
        saved_file = save_results(output_path, model_info, detailed_results, all_case_results, statistics, args.format)
        
        # Summary of generated files
        print(f"\n{'='*80}")
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Generated files in: {base_dir}")
        print(f"  📄 Raw results: {saved_file}")
        
        if markdown_file:
            print(f"  📝 Detailed report: {markdown_file}")
        
        if summary_file:
            print(f"  📋 Summary report: {summary_file}")
        
        if generated_chart_files:
            print(f"  📊 Charts ({len(generated_chart_files)} files):")
            for chart_file in generated_chart_files:
                print(f"    • {chart_file.name}")
        
        if enable_tables:
            print(f"  📋 Basic tables: Displayed in console output")
        
        if enable_detailed_tables:
            print(f"  📋 Detailed tables: Displayed in console output")
        
        print(f"\n💡 Quick access:")
        print(f"  📂 Open directory: {base_dir}")
        if summary_file:
            print(f"  📊 View summary: {summary_file}")
        if markdown_file:
            print(f"  📖 View detailed report: {markdown_file}")
        
        print(f"{'='*80}")
        return 0
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
