#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log_file(log_file_path):
    """Parse the benchmark log file and extract forward performance data"""
    data = defaultdict(lambda: defaultdict(dict))
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Split into sections for each configuration
    sections = content.split('\n### ')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        # Parse header
        header_match = re.match(r'batch_size = (\d+), headdim = (\d+), causal = (True|False), seqlen = (\d+)', lines[0])
        if not header_match:
            continue
            
        batch_size = int(header_match.group(1))
        headdim = int(header_match.group(2))
        causal = header_match.group(3) == 'True'
        seqlen = int(header_match.group(4))
        
        # Parse performance data
        for line in lines[1:]:
            # Parse forward performance
            fwd_match = re.match(r'(\w+) fwd: ([\d.]+)ms, ([\d.]+) TFLOPS', line)
            if fwd_match:
                method = fwd_match.group(1)
                if method == 'Fav2':
                    method = 'FlashAttention-2'
                elif method == 'Fav3':
                    method = 'FlashAttention-3'
                elif method == 'CuDNN':
                    method = 'cuDNN'
                
                tflops = float(fwd_match.group(3))
                data[(headdim, causal)][seqlen][method] = tflops
    
    return data

def plot_benchmark_results(data, output_dir='.'):
    """Plot benchmark results in 6 subplots"""
    
    # Define sequence lengths and methods
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    methods = ['Standard attention', 'FlashAttention-2', 'Triton', 'cuDNN', 'FlashAttention-3']
    colors = {
        'Standard attention': '#1f77b4',  # Blue
        'FlashAttention-2': '#ff7f0e',    # Orange  
        'Triton': '#2ca02c',              # Green
        'cuDNN': '#d62728',               # Red
        'FlashAttention-3': '#9467bd'     # Purple
    }
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Attention forward speed, head dim 64 (H100 80GB PCIE)', fontsize=16, fontweight='bold', y=0.95)
    
    headdims = [64, 128, 256]
    causal_states = [False, True]
    
    plot_idx = 0
    for causal in causal_states:
        for headdim_idx, headdim in enumerate(headdims):
            ax = axes[0 if not causal else 1, headdim_idx]
            
            # Extract data for this configuration
            config_data = data.get((headdim, causal), {})
            
            x_pos = np.arange(len(seq_lengths))
            width = 0.15
            
            # Plot methods, but skip Standard attention and Triton since they're not in our data
            plot_methods = ['FlashAttention-2', 'cuDNN', 'FlashAttention-3']
            for i, method in enumerate(plot_methods):
                tflops_values = []
                for seqlen in seq_lengths:
                    seqlen_data = config_data.get(seqlen, {})
                    tflops = seqlen_data.get(method, 0)
                    tflops_values.append(tflops)
                
                # Only plot if we have data
                if any(v > 0 for v in tflops_values):
                    bars = ax.bar(x_pos + i * width, tflops_values, width, 
                                 label=method, color=colors[method], alpha=0.9, edgecolor='black', linewidth=0.5)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, tflops_values):
                        if value > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                                   f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Customize subplot
            ax.set_xlabel('Sequence length', fontsize=12)
            ax.set_ylabel('Speed (TFLOPS/s)', fontsize=12)
            causal_text = "with" if causal else "without"
            ax.set_title(f'({chr(97 + plot_idx)}) Forward, {causal_text} causal mask, head dim {headdim}', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(['512', '1k', '2k', '4k', '8k', '16k'])
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 600)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add legend only to the first subplot
            if plot_idx == 0:
                ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
            
            plot_idx += 1
    
    # Update figure title based on the first headdim processed
    fig.suptitle('Attention forward speed (H100 80GB PCIE)', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    plt.savefig(f'{output_dir}/attention_forward_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/attention_forward_benchmark.pdf', bbox_inches='tight')
    print(f"Plots saved to {output_dir}/attention_forward_benchmark.png and .pdf")
    plt.show()

def main():
    log_file = '/workspace/flash-attention/hopper/benchmark_attn.log'
    
    print("Parsing log file...")
    data = parse_log_file(log_file)
    
    print("Creating plots...")
    plot_benchmark_results(data, '/workspace/flash-attention/hopper')
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for (headdim, causal), seq_data in data.items():
        causal_text = "causal" if causal else "non-causal"
        print(f"\nHead dim {headdim}, {causal_text}:")
        for seqlen, method_data in sorted(seq_data.items()):
            print(f"  Seq len {seqlen}:")
            for method, tflops in sorted(method_data.items()):
                print(f"    {method}: {tflops:.1f} TFLOPS")

if __name__ == "__main__":
    main()