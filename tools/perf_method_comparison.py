import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import time
# from tests.f32.fused_softmax.triton_.triton_native import fused_softmax_triton
from torch.utils.cpp_extension import load_inline
from tools.performer import KernelPerfBench

callable_name = "sgemv_k32_cuda"
# CUDA kernel代码
cuda_cpp_kernel_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a)+((b)-1))/(b)

__global__ void sgemv_k32_kernel(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;  // 0~M-1
    if (row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K, warpSize);  // 每个线程需要负责计算的数据个数

    #pragma unroll
    for(int i = 0; i < kIteration; i++){
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if(laneId == 0) y[row] = res;
}
'''

cuda_cpp_wrapper_source = '''
torch::Tensor {callable_name}(torch::Tensor A, torch::Tensor x, torch::Tensor y, int M, int K) {{
    
    dim3 dimGrid(M);
    dim3 dimBlock(32);
    
    sgemv_k32_kernel<<<dimGrid, dimBlock>>>(
        A.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        M, K
    );
    
    return y;
}}
'''.format(callable_name=callable_name)

cpp_source = '''
    torch::Tensor {callable_name}(torch::Tensor A, torch::Tensor x, torch::Tensor y, int M, int K);
'''.format(callable_name=callable_name)

# --- Helper functions for visual width alignment ---
def get_visual_width(s):
    """
    Calculates the approximate visual width of a string in a terminal,
    treating CJK and full-width characters as 2 units wide.
    """
    width = 0
    for char_code in map(ord, s):
        # CJK Unified Ideographs, Hiragana, Katakana, Hangul Syllables, Fullwidth Forms
        if (0x4E00 <= char_code <= 0x9FFF) or \
           (0x3040 <= char_code <= 0x30FF) or \
           (0xAC00 <= char_code <= 0xD7A3) or \
           (0xFF00 <= char_code <= 0xFFEF) or \
           (0x1100 <= char_code <= 0x11FF) or \
           (0x3130 <= char_code <= 0x318F): # Adding Hangul Jamo
            width += 2
        else:
            width += 1
    return width

def pad_str_to_visual_width(s, target_visual_width):
    """Pads a string with spaces to achieve a target visual width."""
    current_visual_width = get_visual_width(s)
    padding_spaces = target_visual_width - current_visual_width
    if padding_spaces < 0:
        padding_spaces = 0 # Avoid negative padding if string is already wider
    return s + ' ' * padding_spaces

# --- Printing functions ---
_global_separator_length = 100 # Default, will be updated

def print_separator(char="=", length=None):
    """打印分隔符"""
    global _global_separator_length
    print(char * (length if length is not None else _global_separator_length))


def print_results_table(results, quantiles=[0.2, 0.5, 0.8]):
    """以表格形式打印结果"""
    global _global_separator_length
    print_separator("=")
    print("📊 时间测量结果对比表")
    print_separator("=")
    
    target_visual_width_method_name = 35 
    target_visual_width_numeric = 18     

    method_names_map = {
        '_method1': '批量执行+单次事件',
        '_method2': '逐次测量+最后同步',
        '_method3': '逐次测量+逐次同步',
        '_method4': '逐次测量+最后同步+缓存清理',
        '_method5': '逐次测量+逐次同步+缓存清理'
    }

    header_method_name = '方法名称'
    header_avg = '平均时间(ms)'
    header_median = '中位数(ms)'
    header_var = '方差'
    
    header_parts = [
        pad_str_to_visual_width(header_method_name, target_visual_width_method_name),
        pad_str_to_visual_width(header_avg, target_visual_width_numeric),
        pad_str_to_visual_width(header_median, target_visual_width_numeric),
        pad_str_to_visual_width(header_var, target_visual_width_numeric),
    ]
    for q in quantiles:
        header_parts.append(pad_str_to_visual_width(f'q{int(q*100)} (ms)', target_visual_width_numeric))
    
    _global_separator_length = sum(get_visual_width(part) for part in header_parts) # Calculate from actual header parts visual width
    # Add a little extra for potential minor discrepancies or if join adds spaces (it doesn't here)
    _global_separator_length = max(_global_separator_length, 95)


    print("".join(header_parts)) 
    print_separator("-") 
    
    for method_key, result_data in results.items():
        method_name_str = method_names_map.get(method_key, method_key)
        
        # Ensure result_data is not None and keys exist
        avg_val = result_data.get('mean', 0.0)
        median_val = result_data.get('median', 0.0)
        var_val = result_data.get('var', 0.0)

        avg_str = f"{avg_val:.4f}"
        median_str = f"{median_val:.4f}"
        var_str = f"{var_val:.6f}"

        row_parts = [
            pad_str_to_visual_width(method_name_str, target_visual_width_method_name),
            pad_str_to_visual_width(avg_str, target_visual_width_numeric),
            pad_str_to_visual_width(median_str, target_visual_width_numeric),
            pad_str_to_visual_width(var_str, target_visual_width_numeric),
        ]
        for q in quantiles:
            q_key = f'q{int(q*100)}'
            value = result_data.get(q_key, 0.0) # Default to 0.0 if key missing
            value_str = f"{value:.4f}"
            row_parts.append(pad_str_to_visual_width(value_str, target_visual_width_numeric))
        
        print("".join(row_parts))
    
    print_separator("=")


def print_analysis(results):
    """打印分析结果"""
    global _global_separator_length
    print("\n🔍 分析结果:")
    print_separator("-")
    
    avg_times = {k: v.get('mean', float('inf')) for k, v in results.items() if v} # Use inf for missing avg
    if not avg_times or all(t == float('inf') for t in avg_times.values()):
        print("没有可供分析的结果或所有平均时间无效。")
        print_separator("-")
        return

    # Filter out invalid times before finding min/max
    valid_avg_times = {k: t for k, t in avg_times.items() if t != float('inf')}
    if not valid_avg_times:
        print("所有平均时间均无效。")
        print_separator("-")
        return

    fastest_method_key = min(valid_avg_times, key=valid_avg_times.get)
    slowest_method_key = max(valid_avg_times, key=valid_avg_times.get)
    
    method_names_map = {
        '_method1': '批量执行+单次事件',
        '_method2': '逐次测量+最后同步',
        '_method3': '逐次测量+逐次同步',
        '_method4': '逐次测量+最后同步+缓存清理',
        '_method5': '逐次测量+逐次同步+缓存清理'
    }
    
    fastest_method_name = method_names_map.get(fastest_method_key, fastest_method_key)
    slowest_method_name = method_names_map.get(slowest_method_key, slowest_method_key)

    print(f"⚡ 最快方法: {fastest_method_name} ({valid_avg_times[fastest_method_key]:.4f} ms)")
    print(f"🐌 最慢方法: {slowest_method_name} ({valid_avg_times[slowest_method_key]:.4f} ms)")
    
    fastest_time = valid_avg_times[fastest_method_key]
    slowest_time = valid_avg_times[slowest_method_key]

    if fastest_time > 0 and fastest_time != float('inf'):
        diff_percent = ((slowest_time - fastest_time) / fastest_time) * 100
        print(f"📈 性能差异: {diff_percent:.2f}%")
    else:
        print("📈 性能差异: N/A (最快时间为0或无效)")

    print("\n📊 稳定性分析 (方差越小越稳定):")
    # Use target_visual_width_method_name for consistency
    target_visual_width_method_name_for_analysis = 35 
    
    variance_data = [(k, v.get('var', -1)) for k, v in results.items() if v and v.get('var', -1) >= 0]
    if variance_data:
        variance_data.sort(key=lambda x: x[1])
        for method_key, variance in variance_data:
            method_display_name = method_names_map.get(method_key, method_key)
            # Apply padding for alignment
            padded_name = pad_str_to_visual_width(method_display_name, target_visual_width_method_name_for_analysis)
            print(f"  {padded_name}: {variance:.6f}")
    else:
        print("  没有有效的方差数据进行稳定性分析。")
    
    print_separator("-")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print_separator("=")
    print("🚀 CUDA Kernel 时间测量方法比较工具")
    print_separator("=")
    di = torch.cuda
    if not di.is_available():
        print("❌ CUDA不可用!")
        return
    
    device = di.current_device()
    print(f"🖥️  使用设备: {di.get_device_name(device)}")
    try:
        device_props = di.get_device_properties(device)
        print(f"💾 设备内存: {device_props.total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"无法获取设备属性: {e}")

    try:
        x = torch.randn((4096, 4096), device='cuda', dtype=torch.float32)
    except Exception as e:
        print(f"❌ 创建测试张量失败: {e}")
        return
        
    warmup_rounds = 1000
    test_iterations = 10000
    quantiles = [0.2, 0.5, 0.8]
    
    print(f"⚙️  测试配置:")
    print(f"  • 预热轮数: {warmup_rounds}")
    print(f"  • 测试迭代: {test_iterations}")
    
    # test_func = fused_softmax_triton
    # args = (x,)
    M = 1024 * 16
    K = 32 * 16
    A = torch.rand(M * K, dtype=torch.float32, device=device).reshape(M, K) / K
    x = torch.rand(K, dtype=torch.float32, device=device)
    y = torch.zeros(M, dtype=torch.float32, device=device)
    os.makedirs("./build_sgemv_k32", exist_ok=True)
    sgemv_module = load_inline(
        name="sgemv_k32",
        cpp_sources=cpp_source,
        cuda_sources=cuda_cpp_kernel_source + cuda_cpp_wrapper_source,
        functions=["sgemv_k32_cuda"],
        verbose=True,
        build_directory="./build_sgemv_k32"
    )
    test_func = getattr(sgemv_module, callable_name)
    args = (A, x, y, M, K)
    
    
    print("\n🔬 开始测试...")
    print_separator("-")
    
    results = {}
    # kernel_perf_bench = KernelPerfBench()
    kernel_perf_bench = KernelPerfBench
    methods_to_test = [
        ("方法1: 批量执行+单次事件", kernel_perf_bench._method1, '_method1'),
        ("方法2: 逐次测量+最后同步", kernel_perf_bench._method2, '_method2'),
        ("方法3: 逐次测量+逐次同步", kernel_perf_bench._method3, '_method3'),
        ("方法4: 逐次测量+最后同步+缓存清理", kernel_perf_bench._method4, '_method4'),
        ("方法5: 逐次测量+逐次同步+缓存清理", kernel_perf_bench._method5, '_method5'),
    ]
    
    for i, (desc, method_func, method_key) in enumerate(methods_to_test, 1):
        print(f"🔄 [{i}/{len(methods_to_test)}] 正在测试 {desc}...")
        try:
            start_time_cpu = time.time()
            result = method_func(test_func, args, warmup_rounds, test_iterations, quantiles)
            end_time_cpu = time.time()
            print(f"   ⏰ CPU计时测试耗时: {end_time_cpu - start_time_cpu:.4f} s")
            results[method_key] = result
            if result and 'mean' in result and result['mean'] is not None : # Check for None
                 print(f"   ✅ 完成 - 平均GPU时间: {result['mean']:.4f} ms")
            else:
                 print(f"   ⚠️ 完成但未获取到有效的平均时间。")
        except Exception as e:
            print(f"   ❌ 测试 {desc} 时出错: {e}")
            # Ensure a dictionary with expected keys is stored for error cases
            error_result = {'mean': 0.0, 'median': 0.0, 'var': -1.0} # var -1 to indicate error
            for q_val in quantiles:
                error_result[f'q{int(q_val*100)}'] = 0.0
            results[method_key] = error_result
    test_func = torch.mv
    from utils.bench import bench_simple_perf
    result = bench_simple_perf(test_func, (A, x), warmup_rounds, test_iterations, quantiles)
    results['torch.mv'] = result

    print("\n")
    print_results_table(results, quantiles)
    print_analysis(results)
    
    print("\n🎯 建议:")
    print("• 如果追求最高精度，建议使用逐次测量+逐次同步 (方法3或5)。")
    print("• 如果追求测试速度，可以使用批量执行+单次事件 (方法1)，但注意其对分位数测量的局限性。")
    print("• 缓存清理 (方法4和5) 可以提高测量一致性，但会增加开销。")

if __name__ == "__main__":
    main()