
from eval_.common.loader import load_cuda_extension_from_cufile
from eval_.common.config import EvalConfig
def build_kernel(file_name, header_lib_path="./", build_dir="./build"):
    print("🔧 build_kernel\t:"+file_name)
    print("🔧 build_dir\t:"+build_dir)
    
    # 配置构建参数
    config = EvalConfig()
    config.build_dir = build_dir
    config.cuda_kernel_name = file_name
    
    # CUDA编译标志
    cuda_flags = {
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-std=c++17",
        # "-Idsl_template.cuh",
        "-I" + header_lib_path,  # 包含utils头文件
        # "-DNO_MMA_HGEMM_BIN",  # 启用PyTorch绑定而不是独立二进制
    }
    config.extra_cuda_cflags = list(set(config.extra_cuda_cflags) | cuda_flags)
    
    # 链接标志
    link_flags = [
        "-lcublas",  # 链接cuBLAS库
        "-L/usr/local/cuda/lib64",  # CUDA库路径
    ]
    config.extra_ldflags.extend(link_flags)
    
    # 加载内核
    lib = load_cuda_extension_from_cufile(
        cuda_file=f"{file_name}.cu",
        # cuda_file=[f"{file_name}.cu", "dsl_template.cuh"],
        config=config,
    )
    
    if isinstance(lib, str):
        print(f"❌ build_kernel: {lib}")
        return None
    
    print("✅ build_kernel: success")
    return lib
