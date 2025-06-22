# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import torch

def print_gpu_status():
    """Print GPU device status information"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    print(f"\n=== GPU Device Status ===")
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} GPU device(s)")
    
    available_devices = []
    
    cnt = 0
    i = 0
    compute_capability = set()
    while len(available_devices) < device_count and cnt < 10:
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            # 尝试设置设备，如果失败则跳过
            torch.cuda.set_device(i)
            # mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            # mem_cached = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = props.total_memory / 1024**3
            # mem_free = mem_total - mem_cached
            # mem_usage_percent = (mem_cached / mem_total) * 100
            
            print(f"\nGPU {i}: {props.name}")
            compute_capability.add(int(f"{props.major}{props.minor}"))
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  Total memory: {mem_total:.2f}GB")
            # print(f"  Allocated memory: {mem_allocated:.2f}GB")
            # print(f"  Cached memory: {mem_cached:.2f}GB")
            # print(f"  Free memory: {mem_free:.2f}GB")
            # print(f"  Memory usage: {mem_usage_percent:.1f}%")
            
            available_devices.append(i)
            i = i + 1
            
            # 在当前设备上分配tensor并进行加法，检验当前GPU是否可用
            try:
                test_tensor1 = torch.tensor([1.0, 2.0], device=i)
                test_tensor2 = torch.tensor([3.0, 4.0], device=i)
                result = test_tensor1 + test_tensor2
                print(f"  GPU functionality test: PASSED")
                del test_tensor1, test_tensor2, result
                torch.cuda.empty_cache()
            except Exception as test_e:
                print(f"  GPU functionality test: FAILED - {str(test_e)}")
            
        except RuntimeError as e:
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name} [UNAVAILABLE]")
            print(f"  Error: {str(e)}")
            print(f"  Status: Device is busy or inaccessible")
        cnt = cnt + 1
    
    if available_devices:
        current_device = torch.cuda.current_device()
        print(f"\nPyTorch current device: GPU {current_device}")
        print(f"Available devices: {available_devices}")
    else:
        print(f"\nNo GPU devices are currently available for use")
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Compute capability: {sorted(list(compute_capability))}")

if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5"
    import torch
    print_gpu_status()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current device: {current_device}, name: {device_name}") # 全是 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current device: {current_device}, name: {device_name}") # 全是 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current device: {current_device}, name: {device_name}") # 全是 5