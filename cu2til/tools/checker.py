import torch
def compare_results(output_torch, output_cuda, atol=1e-3, rtol=1e-2):
    """Compare results from two implementations"""
    # Ensure both tensors are on the same device for comparison
    if output_torch.device != output_cuda.device:
        output_torch = output_torch.to(output_cuda.device)
    
    output_cuda = output_cuda.to(torch.float32)
    output_torch = output_torch.to(torch.float32)
    diff = torch.abs(output_torch - output_cuda)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    rel_err = diff / torch.maximum(torch.abs(output_torch), torch.tensor(1e-9, device=output_torch.device, dtype=torch.float32))
    max_rel_err = torch.max(rel_err)
    mean_rel_err = torch.mean(rel_err)
    
    print(f"Max diff: {max_diff:.2e}")
    print(f"Mean diff: {mean_diff:.2e}")
    print(f"Max rel error: {max_rel_err:.2e}")
    print(f"Mean rel error: {mean_rel_err:.2e}")
    
    if torch.allclose(output_torch, output_cuda, atol=atol, rtol=rtol):
        print(f"✅ Results match")
        return True
    else:
        print(f"❌ Results do not match")
        return False
