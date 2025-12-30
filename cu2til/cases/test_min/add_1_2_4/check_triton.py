#!/usr/bin/env python3
"""Simple test for LLM feedback fix"""

def get_data():
    import torch
    return (torch.randn(1, 2), torch.randn(1, 2))

def triton_ref():
    def add_kernel(A, B, C):
        C[0, 0] = A[0, 0] + B[0, 0]
    return add_kernel

def torch_kernel(A, B):
    return A + B

if __name__ == "__main__":
    A, B = get_data()
    C = torch.zeros_like(A)

    # Test triton kernel
    triton_func = triton_ref()
    triton_func(A, B, C)

    # Test torch kernel
    torch_result = torch_kernel(A, B)

    # Compare
    if torch.allclose(C, torch_result):
        print("✅ Test PASSED")
        exit(0)
    else:
        print("❌ Test FAILED")
        exit(1)
