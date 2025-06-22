#!/usr/bin/env python3
"""
Triton 3.2 RAG Database
完整的Triton 3.2知识库，包含API文档、最佳实践、常见问题解决方案
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class TritonRAGDatabase:
    """Triton 3.2 RAG知识库"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent / "data" / "triton_rag.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path.parent / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 初始化embedding模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化知识库
        self._init_database()
        self._populate_triton_knowledge()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            self.collection = self.chroma_client.get_collection("triton_knowledge")
        except:
            self.collection = self.chroma_client.create_collection(
                name="triton_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
    
    def _populate_triton_knowledge(self):
        """填充Triton 3.2知识库"""
        if self.collection.count() > 0:
            return  # 已经填充过了
        
        triton_knowledge = self._get_triton_32_knowledge()
        
        documents = []
        metadatas = []
        ids = []
        
        for i, (category, items) in enumerate(triton_knowledge.items()):
            for j, item in enumerate(items):
                doc_id = f"{category}_{j}"
                content = f"Category: {category}\nTitle: {item['title']}\nContent: {item['content']}"
                if 'code' in item:
                    content += f"\nCode Example:\n{item['code']}"
                
                documents.append(content)
                metadatas.append({
                    "category": category,
                    "title": item['title'],
                    "type": item.get('type', 'general')
                })
                ids.append(doc_id)
        
        # 批量添加到向量数据库
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def _get_triton_32_knowledge(self) -> Dict:
        """获取Triton 3.2完整知识库"""
        return {
            "api_reference": [
                {
                    "title": "triton.jit Decorator",
                    "content": "The @triton.jit decorator compiles Python functions into GPU kernels using Triton's compiler infrastructure.",
                    "code": """
@triton.jit
def kernel_function(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation
    pass
                    """,
                    "type": "api"
                },
                {
                    "title": "triton.language Module",
                    "content": "triton.language (tl) provides GPU-specific language constructs for memory operations, arithmetic, and control flow.",
                    "code": """
import triton.language as tl

# Memory operations
data = tl.load(ptr + offsets, mask=mask)
tl.store(ptr + offsets, data, mask=mask)

# Arithmetic operations
result = tl.dot(a, b)
result = tl.sum(data, axis=0)
result = tl.max(data, axis=1)
                    """,
                    "type": "api"
                },
                {
                    "title": "Program ID and Grid",
                    "content": "tl.program_id() returns the current program's ID in the grid. Used for thread block indexing.",
                    "code": """
pid_x = tl.program_id(axis=0)
pid_y = tl.program_id(axis=1) 
pid_z = tl.program_id(axis=2)

# Calculate global thread index
block_start = pid_x * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    """,
                    "type": "api"
                },
                {
                    "title": "Memory Access Patterns",
                    "content": "Efficient memory access using coalesced reads/writes and proper masking.",
                    "code": """
# Coalesced memory access
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

# 2D memory access
row_offsets = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
col_offsets = pid_y * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
ptrs = input_ptr + row_offsets[:, None] * stride + col_offsets[None, :]
                    """,
                    "type": "api"
                }
            ],
            "best_practices": [
                {
                    "title": "Memory Coalescing",
                    "content": "Ensure memory accesses are coalesced for optimal bandwidth utilization. Use contiguous memory patterns.",
                    "code": """
# Good: Coalesced access
offsets = tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)

# Bad: Strided access (avoid when possible)
offsets = tl.arange(0, BLOCK_SIZE) * stride
data = tl.load(ptr + offsets)
                    """,
                    "type": "optimization"
                },
                {
                    "title": "Block Size Selection",
                    "content": "Choose block sizes that are multiples of warp size (32) and optimize for memory hierarchy.",
                    "code": """
# Recommended block sizes
BLOCK_SIZE_M: tl.constexpr = 128
BLOCK_SIZE_N: tl.constexpr = 128  
BLOCK_SIZE_K: tl.constexpr = 32

# For vector operations
BLOCK_SIZE: tl.constexpr = 1024  # Must be power of 2
                    """,
                    "type": "optimization"
                },
                {
                    "title": "Boundary Checking",
                    "content": "Always use masks for boundary checking to handle arbitrary input sizes safely.",
                    "code": """
# Safe boundary checking
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

# Load with mask
data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
# Store with mask  
tl.store(output_ptr + offsets, result, mask=mask)
                    """,
                    "type": "safety"
                }
            ],
            "common_patterns": [
                {
                    "title": "Vector Addition",
                    "content": "Element-wise vector addition pattern with proper masking and coalesced access.",
                    "code": """
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
                    """,
                    "type": "pattern"
                },
                {
                    "title": "Matrix Multiplication",
                    "content": "Tiled matrix multiplication with shared memory optimization.",
                    "code": """
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
                    """,
                    "type": "pattern"
                },
                {
                    "title": "Reduction Operations",
                    "content": "Efficient reduction using Triton's built-in reduction primitives.",
                    "code": """
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Load data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform reduction
    result = tl.sum(data)  # or tl.max(data), tl.min(data), etc.
    
    # Store result (only first thread)
    if pid == 0:
        tl.store(output_ptr, result)
                    """,
                    "type": "pattern"
                }
            ],
            "advanced_features": [
                {
                    "title": "Atomic Operations",
                    "content": "Triton supports atomic operations for thread-safe updates to shared memory locations.",
                    "code": """
# Atomic add
tl.atomic_add(ptr, value)

# Atomic max
tl.atomic_max(ptr, value)

# Atomic compare and swap
tl.atomic_cas(ptr, cmp, val)

# Usage example
@triton.jit
def histogram_kernel(data_ptr, hist_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(data_ptr + offsets, mask=mask)
    # Atomic increment histogram bins
    tl.atomic_add(hist_ptr + data, 1)
                    """,
                    "type": "advanced"
                },
                {
                    "title": "Mixed Precision",
                    "content": "Triton supports mixed precision arithmetic for performance optimization.",
                    "code": """
@triton.jit
def mixed_precision_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load as fp16
    data_fp16 = tl.load(input_ptr + offsets, mask=mask).to(tl.float16)
    
    # Compute in fp32 for accuracy
    data_fp32 = data_fp16.to(tl.float32)
    result_fp32 = tl.exp(data_fp32)  # Example computation
    
    # Store as fp16 for memory efficiency
    result_fp16 = result_fp32.to(tl.float16)
    tl.store(output_ptr + offsets, result_fp16, mask=mask)
                    """,
                    "type": "advanced"
                },
                {
                    "title": "Dynamic Shapes",
                    "content": "Handling dynamic input shapes using tl.cdiv for ceiling division.",
                    "code": """
@triton.jit  
def dynamic_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Handle dynamic shapes
    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE
    
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    
    # Proper boundary checking for dynamic shapes
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]
    
    # 2D indexing
    ptrs = input_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    data = tl.load(ptrs, mask=mask)
    
    # Process and store
    result = data * 2.0
    tl.store(ptrs, result, mask=mask)

# Launch with dynamic grid
def launch_dynamic_kernel(input_tensor, output_tensor):
    M, N = input_tensor.shape
    BLOCK_SIZE = 32
    grid = (tl.cdiv(M, BLOCK_SIZE), tl.cdiv(N, BLOCK_SIZE))
    dynamic_kernel[grid](input_tensor, output_tensor, M, N, BLOCK_SIZE)
                    """,
                    "type": "advanced"
                }
            ],
            "debugging_tips": [
                {
                    "title": "Common Compilation Errors",
                    "content": "Solutions for frequent Triton compilation issues.",
                    "code": """
# Error: Incompatible pointer types
# Solution: Ensure consistent data types
data = tl.load(ptr + offsets).to(tl.float32)  # Explicit cast

# Error: Invalid block size
# Solution: Use constexpr and powers of 2
BLOCK_SIZE: tl.constexpr = 1024  # Must be constexpr

# Error: Out of bounds access
# Solution: Always use proper masking
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
                    """,
                    "type": "debugging"
                },
                {
                    "title": "Performance Debugging",
                    "content": "Tools and techniques for identifying performance bottlenecks.",
                    "code": """
# Use triton profiler
import triton.profiler as profiler

@profiler.profile
@triton.jit
def kernel_to_profile(...):
    # Kernel code
    pass

# Check occupancy
def check_occupancy(kernel_func, *args):
    return kernel_func.occupancy(*args)

# Memory bandwidth analysis
def analyze_bandwidth(kernel_func, input_size, dtype_size):
    # Theoretical bandwidth calculation
    bytes_accessed = input_size * dtype_size * 2  # read + write
    # Compare with measured time to get effective bandwidth
                    """,
                    "type": "debugging"
                }
            ],
            "error_patterns": [
                {
                    "title": "Memory Access Violations",
                    "content": "Common memory access errors and their solutions.",
                    "code": """
# Problem: Buffer overrun
offsets = tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)  # May access out of bounds

# Solution: Always use masking
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# Problem: Misaligned access
# Solution: Ensure proper alignment
assert input_ptr % 16 == 0, "Pointer must be 16-byte aligned"
                    """,
                    "type": "error"
                },
                {
                    "title": "Type Conversion Errors",
                    "content": "Common type-related errors in Triton kernels.",
                    "code": """
# Problem: Implicit type conversion
result = int_data + float_data  # May cause compilation error

# Solution: Explicit type conversion
result = int_data.to(tl.float32) + float_data

# Problem: Precision loss
fp16_result = tl.dot(a.to(tl.float16), b.to(tl.float16))

# Solution: Use mixed precision carefully
fp32_result = tl.dot(a.to(tl.float32), b.to(tl.float32))
fp16_result = fp32_result.to(tl.float16)
                    """,
                    "type": "error"
                }
            ]
        }
    
    def search(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """搜索相关知识"""
        where = {}
        if category:
            where["category"] = category
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where if where else None
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0.0,
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def get_api_reference(self, api_name: str) -> Optional[str]:
        """获取特定API的参考文档"""
        results = self.search(f"API {api_name}", category="api_reference", top_k=1)
        return results[0]['content'] if results else None
    
    def get_best_practices(self, topic: str) -> List[str]:
        """获取最佳实践建议"""
        results = self.search(topic, category="best_practices", top_k=3)
        return [result['content'] for result in results]
    
    def get_code_pattern(self, pattern_name: str) -> Optional[str]:
        """获取代码模式示例"""
        results = self.search(pattern_name, category="common_patterns", top_k=1)
        return results[0]['content'] if results else None
    
    def debug_error(self, error_message: str) -> List[str]:
        """根据错误信息获取调试建议"""
        debug_results = self.search(error_message, category="debugging_tips", top_k=2)
        error_results = self.search(error_message, category="error_patterns", top_k=2)
        
        suggestions = []
        for result in debug_results + error_results:
            suggestions.append(result['content'])
        
        return suggestions
    
    def get_similar_examples(self, code_snippet: str, top_k: int = 3) -> List[Dict]:
        """根据代码片段找到相似的示例"""
        return self.search(f"code example: {code_snippet}", top_k=top_k)

if __name__ == "__main__":
    # 测试RAG数据库
    db = TritonRAGDatabase()
    
    print("=== Triton 3.2 RAG Database Test ===")
    
    # 测试搜索功能
    print("\n1. Search for 'matrix multiplication':")
    results = db.search("matrix multiplication", top_k=2)
    for result in results:
        print(f"- {result['metadata']['title']}")
        print(f"  Distance: {result['distance']:.3f}")
    
    # 测试API参考
    print("\n2. API Reference for 'triton.jit':")
    api_ref = db.get_api_reference("triton.jit")
    if api_ref:
        print(api_ref[:200] + "...")
    
    # 测试最佳实践
    print("\n3. Best practices for 'memory access':")
    practices = db.get_best_practices("memory access")
    for practice in practices[:2]:
        print(f"- {practice[:100]}...")
    
    # 测试错误调试
    print("\n4. Debug 'out of bounds' error:")
    debug_tips = db.debug_error("out of bounds access")
    for tip in debug_tips[:1]:
        print(f"- {tip[:150]}...")
    
    print("\n✅ Triton RAG Database initialized successfully!") 