#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Enable both serial verification and parallel execution for comparison
#define TILE_SERIAL_VERIFY 0  // Set to 1 for verification mode
#include "dsl_template.cuh"

// ========================= Universal Tile Framework Demo =========================

// Example 1: Flash Attention QK^T Pattern
template<const TunableInt kHeadDim = 64>
__global__ void demo_flash_attention_qkt() {
    constexpr TunableInt kWarpTileSeqQ = 1;
    constexpr TunableInt kWarpTileSeqK = 4; 
    constexpr TunableInt kMmaAtomM = 16;
    constexpr TunableInt kMmaAtomN = 8;
    constexpr TunableInt kMmaAtomK = 16;
    
    // Shared memory allocation
    mem_alloc_shared(fp16, Q_smem, [16][64]);  // Query tile
    mem_alloc_shared(fp16, K_smem, [32][64]);  // Key tile
    
    // Register allocation
    mem_alloc_register(uint32_t, R_S, [kWarpTileSeqQ][kWarpTileSeqK][2]);
    mem_alloc_register(uint32_t, R_Q, [kWarpTileSeqQ][4]);
    mem_alloc_register(uint32_t, R_K, [kWarpTileSeqK][2]);
    
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    
    printf("Demo 1: Flash Attention QK^T using universal tile framework\n");
    
    // ✅ BEFORE: 22 lines of complex MMA code
    // ✅ AFTER: 1 line with semantic clarity
    serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) {
        tile_flash_qkt<kWarpTileSeqQ, kWarpTileSeqK, kHeadDim>(
            Q_smem, K_smem, R_S, R_Q, R_K, tile_K_d, 0, 0, lane_id);
    }
    
    printf("Flash Attention QK^T computation completed\n");
}

// Example 2: Standard GEMM Pattern  
template<const TunableInt kM = 64, const TunableInt kN = 64, const TunableInt kK = 64>
__global__ void demo_standard_gemm() {
    constexpr TunableInt kWarpM = 4;  // 4x16 = 64
    constexpr TunableInt kWarpN = 8;  // 8x8 = 64
    constexpr TunableInt kWarpK = 4;  // 4x16 = 64
    
    // Shared memory allocation
    mem_alloc_shared(fp16, A_smem, [64][64]);
    mem_alloc_shared(fp16, B_smem, [64][64]);
    
    // Register allocation
    mem_alloc_register(uint32_t, R_C, [kWarpM][kWarpN][2]);
    mem_alloc_register(uint32_t, R_A, [kWarpM][4]);
    mem_alloc_register(uint32_t, R_B, [kWarpN][2]);
    
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    
    printf("Demo 2: Standard GEMM using universal tile framework\n");
    
    // ✅ BEFORE: Complex nested loops with MMA calls
    // ✅ AFTER: Single generic call that works for any GEMM
    serial_range_for(k_tile, 0, (kK / 16), 1) {
        tile_standard_gemm<kWarpM, kWarpN, kWarpK>(
            A_smem, B_smem, R_C, R_A, R_B, k_tile, 0, 0, lane_id);
    }
    
    printf("Standard GEMM computation completed\n");
}

// Example 3: Softmax Pattern
__global__ void demo_tile_softmax() {
    constexpr TunableInt kWarpTileSeq = 4;
    constexpr TunableInt kAtomN = 8;
    
    // Register allocation for scores, max, and sum
    mem_alloc_register(uint32_t, R_S, [1][kWarpTileSeq][2]);
    mem_alloc_register(fp32, max_regs, [1][2]);
    mem_alloc_register(fp32, sum_regs, [1][2]);
    
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    fp32 scale = 1.0f / 8.0f;  // Attention scale
    
    printf("Demo 3: Tile-level Softmax using universal framework\n");
    
    // Initialize with test values
    serial_range_for(j, 0, kWarpTileSeq, 1) {
        fp16* s_ptr = register_as_fp16(&R_S[0][j][0]);
        s_ptr[0] = compute_cast<fp32, fp16>((fp32)(j + 1.0f));
        s_ptr[1] = compute_cast<fp32, fp16>((fp32)(j + 2.0f)); 
        s_ptr[2] = compute_cast<fp32, fp16>((fp32)(j + 3.0f));
        s_ptr[3] = compute_cast<fp32, fp16>((fp32)(j + 4.0f));
    }
    
    mem_fill(max_regs, -INFINITY);
    mem_fill(sum_regs, 0.0f);
    
    // ✅ BEFORE: ~30 lines of complex softmax computation
    // ✅ AFTER: 1 line with clear semantic intent
    tile_softmax<kWarpTileSeq, kAtomN>(R_S, scale, max_regs, sum_regs, lane_id);
    
    if (threadIdx.x == 0) {
        printf("Softmax: max=[%.4f, %.4f], sum=[%.4f, %.4f]\n",
               max_regs[0][0], max_regs[0][1], sum_regs[0][0], sum_regs[0][1]);
    }
}

// Example 4: Accumulator Rescaling Pattern
__global__ void demo_tile_rescale() {
    constexpr TunableInt kWarpTileM = 1;
    constexpr TunableInt kWarpTileN = 8;
    
    mem_alloc_register(uint32_t, R_O, [kWarpTileM][kWarpTileN][2]); // Current accumulator
    mem_alloc_register(uint32_t, R_D, [kWarpTileM][kWarpTileN][4]); // Final accumulator
    
    printf("Demo 4: Tile-level Accumulator Rescaling\n");
    
    // Initialize test values
    serial_range_for(j, 0, kWarpTileN, 1) {
        fp16* o_ptr = register_as_fp16(&R_O[0][j][0]);
        fp32* d_ptr = register_as_fp32(&R_D[0][j][0]);
        
        o_ptr[0] = compute_cast<fp32, fp16>((fp32)(j + 1.0f));
        o_ptr[1] = compute_cast<fp32, fp16>((fp32)(j + 2.0f));
        o_ptr[2] = compute_cast<fp32, fp16>((fp32)(j + 3.0f));
        o_ptr[3] = compute_cast<fp32, fp16>((fp32)(j + 4.0f));
        
        d_ptr[0] = (fp32)(j * 0.5f);
        d_ptr[1] = (fp32)(j * 0.6f);
        d_ptr[2] = (fp32)(j * 0.7f);
        d_ptr[3] = (fp32)(j * 0.8f);
    }
    
    fp32 rescale_factor_0 = 0.8f;
    fp32 rescale_factor_1 = 0.9f;
    
    // ✅ BEFORE: Manual loops with complex FMA operations  
    // ✅ AFTER: Clear semantic rescaling operation
    tile_rescale_accumulator<kWarpTileM, kWarpTileN>(
        R_O, R_D, rescale_factor_0, rescale_factor_1);
    
    printf("Accumulator rescaling completed\n");
}

// Example 5: Result Redistribution Pattern
__global__ void demo_tile_redistribute() {
    constexpr TunableInt kWarpTileN = 8;
    
    mem_alloc_register(uint32_t, src_regs, [1][kWarpTileN][2]);
    mem_alloc_register(uint32_t, dst_regs, [2][4]);
    
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    
    printf("Demo 5: Tile-level Result Redistribution\n");
    
    // Initialize test pattern
    serial_range_for(j, 0, kWarpTileN, 1) {
        fp16* src_ptr = register_as_fp16(&src_regs[0][j][0]);
        src_ptr[0] = compute_cast<fp32, fp16>((fp32)(lane_id + j * 10.0f));
        src_ptr[1] = compute_cast<fp32, fp16>((fp32)(lane_id + j * 10.0f + 1.0f));
        src_ptr[2] = compute_cast<fp32, fp16>((fp32)(lane_id + j * 10.0f + 2.0f));
        src_ptr[3] = compute_cast<fp32, fp16>((fp32)(lane_id + j * 10.0f + 3.0f));
    }
    
    // ✅ BEFORE: Manual warp shuffle patterns
    // ✅ AFTER: Semantic redistribution primitive
    tile_redistribute_results<kWarpTileN>(src_regs[0], dst_regs, lane_id);
    
    printf("Result redistribution completed\n");
}

// Main demonstration kernel
__global__ void universal_tile_framework_demo() {
    printf("====== Universal Tile-Level Framework Demo ======\n");
    printf("Thread %d starting demonstrations\n", threadIdx.x);
    
    // Run all pattern demonstrations
    demo_flash_attention_qkt<64>();
    block_sync();
    
    demo_standard_gemm<64, 64, 64>();
    block_sync();
    
    demo_tile_softmax();
    block_sync();
    
    demo_tile_rescale();
    block_sync();
    
    demo_tile_redistribute();
    block_sync();
    
    if (threadIdx.x == 0) {
        printf("====== All Tile-Level Patterns Demonstrated ======\n");
        printf("Framework Benefits:\n");
        printf("1. ✅ Universal: Works for Flash Attention, GEMM, Softmax, etc.\n");
        printf("2. ✅ Layered: Micro-tile → Tile → Pattern abstractions\n");
        printf("3. ✅ Verifiable: Serial verification mode available\n");
        printf("4. ✅ Maintainable: High-level semantic operations\n");
        printf("5. ✅ Extensible: Easy to add new patterns\n");
    }
}

// ========================= Host Side Demo =========================

void run_universal_tile_demo() {
    printf("Testing Universal Tile-Level Framework\n");
    printf("Compilation Mode: %s\n", TILE_SERIAL_VERIFY ? "Serial Verification" : "Parallel Production");
    
    // Launch demonstration kernel
    dim3 grid(1);
    dim3 block(32); // Single warp for simplicity
    
    universal_tile_framework_demo<<<grid, block>>>();
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    printf("\n====== Pattern Recognition Guide ======\n");
    printf("When you see these patterns, use these primitives:\n\n");
    
    printf("🔍 MMA Load-Compute Pattern:\n");
    printf("   - Complex ldmatrix + mma loops → tile_flash_qkt/tile_standard_gemm\n\n");
    
    printf("🔍 Softmax Pattern:\n");
    printf("   - Max finding + exp + sum + shuffle → tile_softmax\n\n");
    
    printf("🔍 Accumulator Update Pattern:\n"); 
    printf("   - Manual FMA with rescale factors → tile_rescale_accumulator\n\n");
    
    printf("🔍 Result Distribution Pattern:\n");
    printf("   - warp_shuffle_spread_x4 chains → tile_redistribute_results\n\n");
    
    printf("🎯 Migration Strategy:\n");
    printf("   1. Identify pattern type (MMA, Softmax, Rescale, etc.)\n");
    printf("   2. Replace with appropriate tile_* primitive\n");
    printf("   3. Enable TILE_SERIAL_VERIFY=1 for verification\n");
    printf("   4. Test both modes to ensure correctness\n");
    printf("   5. Deploy with TILE_SERIAL_VERIFY=0 for production\n\n");
}

int main() {
    run_universal_tile_demo();
    return 0;
}