from functools import partial
import math
import os
from typing import NamedTuple
import torch
import time

try:
    import cudnn
except ImportError:
    cudnn = None

Timing = NamedTuple('timing', [('mean', float)])

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

from triton.testing import do_bench

try:
    from fa_320_v2.fa_320_bf16_v2 import attention_causal as triton_attn_causal, attention_non_causal as triton_attn_non_causal
except ImportError:
    triton_attn_causal = triton_attn_non_causal = None


def time_fwd(func, *args, repeats=100, verbose=True, desc="", **kwargs):
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=50, rep=repeats) * 1e-3)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q - window_size[1], torch.tensor(seqlen_k - 1))
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=-1):
    print(f"CuDNN fwd: {q.shape}|{q.stride()}, {k.shape}|{k.stride()}, {v.shape}|{v.stride()}")
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty_like(q_gpu)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


torch.manual_seed(0)
repeats = 10
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
dtype_gen = torch.bfloat16
device = 'cuda'
verbose = True
varlen = False
page_size = None
softcap = 0.0
V_colmajor = False
deterministic = False
dim = 2048
headdim = 256
bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
time_f = {}

for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [64, 96, 128]:
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [128]:
    nheads = dim // headdim
    nheads_kv = nheads
    headdim_v = headdim
    has_qv = headdim == 64 and headdim_v == 512

    for batch_size, seqlen in bs_seqlen_vals:
        num_splits = 0
        window_size = (-1, -1)
        pack_gqa = None
        seqlen_q = seqlen
        leftpad_k = None
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in [q, k, v]]
        v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_()
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen) if has_qv else None
        
        if varlen:
            q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_() for x in [q, k, v]]
            cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
            cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
        if page_size is not None:
            assert seqlen % page_size == 0
            k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
            page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                   "(b s) -> b s", s=seqlen // page_size)
        else:
            page_table = None

        for causal in [False, True]:
            print(f"\n### {batch_size = }, {nheads = }, {seqlen = }, {headdim = }, {causal = }, ###")
            nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen, headdim if not has_qv else headdim + headdim_v, headdim_v, causal=causal, window_size=window_size)
            
            if cudnn is not None and headdim <= 256:
                cudnn_spda = cudnn_spda_setup(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), causal=causal, window_size_left=window_size[0])
            
            if not varlen:
                print(f"Fav2 fwd: {q.shape}|{q.stride()}, {k.shape}|{k.stride()}, {v.shape}|{v.stride()}")
                m0 = time_fwd(flash_attn_func, q, k, v, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
            else:
                m0 = time_fwd(flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
            time_f[(causal, headdim, batch_size, seqlen), "Flash2"] = m0.mean

            if headdim <= 256 and triton_attn_causal is not None and triton_attn_non_causal is not None:
                qt, kt, vt = [x.detach().transpose(1, 2).contiguous().requires_grad_() for x in [q, k, v]]
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                print(f"Triton fwd: {qt.shape}|{qt.stride()}, {kt.shape}|{kt.stride()}, {vt.shape}|{vt.stride()}")
                m3 = time_fwd(triton_attn_causal if causal else triton_attn_non_causal, qt, kt, vt, 1 / math.sqrt(headdim), repeats=repeats, verbose=verbose, desc='Triton')
                time_f[(causal, headdim, batch_size, seqlen), "Triton"] = m3.mean
            
            if cudnn is not None and headdim <= 256:
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                m2 = time_fwd(cudnn_spda, repeats=repeats, verbose=verbose, desc='CuDNN')
                time_f[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2.mean

            time.sleep(1)
            if not varlen:
                k = k if page_size is None else k_paged
                v = v_fa3 if page_size is None else v_paged
                print(f"Fav3 fwd: {q.shape}|{q.stride()}, {k.shape}|{k.stride()}, {v.shape}|{v.stride()}")
                # exit(0)
                m1 = time_fwd(flash_attn_func_v3, q, k, v, qv=qv, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
            else:
                m1 = time_fwd(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
            time_f[(causal, headdim, batch_size, seqlen), "Flash3"] = m1.mean

            print(f'Fav2 fwd: {m0.mean * 1e3:.3f}ms, {(nFLOPS / m0.mean * 1e-12):.1f} TFLOPS')
            if headdim <= 256:
                if triton_attn_causal is not None and triton_attn_non_causal is not None:
                    print(f'Triton fwd: {m3.mean * 1e3:.3f}ms, {(nFLOPS / m3.mean * 1e-12):.1f} TFLOPS')
                if cudnn is not None:
                    print(f'CuDNN fwd: {m2.mean * 1e3:.3f}ms, {(nFLOPS / m2.mean * 1e-12):.1f} TFLOPS')
            print(f'Fav3 fwd: {m1.mean * 1e3:.3f}ms, {(nFLOPS / m1.mean * 1e-12):.1f} TFLOPS')

'''
gpu2
root@sigma106:/workspace/cases/fa/triton# python benchmark_attn_debug.py 

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64])
Fav2 fwd: torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64])
Triton fwd: torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64])
Fav3 fwd: torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64])
Fav2 fwd: 0.345ms, 199.1 TFLOPS
Triton fwd: 0.319ms, 215.4 TFLOPS
CuDNN fwd: 0.265ms, 258.9 TFLOPS
Fav3 fwd: 0.296ms, 231.9 TFLOPS

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64])
Fav2 fwd: torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64])
Triton fwd: torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64]), torch.Size([32, 32, 512, 64])
Fav3 fwd: torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64]), torch.Size([32, 512, 32, 64])
Fav2 fwd: 0.264ms, 130.2 TFLOPS
Triton fwd: 0.250ms, 137.6 TFLOPS
CuDNN fwd: 0.232ms, 148.3 TFLOPS
Fav3 fwd: 0.236ms, 145.7 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64])
Fav2 fwd: torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64])
Triton fwd: torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64])
Fav3 fwd: torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64])
Fav2 fwd: 0.702ms, 195.8 TFLOPS
Triton fwd: 0.527ms, 260.6 TFLOPS
CuDNN fwd: 0.443ms, 310.2 TFLOPS
Fav3 fwd: 0.527ms, 260.8 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64])
Fav2 fwd: torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64])
Triton fwd: torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64]), torch.Size([16, 32, 1024, 64])
Fav3 fwd: torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64]), torch.Size([16, 1024, 32, 64])
Fav2 fwd: 0.411ms, 167.2 TFLOPS
Triton fwd: 0.388ms, 177.0 TFLOPS
CuDNN fwd: 0.326ms, 211.0 TFLOPS
Fav3 fwd: 0.355ms, 193.6 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64])
Fav2 fwd: torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64])
Triton fwd: torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64])
Fav3 fwd: torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64])
Fav2 fwd: 1.339ms, 205.3 TFLOPS
Triton fwd: 1.041ms, 263.9 TFLOPS
CuDNN fwd: 0.820ms, 335.2 TFLOPS
Fav3 fwd: 0.850ms, 323.3 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64])
Fav2 fwd: torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64])
Triton fwd: torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64]), torch.Size([8, 32, 2048, 64])
Fav3 fwd: torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64]), torch.Size([8, 2048, 32, 64])
Fav2 fwd: 0.771ms, 178.2 TFLOPS
Triton fwd: 0.660ms, 208.1 TFLOPS
CuDNN fwd: 0.535ms, 256.9 TFLOPS
Fav3 fwd: 0.504ms, 272.6 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64])
Fav2 fwd: torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64])
Triton fwd: torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64])
Fav3 fwd: torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64])
Fav2 fwd: 2.558ms, 214.9 TFLOPS
Triton fwd: 1.941ms, 283.2 TFLOPS
CuDNN fwd: 1.511ms, 363.7 TFLOPS
Fav3 fwd: 1.644ms, 334.4 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64])
Fav2 fwd: torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64])
Triton fwd: torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64]), torch.Size([4, 32, 4096, 64])
Fav3 fwd: torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64]), torch.Size([4, 4096, 32, 64])
Fav2 fwd: 1.390ms, 197.7 TFLOPS
Triton fwd: 1.186ms, 231.8 TFLOPS
CuDNN fwd: 0.915ms, 300.3 TFLOPS
Fav3 fwd: 0.920ms, 298.8 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64])
Fav2 fwd: torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64])
Triton fwd: torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64])
Fav3 fwd: torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64])
Fav2 fwd: 5.097ms, 215.7 TFLOPS
Triton fwd: 3.913ms, 281.0 TFLOPS
CuDNN fwd: 3.261ms, 337.1 TFLOPS
Fav3 fwd: 3.136ms, 350.7 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64])
Fav2 fwd: torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64])
Triton fwd: torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64]), torch.Size([2, 32, 8192, 64])
Fav3 fwd: torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64]), torch.Size([2, 8192, 32, 64])
Fav2 fwd: 2.684ms, 204.8 TFLOPS
Triton fwd: 2.383ms, 230.7 TFLOPS
CuDNN fwd: 1.601ms, 343.5 TFLOPS
Fav3 fwd: 1.663ms, 330.6 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64])
Fav2 fwd: torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64])
Triton fwd: torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64])
Fav3 fwd: torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64])
Fav2 fwd: 9.498ms, 231.5 TFLOPS
Triton fwd: 7.578ms, 290.2 TFLOPS
CuDNN fwd: 6.455ms, 340.7 TFLOPS
Fav3 fwd: 6.192ms, 355.2 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64])
Fav2 fwd: torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64])
Triton fwd: torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64]), torch.Size([1, 32, 16384, 64])
Fav3 fwd: torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64]), torch.Size([1, 16384, 32, 64])
Fav2 fwd: 5.075ms, 216.6 TFLOPS
Triton fwd: 4.547ms, 241.8 TFLOPS
CuDNN fwd: 3.477ms, 316.2 TFLOPS
Fav3 fwd: 3.156ms, 348.4 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128])
Fav2 fwd: torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128])
Triton fwd: torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128])
Fav3 fwd: torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128])
Fav2 fwd: 0.319ms, 215.1 TFLOPS
Triton fwd: 0.282ms, 243.8 TFLOPS
CuDNN fwd: 0.205ms, 335.2 TFLOPS
Fav3 fwd: 0.205ms, 335.6 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128])
Fav2 fwd: torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128])
Triton fwd: torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128]), torch.Size([32, 16, 512, 128])
Fav3 fwd: torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128]), torch.Size([32, 512, 16, 128])
Fav2 fwd: 0.246ms, 139.8 TFLOPS
Triton fwd: 0.203ms, 168.9 TFLOPS
CuDNN fwd: 0.164ms, 209.1 TFLOPS
Fav3 fwd: 0.195ms, 176.6 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128])
Fav2 fwd: torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128])
Triton fwd: torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128])
Fav3 fwd: torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128])
Fav2 fwd: 0.547ms, 251.3 TFLOPS
Triton fwd: 0.497ms, 276.5 TFLOPS
CuDNN fwd: 0.345ms, 398.0 TFLOPS
Fav3 fwd: 0.310ms, 442.7 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128])
Fav2 fwd: torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128])
Triton fwd: torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128]), torch.Size([16, 16, 1024, 128])
Fav3 fwd: torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128]), torch.Size([16, 1024, 16, 128])
Fav2 fwd: 0.349ms, 196.7 TFLOPS
Triton fwd: 0.369ms, 186.4 TFLOPS
CuDNN fwd: 0.243ms, 283.0 TFLOPS
Fav3 fwd: 0.244ms, 281.7 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128])
Fav2 fwd: torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128])
Triton fwd: torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128])
Fav3 fwd: torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128])
Fav2 fwd: 1.048ms, 262.2 TFLOPS
Triton fwd: 0.888ms, 309.5 TFLOPS
CuDNN fwd: 0.695ms, 395.3 TFLOPS
Fav3 fwd: 0.664ms, 413.8 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128])
Fav2 fwd: torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128])
Triton fwd: torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128]), torch.Size([8, 16, 2048, 128])
Fav3 fwd: torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128]), torch.Size([8, 2048, 16, 128])
Fav2 fwd: 0.646ms, 212.7 TFLOPS
Triton fwd: 0.595ms, 230.8 TFLOPS
CuDNN fwd: 0.410ms, 334.9 TFLOPS
Fav3 fwd: 0.372ms, 369.4 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128])
Fav2 fwd: torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128])
Triton fwd: torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128])
Fav3 fwd: torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128])
Fav2 fwd: 2.018ms, 272.4 TFLOPS
Triton fwd: 1.854ms, 296.5 TFLOPS
CuDNN fwd: 1.339ms, 410.7 TFLOPS
Fav3 fwd: 1.268ms, 433.5 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128])
Fav2 fwd: torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128])
Triton fwd: torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128]), torch.Size([4, 16, 4096, 128])
Fav3 fwd: torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128]), torch.Size([4, 4096, 16, 128])
Fav2 fwd: 1.218ms, 225.7 TFLOPS
Triton fwd: 1.025ms, 268.2 TFLOPS
CuDNN fwd: 0.767ms, 358.3 TFLOPS
Fav3 fwd: 0.686ms, 400.7 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128])
Fav2 fwd: torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128])
Triton fwd: torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128])
Fav3 fwd: torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128])
Fav2 fwd: 3.950ms, 278.4 TFLOPS
Triton fwd: 3.451ms, 318.6 TFLOPS
CuDNN fwd: 2.712ms, 405.5 TFLOPS
Fav3 fwd: 2.335ms, 470.9 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128])
Fav2 fwd: torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128])
Triton fwd: torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128]), torch.Size([2, 16, 8192, 128])
Fav3 fwd: torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128]), torch.Size([2, 8192, 16, 128])
Fav2 fwd: 2.377ms, 231.3 TFLOPS
Triton fwd: 2.262ms, 243.0 TFLOPS
CuDNN fwd: 1.301ms, 422.4 TFLOPS
Fav3 fwd: 1.252ms, 438.9 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128])
Fav2 fwd: torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128])
Triton fwd: torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128])
Fav3 fwd: torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128])
Fav2 fwd: 8.078ms, 272.2 TFLOPS
Triton fwd: 6.649ms, 330.7 TFLOPS
CuDNN fwd: 5.204ms, 422.6 TFLOPS
Fav3 fwd: 4.405ms, 499.2 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128])
Fav2 fwd: torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128])
Triton fwd: torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128]), torch.Size([1, 16, 16384, 128])
Fav3 fwd: torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128]), torch.Size([1, 16384, 16, 128])
Fav2 fwd: 4.724ms, 232.8 TFLOPS
Triton fwd: 4.041ms, 272.1 TFLOPS
CuDNN fwd: 2.850ms, 385.7 TFLOPS
Fav3 fwd: 2.299ms, 478.2 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256])
Fav2 fwd: torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256])
Triton fwd: torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256])
Fav3 fwd: torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256])
Fav2 fwd: 0.319ms, 215.5 TFLOPS
Triton fwd: 0.312ms, 220.6 TFLOPS
CuDNN fwd: 0.208ms, 329.9 TFLOPS
Fav3 fwd: 0.193ms, 356.4 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256])
Fav2 fwd: torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256])
Triton fwd: torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256]), torch.Size([32, 8, 512, 256])
Fav3 fwd: torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256]), torch.Size([32, 512, 8, 256])
Fav2 fwd: 0.223ms, 154.0 TFLOPS
Triton fwd: 0.246ms, 139.5 TFLOPS
CuDNN fwd: 0.166ms, 206.7 TFLOPS
Fav3 fwd: 0.187ms, 184.2 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256])
Fav2 fwd: torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256])
Triton fwd: torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256])
Fav3 fwd: torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256])
Fav2 fwd: 0.621ms, 221.3 TFLOPS
Triton fwd: 0.535ms, 256.7 TFLOPS
CuDNN fwd: 0.349ms, 393.9 TFLOPS
Fav3 fwd: 0.324ms, 424.0 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256])
Fav2 fwd: torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256])
Triton fwd: torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256]), torch.Size([16, 8, 1024, 256])
Fav3 fwd: torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256]), torch.Size([16, 1024, 8, 256])
Fav2 fwd: 0.400ms, 171.7 TFLOPS
Triton fwd: 0.401ms, 171.5 TFLOPS
CuDNN fwd: 0.217ms, 316.2 TFLOPS
Fav3 fwd: 0.221ms, 311.4 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256])
Fav2 fwd: torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256])
Triton fwd: torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256])
Fav3 fwd: torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256])
Fav2 fwd: 1.130ms, 243.3 TFLOPS
Triton fwd: 0.917ms, 299.8 TFLOPS
CuDNN fwd: 0.652ms, 421.6 TFLOPS
Fav3 fwd: 0.597ms, 460.3 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256])
Fav2 fwd: torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256])
Triton fwd: torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256]), torch.Size([8, 8, 2048, 256])
Fav3 fwd: torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256]), torch.Size([8, 2048, 8, 256])
Fav2 fwd: 0.772ms, 178.1 TFLOPS
Triton fwd: 0.633ms, 217.1 TFLOPS
CuDNN fwd: 0.405ms, 339.4 TFLOPS
Fav3 fwd: 0.358ms, 383.8 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256])
Fav2 fwd: torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256])
Triton fwd: torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256])
Fav3 fwd: torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256])
Fav2 fwd: 2.482ms, 221.5 TFLOPS
Triton fwd: 1.799ms, 305.5 TFLOPS
CuDNN fwd: 1.157ms, 475.0 TFLOPS
Fav3 fwd: 1.159ms, 474.5 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256])
Fav2 fwd: torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256])
Triton fwd: torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256]), torch.Size([4, 8, 4096, 256])
Fav3 fwd: torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256]), torch.Size([4, 4096, 8, 256])
Fav2 fwd: 1.218ms, 225.8 TFLOPS
Triton fwd: 1.187ms, 231.7 TFLOPS
CuDNN fwd: 0.672ms, 408.9 TFLOPS
Fav3 fwd: 0.604ms, 455.1 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256])
Fav2 fwd: torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256])
Triton fwd: torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256])
Fav3 fwd: torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256])
Fav2 fwd: 4.187ms, 262.6 TFLOPS
Triton fwd: 4.004ms, 274.6 TFLOPS
CuDNN fwd: 2.408ms, 456.7 TFLOPS
Fav3 fwd: 2.199ms, 500.0 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256])
Fav2 fwd: torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256])
Triton fwd: torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256]), torch.Size([2, 8, 8192, 256])
Fav3 fwd: torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256]), torch.Size([2, 8192, 8, 256])
Fav2 fwd: 2.372ms, 231.8 TFLOPS
Triton fwd: 2.130ms, 258.1 TFLOPS
CuDNN fwd: 1.326ms, 414.7 TFLOPS
Fav3 fwd: 1.160ms, 474.0 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256])
Fav2 fwd: torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256])
Triton fwd: torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256])
Fav3 fwd: torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256])
Fav2 fwd: 9.792ms, 224.6 TFLOPS
Triton fwd: 6.285ms, 349.9 TFLOPS
CuDNN fwd: 4.889ms, 449.8 TFLOPS
Fav3 fwd: 4.512ms, 487.4 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256])
Fav2 fwd: torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256])
Triton fwd: torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256]), torch.Size([1, 8, 16384, 256])
Fav3 fwd: torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256]), torch.Size([1, 16384, 8, 256])
Fav2 fwd: 5.501ms, 199.9 TFLOPS
Triton fwd: 4.231ms, 259.9 TFLOPS
CuDNN fwd: 2.417ms, 454.9 TFLOPS
Fav3 fwd: 2.336ms, 470.8 TFLOPS
'''
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
gpu 2

root@sigma106:/workspace/cases/fa/triton# python benchmark_attn_debug.py 

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1), torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1), torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Triton fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav3 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Fav2 fwd: 0.347ms, 198.3 TFLOPS
Triton fwd: 0.319ms, 215.7 TFLOPS
CuDNN fwd: 0.266ms, 258.7 TFLOPS
Fav3 fwd: 0.316ms, 217.7 TFLOPS

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1), torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1), torch.Size([32, 32, 512, 64])|(1048576, 64, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Triton fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav3 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Fav2 fwd: 0.274ms, 125.5 TFLOPS
Triton fwd: 0.246ms, 139.7 TFLOPS
CuDNN fwd: 0.230ms, 149.6 TFLOPS
Fav3 fwd: 0.242ms, 141.8 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1), torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1), torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Triton fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav3 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Fav2 fwd: 0.606ms, 226.8 TFLOPS
Triton fwd: 0.547ms, 251.0 TFLOPS
CuDNN fwd: 0.414ms, 331.9 TFLOPS
Fav3 fwd: 0.510ms, 269.5 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1), torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1), torch.Size([16, 32, 1024, 64])|(2097152, 64, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Triton fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav3 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Fav2 fwd: 0.434ms, 158.5 TFLOPS
Triton fwd: 0.387ms, 177.7 TFLOPS
CuDNN fwd: 0.325ms, 211.2 TFLOPS
Fav3 fwd: 0.364ms, 189.0 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1), torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1), torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Triton fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav3 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Fav2 fwd: 1.256ms, 218.8 TFLOPS
Triton fwd: 1.039ms, 264.6 TFLOPS
CuDNN fwd: 0.879ms, 312.7 TFLOPS
Fav3 fwd: 0.800ms, 343.6 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1), torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1), torch.Size([8, 32, 2048, 64])|(4194304, 64, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Triton fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav3 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Fav2 fwd: 0.775ms, 177.3 TFLOPS
Triton fwd: 0.579ms, 237.6 TFLOPS
CuDNN fwd: 0.529ms, 259.9 TFLOPS
Fav3 fwd: 0.486ms, 282.6 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1), torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1), torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Triton fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav3 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Fav2 fwd: 2.568ms, 214.1 TFLOPS
Triton fwd: 1.916ms, 287.0 TFLOPS
CuDNN fwd: 1.403ms, 391.9 TFLOPS
Fav3 fwd: 1.655ms, 332.1 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1), torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1), torch.Size([4, 32, 4096, 64])|(8388608, 64, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Triton fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav3 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Fav2 fwd: 1.349ms, 203.8 TFLOPS
Triton fwd: 1.251ms, 219.8 TFLOPS
CuDNN fwd: 0.873ms, 315.0 TFLOPS
Fav3 fwd: 0.814ms, 337.8 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1), torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1), torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Triton fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav3 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Fav2 fwd: 4.826ms, 227.8 TFLOPS
Triton fwd: 3.636ms, 302.4 TFLOPS
CuDNN fwd: 3.080ms, 357.0 TFLOPS
Fav3 fwd: 2.934ms, 374.7 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1), torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1), torch.Size([2, 32, 8192, 64])|(16777216, 64, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Triton fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav3 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Fav2 fwd: 2.599ms, 211.5 TFLOPS
Triton fwd: 2.259ms, 243.3 TFLOPS
CuDNN fwd: 1.524ms, 360.7 TFLOPS
Fav3 fwd: 1.577ms, 348.7 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1), torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1), torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Triton fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav3 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Fav2 fwd: 10.124ms, 217.2 TFLOPS
Triton fwd: 8.924ms, 246.4 TFLOPS
CuDNN fwd: 5.700ms, 385.8 TFLOPS
Fav3 fwd: 5.606ms, 392.2 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1), torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1), torch.Size([1, 32, 16384, 64])|(33554432, 64, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Triton fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav3 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Fav2 fwd: 5.388ms, 204.1 TFLOPS
Triton fwd: 4.556ms, 241.3 TFLOPS
CuDNN fwd: 3.397ms, 323.7 TFLOPS
Fav3 fwd: 3.296ms, 333.5 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1), torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1), torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Triton fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav3 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Fav2 fwd: 0.329ms, 208.9 TFLOPS
Triton fwd: 0.323ms, 212.5 TFLOPS
CuDNN fwd: 0.199ms, 344.8 TFLOPS
Fav3 fwd: 0.193ms, 356.7 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1), torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1), torch.Size([32, 16, 512, 128])|(1048576, 128, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Triton fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav3 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Fav2 fwd: 0.244ms, 140.7 TFLOPS
Triton fwd: 0.224ms, 153.4 TFLOPS
CuDNN fwd: 0.165ms, 208.7 TFLOPS
Fav3 fwd: 0.196ms, 175.6 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1), torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1), torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Triton fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav3 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Fav2 fwd: 0.574ms, 239.3 TFLOPS
Triton fwd: 0.527ms, 260.6 TFLOPS
CuDNN fwd: 0.376ms, 365.9 TFLOPS
Fav3 fwd: 0.352ms, 390.9 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1), torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1), torch.Size([16, 16, 1024, 128])|(2097152, 128, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Triton fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav3 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Fav2 fwd: 0.367ms, 187.3 TFLOPS
Triton fwd: 0.362ms, 189.9 TFLOPS
CuDNN fwd: 0.250ms, 275.0 TFLOPS
Fav3 fwd: 0.244ms, 282.1 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1), torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1), torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Triton fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav3 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Fav2 fwd: 1.060ms, 259.4 TFLOPS
Triton fwd: 0.935ms, 294.0 TFLOPS
CuDNN fwd: 0.674ms, 408.0 TFLOPS
Fav3 fwd: 0.653ms, 420.9 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1), torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1), torch.Size([8, 16, 2048, 128])|(4194304, 128, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Triton fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav3 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Fav2 fwd: 0.685ms, 200.5 TFLOPS
Triton fwd: 0.557ms, 246.8 TFLOPS
CuDNN fwd: 0.398ms, 345.6 TFLOPS
Fav3 fwd: 0.381ms, 361.0 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1), torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1), torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Triton fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav3 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Fav2 fwd: 2.022ms, 271.9 TFLOPS
Triton fwd: 1.871ms, 293.8 TFLOPS
CuDNN fwd: 1.190ms, 462.0 TFLOPS
Fav3 fwd: 1.224ms, 449.3 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1), torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1), torch.Size([4, 16, 4096, 128])|(8388608, 128, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Triton fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav3 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Fav2 fwd: 1.110ms, 247.6 TFLOPS
Triton fwd: 1.147ms, 239.6 TFLOPS
CuDNN fwd: 0.714ms, 384.8 TFLOPS
Fav3 fwd: 0.653ms, 421.1 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1), torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1), torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Triton fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav3 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Fav2 fwd: 4.027ms, 273.0 TFLOPS
Triton fwd: 3.646ms, 301.6 TFLOPS
CuDNN fwd: 2.568ms, 428.2 TFLOPS
Fav3 fwd: 2.191ms, 501.8 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1), torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1), torch.Size([2, 16, 8192, 128])|(16777216, 128, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Triton fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav3 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Fav2 fwd: 2.523ms, 217.9 TFLOPS
Triton fwd: 2.340ms, 235.0 TFLOPS
CuDNN fwd: 1.445ms, 380.5 TFLOPS
Fav3 fwd: 1.142ms, 481.5 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1), torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1), torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Triton fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav3 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Fav2 fwd: 8.060ms, 272.8 TFLOPS
Triton fwd: 6.783ms, 324.2 TFLOPS
CuDNN fwd: 4.869ms, 451.6 TFLOPS
Fav3 fwd: 4.208ms, 522.6 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1), torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1), torch.Size([1, 16, 16384, 128])|(33554432, 128, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Triton fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav3 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Fav2 fwd: 4.761ms, 230.9 TFLOPS
Triton fwd: 4.841ms, 227.1 TFLOPS
CuDNN fwd: 2.791ms, 393.9 TFLOPS
Fav3 fwd: 2.179ms, 504.6 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1), torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1), torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Triton fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav3 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Fav2 fwd: 0.366ms, 187.8 TFLOPS
Triton fwd: 0.315ms, 218.4 TFLOPS
CuDNN fwd: 0.212ms, 323.4 TFLOPS
Fav3 fwd: 0.205ms, 335.2 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1), torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1), torch.Size([32, 8, 512, 256])|(1048576, 256, 2048, 1)
Fav2 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Triton fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav3 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Fav2 fwd: 0.234ms, 146.8 TFLOPS
Triton fwd: 0.246ms, 139.5 TFLOPS
CuDNN fwd: 0.165ms, 208.3 TFLOPS
Fav3 fwd: 0.184ms, 186.4 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1), torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1), torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Triton fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav3 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Fav2 fwd: 0.627ms, 219.2 TFLOPS
Triton fwd: 0.530ms, 259.6 TFLOPS
CuDNN fwd: 0.324ms, 423.8 TFLOPS
Fav3 fwd: 0.310ms, 443.2 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1), torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1), torch.Size([16, 8, 1024, 256])|(2097152, 256, 2048, 1)
Fav2 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Triton fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav3 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Fav2 fwd: 0.378ms, 181.6 TFLOPS
Triton fwd: 0.400ms, 171.7 TFLOPS
CuDNN fwd: 0.234ms, 294.3 TFLOPS
Fav3 fwd: 0.218ms, 314.5 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1), torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1), torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Triton fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav3 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Fav2 fwd: 1.181ms, 232.7 TFLOPS
Triton fwd: 0.933ms, 294.7 TFLOPS
CuDNN fwd: 0.637ms, 431.3 TFLOPS
Fav3 fwd: 0.604ms, 455.4 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1), torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1), torch.Size([8, 8, 2048, 256])|(4194304, 256, 2048, 1)
Fav2 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Triton fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav3 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Fav2 fwd: 0.681ms, 201.9 TFLOPS
Triton fwd: 0.648ms, 212.2 TFLOPS
CuDNN fwd: 0.395ms, 348.3 TFLOPS
Fav3 fwd: 0.368ms, 373.7 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1), torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1), torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Triton fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav3 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Fav2 fwd: 2.151ms, 255.6 TFLOPS
Triton fwd: 1.799ms, 305.6 TFLOPS
CuDNN fwd: 1.214ms, 452.7 TFLOPS
Fav3 fwd: 1.167ms, 470.9 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1), torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1), torch.Size([4, 8, 4096, 256])|(8388608, 256, 2048, 1)
Fav2 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Triton fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav3 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Fav2 fwd: 1.370ms, 200.6 TFLOPS
Triton fwd: 1.186ms, 231.8 TFLOPS
CuDNN fwd: 0.686ms, 400.6 TFLOPS
Fav3 fwd: 0.577ms, 476.5 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1), torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1), torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Triton fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav3 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Fav2 fwd: 4.939ms, 222.6 TFLOPS
Triton fwd: 3.607ms, 304.8 TFLOPS
CuDNN fwd: 2.455ms, 447.9 TFLOPS
Fav3 fwd: 2.134ms, 515.3 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1), torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1), torch.Size([2, 8, 8192, 256])|(16777216, 256, 2048, 1)
Fav2 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Triton fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav3 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Fav2 fwd: 2.631ms, 209.0 TFLOPS
Triton fwd: 2.129ms, 258.2 TFLOPS
CuDNN fwd: 1.389ms, 395.7 TFLOPS
Fav3 fwd: 1.066ms, 515.8 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1), torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1), torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Triton fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav3 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Fav2 fwd: 9.974ms, 220.5 TFLOPS
Triton fwd: 7.247ms, 303.5 TFLOPS
CuDNN fwd: 5.011ms, 438.8 TFLOPS
Fav3 fwd: 4.390ms, 501.0 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1), torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1), torch.Size([1, 8, 16384, 256])|(33554432, 256, 2048, 1)
Fav2 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Triton fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav3 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Fav2 fwd: 5.058ms, 217.4 TFLOPS
Triton fwd: 5.022ms, 218.9 TFLOPS
CuDNN fwd: 2.653ms, 414.4 TFLOPS
Fav3 fwd: 2.068ms, 531.6 TFLOPS
'''
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
cudnn conti gpu 2
root@sigma106:/workspace/cases/fa/triton# python benchmark_attn_debug.py 

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav2 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Triton fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav3 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Fav2 fwd: 0.345ms, 198.9 TFLOPS
Triton fwd: 0.310ms, 221.4 TFLOPS
CuDNN fwd: 0.241ms, 285.2 TFLOPS
Fav3 fwd: 0.301ms, 228.6 TFLOPS

### batch_size = 32, nheads = 32, seqlen = 512, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav2 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Triton fwd: torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1), torch.Size([32, 32, 512, 64])|(1048576, 32768, 64, 1)
Fav3 fwd: torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1), torch.Size([32, 512, 32, 64])|(1048576, 2048, 64, 1)
Fav2 fwd: 0.281ms, 122.1 TFLOPS
Triton fwd: 0.246ms, 139.8 TFLOPS
CuDNN fwd: 0.215ms, 160.1 TFLOPS
Fav3 fwd: 0.239ms, 143.8 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav2 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Triton fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav3 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Fav2 fwd: 0.691ms, 199.0 TFLOPS
Triton fwd: 0.556ms, 247.2 TFLOPS
CuDNN fwd: 0.412ms, 334.0 TFLOPS
Fav3 fwd: 0.505ms, 272.1 TFLOPS

### batch_size = 16, nheads = 32, seqlen = 1024, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav2 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Triton fwd: torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1), torch.Size([16, 32, 1024, 64])|(2097152, 65536, 64, 1)
Fav3 fwd: torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1), torch.Size([16, 1024, 32, 64])|(2097152, 2048, 64, 1)
Fav2 fwd: 0.418ms, 164.6 TFLOPS
Triton fwd: 0.378ms, 181.6 TFLOPS
CuDNN fwd: 0.324ms, 212.4 TFLOPS
Fav3 fwd: 0.348ms, 197.6 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav2 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Triton fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav3 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Fav2 fwd: 1.206ms, 228.0 TFLOPS
Triton fwd: 1.048ms, 262.3 TFLOPS
CuDNN fwd: 0.733ms, 375.2 TFLOPS
Fav3 fwd: 0.753ms, 365.2 TFLOPS

### batch_size = 8, nheads = 32, seqlen = 2048, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav2 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Triton fwd: torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1), torch.Size([8, 32, 2048, 64])|(4194304, 131072, 64, 1)
Fav3 fwd: torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1), torch.Size([8, 2048, 32, 64])|(4194304, 2048, 64, 1)
Fav2 fwd: 0.749ms, 183.4 TFLOPS
Triton fwd: 0.682ms, 201.5 TFLOPS
CuDNN fwd: 0.513ms, 267.7 TFLOPS
Fav3 fwd: 0.517ms, 265.7 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav2 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Triton fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav3 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Fav2 fwd: 2.425ms, 226.7 TFLOPS
Triton fwd: 2.045ms, 268.8 TFLOPS
CuDNN fwd: 1.587ms, 346.4 TFLOPS
Fav3 fwd: 1.650ms, 333.1 TFLOPS

### batch_size = 4, nheads = 32, seqlen = 4096, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav2 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Triton fwd: torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1), torch.Size([4, 32, 4096, 64])|(8388608, 262144, 64, 1)
Fav3 fwd: torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1), torch.Size([4, 4096, 32, 64])|(8388608, 2048, 64, 1)
Fav2 fwd: 1.216ms, 226.1 TFLOPS
Triton fwd: 1.224ms, 224.6 TFLOPS
CuDNN fwd: 0.922ms, 298.2 TFLOPS
Fav3 fwd: 0.937ms, 293.3 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav2 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Triton fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav3 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Fav2 fwd: 4.683ms, 234.8 TFLOPS
Triton fwd: 4.037ms, 272.3 TFLOPS
CuDNN fwd: 3.050ms, 360.5 TFLOPS
Fav3 fwd: 3.101ms, 354.6 TFLOPS

### batch_size = 2, nheads = 32, seqlen = 8192, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav2 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Triton fwd: torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1), torch.Size([2, 32, 8192, 64])|(16777216, 524288, 64, 1)
Fav3 fwd: torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1), torch.Size([2, 8192, 32, 64])|(16777216, 2048, 64, 1)
Fav2 fwd: 2.646ms, 207.7 TFLOPS
Triton fwd: 2.313ms, 237.7 TFLOPS
CuDNN fwd: 1.594ms, 344.9 TFLOPS
Fav3 fwd: 1.513ms, 363.4 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = False, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav2 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Triton fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav3 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Fav2 fwd: 10.287ms, 213.8 TFLOPS
Triton fwd: 8.051ms, 273.1 TFLOPS
CuDNN fwd: 6.214ms, 353.9 TFLOPS
Fav3 fwd: 6.272ms, 350.6 TFLOPS

### batch_size = 1, nheads = 32, seqlen = 16384, headdim = 64, causal = True, ###
CuDNN fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav2 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Triton fwd: torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1), torch.Size([1, 32, 16384, 64])|(33554432, 1048576, 64, 1)
Fav3 fwd: torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1), torch.Size([1, 16384, 32, 64])|(33554432, 2048, 64, 1)
Fav2 fwd: 5.167ms, 212.8 TFLOPS
Triton fwd: 4.271ms, 257.4 TFLOPS
CuDNN fwd: 3.349ms, 328.3 TFLOPS
Fav3 fwd: 3.010ms, 365.3 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav2 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Triton fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav3 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Fav2 fwd: 0.309ms, 222.7 TFLOPS
Triton fwd: 0.281ms, 244.3 TFLOPS
CuDNN fwd: 0.197ms, 348.3 TFLOPS
Fav3 fwd: 0.198ms, 346.3 TFLOPS

### batch_size = 32, nheads = 16, seqlen = 512, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav2 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Triton fwd: torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1), torch.Size([32, 16, 512, 128])|(1048576, 65536, 128, 1)
Fav3 fwd: torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1), torch.Size([32, 512, 16, 128])|(1048576, 2048, 128, 1)
Fav2 fwd: 0.243ms, 141.5 TFLOPS
Triton fwd: 0.231ms, 149.0 TFLOPS
CuDNN fwd: 0.165ms, 208.5 TFLOPS
Fav3 fwd: 0.194ms, 177.4 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav2 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Triton fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav3 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Fav2 fwd: 0.572ms, 240.2 TFLOPS
Triton fwd: 0.517ms, 265.9 TFLOPS
CuDNN fwd: 0.372ms, 369.2 TFLOPS
Fav3 fwd: 0.335ms, 410.5 TFLOPS

### batch_size = 16, nheads = 16, seqlen = 1024, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav2 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Triton fwd: torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1), torch.Size([16, 16, 1024, 128])|(2097152, 131072, 128, 1)
Fav3 fwd: torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1), torch.Size([16, 1024, 16, 128])|(2097152, 2048, 128, 1)
Fav2 fwd: 0.371ms, 185.3 TFLOPS
Triton fwd: 0.340ms, 201.8 TFLOPS
CuDNN fwd: 0.229ms, 300.6 TFLOPS
Fav3 fwd: 0.235ms, 292.6 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav2 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Triton fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav3 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Fav2 fwd: 1.034ms, 265.8 TFLOPS
Triton fwd: 0.902ms, 304.6 TFLOPS
CuDNN fwd: 0.695ms, 395.6 TFLOPS
Fav3 fwd: 0.671ms, 409.5 TFLOPS

### batch_size = 8, nheads = 16, seqlen = 2048, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav2 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Triton fwd: torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1), torch.Size([8, 16, 2048, 128])|(4194304, 262144, 128, 1)
Fav3 fwd: torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1), torch.Size([8, 2048, 16, 128])|(4194304, 2048, 128, 1)
Fav2 fwd: 0.638ms, 215.5 TFLOPS
Triton fwd: 0.653ms, 210.6 TFLOPS
CuDNN fwd: 0.393ms, 349.8 TFLOPS
Fav3 fwd: 0.354ms, 388.1 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav2 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Triton fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav3 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Fav2 fwd: 2.202ms, 249.6 TFLOPS
Triton fwd: 1.814ms, 303.1 TFLOPS
CuDNN fwd: 1.187ms, 463.3 TFLOPS
Fav3 fwd: 1.266ms, 434.2 TFLOPS

### batch_size = 4, nheads = 16, seqlen = 4096, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav2 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Triton fwd: torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1), torch.Size([4, 16, 4096, 128])|(8388608, 524288, 128, 1)
Fav3 fwd: torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1), torch.Size([4, 4096, 16, 128])|(8388608, 2048, 128, 1)
Fav2 fwd: 1.246ms, 220.6 TFLOPS
Triton fwd: 1.115ms, 246.5 TFLOPS
CuDNN fwd: 0.744ms, 369.5 TFLOPS
Fav3 fwd: 0.687ms, 400.2 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav2 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Triton fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav3 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Fav2 fwd: 3.986ms, 275.8 TFLOPS
Triton fwd: 3.708ms, 296.5 TFLOPS
CuDNN fwd: 2.571ms, 427.7 TFLOPS
Fav3 fwd: 2.129ms, 516.5 TFLOPS

### batch_size = 2, nheads = 16, seqlen = 8192, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav2 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Triton fwd: torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1), torch.Size([2, 16, 8192, 128])|(16777216, 1048576, 128, 1)
Fav3 fwd: torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1), torch.Size([2, 8192, 16, 128])|(16777216, 2048, 128, 1)
Fav2 fwd: 2.415ms, 227.6 TFLOPS
Triton fwd: 2.296ms, 239.5 TFLOPS
CuDNN fwd: 1.426ms, 385.7 TFLOPS
Fav3 fwd: 1.195ms, 459.9 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = False, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav2 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Triton fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav3 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Fav2 fwd: 8.654ms, 254.1 TFLOPS
Triton fwd: 7.326ms, 300.2 TFLOPS
CuDNN fwd: 5.245ms, 419.2 TFLOPS
Fav3 fwd: 4.779ms, 460.1 TFLOPS

### batch_size = 1, nheads = 16, seqlen = 16384, headdim = 128, causal = True, ###
CuDNN fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav2 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Triton fwd: torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1), torch.Size([1, 16, 16384, 128])|(33554432, 2097152, 128, 1)
Fav3 fwd: torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1), torch.Size([1, 16384, 16, 128])|(33554432, 2048, 128, 1)
Fav2 fwd: 4.634ms, 237.3 TFLOPS
Triton fwd: 4.226ms, 260.2 TFLOPS
CuDNN fwd: 2.814ms, 390.8 TFLOPS
Fav3 fwd: 2.475ms, 444.2 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav2 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Triton fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav3 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Fav2 fwd: 0.356ms, 192.8 TFLOPS
Triton fwd: 0.308ms, 223.3 TFLOPS
CuDNN fwd: 0.198ms, 347.1 TFLOPS
Fav3 fwd: 0.197ms, 348.5 TFLOPS

### batch_size = 32, nheads = 8, seqlen = 512, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav2 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Triton fwd: torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1), torch.Size([32, 8, 512, 256])|(1048576, 131072, 256, 1)
Fav3 fwd: torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1), torch.Size([32, 512, 8, 256])|(1048576, 2048, 256, 1)
Fav2 fwd: 0.235ms, 146.2 TFLOPS
Triton fwd: 0.254ms, 135.1 TFLOPS
CuDNN fwd: 0.167ms, 206.2 TFLOPS
Fav3 fwd: 0.187ms, 183.7 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav2 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Triton fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav3 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Fav2 fwd: 0.667ms, 206.0 TFLOPS
Triton fwd: 0.524ms, 262.1 TFLOPS
CuDNN fwd: 0.340ms, 404.6 TFLOPS
Fav3 fwd: 0.335ms, 410.5 TFLOPS

### batch_size = 16, nheads = 8, seqlen = 1024, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav2 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Triton fwd: torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1), torch.Size([16, 8, 1024, 256])|(2097152, 262144, 256, 1)
Fav3 fwd: torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1), torch.Size([16, 1024, 8, 256])|(2097152, 2048, 256, 1)
Fav2 fwd: 0.403ms, 170.5 TFLOPS
Triton fwd: 0.402ms, 170.8 TFLOPS
CuDNN fwd: 0.229ms, 300.7 TFLOPS
Fav3 fwd: 0.220ms, 312.3 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav2 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Triton fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav3 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Fav2 fwd: 1.154ms, 238.2 TFLOPS
Triton fwd: 0.928ms, 296.3 TFLOPS
CuDNN fwd: 0.647ms, 425.1 TFLOPS
Fav3 fwd: 0.564ms, 487.0 TFLOPS

### batch_size = 8, nheads = 8, seqlen = 2048, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav2 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Triton fwd: torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1), torch.Size([8, 8, 2048, 256])|(4194304, 524288, 256, 1)
Fav3 fwd: torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1), torch.Size([8, 2048, 8, 256])|(4194304, 2048, 256, 1)
Fav2 fwd: 0.755ms, 182.1 TFLOPS
Triton fwd: 0.636ms, 216.2 TFLOPS
CuDNN fwd: 0.374ms, 367.2 TFLOPS
Fav3 fwd: 0.337ms, 407.4 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav2 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Triton fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav3 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Fav2 fwd: 2.258ms, 243.4 TFLOPS
Triton fwd: 1.828ms, 300.7 TFLOPS
CuDNN fwd: 1.223ms, 449.5 TFLOPS
Fav3 fwd: 1.116ms, 492.7 TFLOPS

### batch_size = 4, nheads = 8, seqlen = 4096, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav2 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Triton fwd: torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1), torch.Size([4, 8, 4096, 256])|(8388608, 1048576, 256, 1)
Fav3 fwd: torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1), torch.Size([4, 4096, 8, 256])|(8388608, 2048, 256, 1)
Fav2 fwd: 1.374ms, 200.1 TFLOPS
Triton fwd: 1.192ms, 230.6 TFLOPS
CuDNN fwd: 0.703ms, 390.8 TFLOPS
Fav3 fwd: 0.650ms, 423.1 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav2 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Triton fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav3 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Fav2 fwd: 4.940ms, 222.6 TFLOPS
Triton fwd: 3.761ms, 292.4 TFLOPS
CuDNN fwd: 2.466ms, 445.8 TFLOPS
Fav3 fwd: 2.148ms, 511.9 TFLOPS

### batch_size = 2, nheads = 8, seqlen = 8192, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav2 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Triton fwd: torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1), torch.Size([2, 8, 8192, 256])|(16777216, 2097152, 256, 1)
Fav3 fwd: torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1), torch.Size([2, 8192, 8, 256])|(16777216, 2048, 256, 1)
Fav2 fwd: 2.667ms, 206.1 TFLOPS
Triton fwd: 2.065ms, 266.2 TFLOPS
CuDNN fwd: 1.354ms, 406.1 TFLOPS
Fav3 fwd: 1.132ms, 485.9 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = False, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav2 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Triton fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav3 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Fav2 fwd: 9.791ms, 224.6 TFLOPS
Triton fwd: 8.094ms, 271.7 TFLOPS
CuDNN fwd: 4.982ms, 441.4 TFLOPS
Fav3 fwd: 4.565ms, 481.8 TFLOPS

### batch_size = 1, nheads = 8, seqlen = 16384, headdim = 256, causal = True, ###
CuDNN fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav2 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Triton fwd: torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1), torch.Size([1, 8, 16384, 256])|(33554432, 4194304, 256, 1)
Fav3 fwd: torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1), torch.Size([1, 16384, 8, 256])|(33554432, 2048, 256, 1)
Fav2 fwd: 5.324ms, 206.5 TFLOPS
Triton fwd: 4.427ms, 248.4 TFLOPS
CuDNN fwd: 2.643ms, 416.0 TFLOPS
Fav3 fwd: 2.062ms, 533.3 TFLOPS
root@sigma106:/workspace/cases/fa/triton# 
'''