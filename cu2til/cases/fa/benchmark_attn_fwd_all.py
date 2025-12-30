from functools import partial
import math
import os
from typing import NamedTuple
import torch
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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


for headdim in [64, 128]:
    for nheads in [1, 4, 8]:
        nheads_kv = nheads
        headdim_v = headdim
        has_qv = headdim == 64 and headdim_v == 512
        for batch_size in [1, 4, 8]:
            for seqlen in [1024, 2048, 4096]:
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