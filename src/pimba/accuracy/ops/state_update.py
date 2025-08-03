import random
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .. import configs
from ..quants import quantize_kernel


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV"],
)
@triton.jit
def state_update_kernel(
    # B: B, H: H, T: T, D: d_head
    seed,
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, K]
    v,  # value [B, H, T, V]
    gk,  # log gate [B, H, T, K] or None
    gv,  # log gate [B, H, T, K] or None
    o,  # output [NK, B, H, T, V]
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    s_qk_h,  # stride size: T * K
    s_vo_h,  # stride size: T * V
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    QUANT: tl.constexpr,
    USE_SR: tl.constexpr,
):
    # indices
    i_v, i_k, i_bh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    p_h0 = (
        h0
        + i_bh * K * V
        + (i_k * BK + tl.arange(0, BK)[None, :]) * V
        + (i_v * BV + tl.arange(0, BV)[:, None])
    )
    b_h = tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        if QUANT != "none":
            b_h = quantize_kernel(b_h, seed, QUANT, USE_SR)
            if USE_SR:
                seed += 1

        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        if QUANT != "none":
            b_k = quantize_kernel(b_k, seed, QUANT, USE_SR)
            if USE_SR:
                seed += 1
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        if QUANT != "none":
            b_v = quantize_kernel(b_v, seed, QUANT, USE_SR)
            if USE_SR:
                seed += 1
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32)
        if QUANT != "none":
            b_q = quantize_kernel(b_q, seed, QUANT, USE_SR)
            if USE_SR:
                seed += 1
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            if QUANT != "none":
                b_gk = quantize_kernel(b_gk, seed, QUANT, USE_SR)
                if USE_SR:
                    seed += 1
            b_h = b_h * b_gk[None, :]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            if QUANT != "none":
                b_gv = quantize_kernel(b_gv, seed, QUANT, USE_SR)
                if USE_SR:
                    seed += 1
            b_h = b_h * b_gv[:, None]

        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V
        if USE_GK:
            p_gk += K
        if USE_GV:
            p_gv += V

    p_ht = (
        ht
        + i_bh * K * V
        + (i_k * BK + tl.arange(0, BK)[None, :]) * V
        + (i_v * BV + tl.arange(0, BV)[:, None])
    )
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)


def state_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: Optional[torch.Tensor],
    gv: Optional[torch.Tensor],
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    USE_GK = gk is not None
    USE_GV = gv is not None

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if USE_GK:
        gk = gk.contiguous()
    if USE_GV:
        gv = gv.contiguous()

    B, H, T, K, V = *q.shape, v.shape[-1]

    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    o = q.new_empty(NK, B, H, T, V)

    h0 = initial_state
    ht = q.new_empty(B, H, K, V)
    seed = random.randint(10, 10000)

    grid = (NV, NK, B * H)
    state_update_kernel[grid](
        seed,
        q,
        k,
        v,
        gk,
        gv,
        o,
        h0,
        ht,
        q.stride(1),
        v.stride(1),
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_GK=USE_GK,
        USE_GV=USE_GV,
        QUANT=configs.QUANT,
        USE_SR=configs.USE_SR,
    )

    o = o.sum(0)
    return o.to(k.dtype), ht
