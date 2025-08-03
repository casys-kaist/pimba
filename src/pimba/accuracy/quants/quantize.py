import random

import triton
import triton.language as tl

from .. import configs
from .bfloat16 import quantize as bfloat16
from .e4m3 import quantize as e4m3
from .e5m2 import quantize as e5m2
from .float16 import quantize as float16
from .int8 import quantize as int8
from .mx8 import quantize as mx8
from .none import quantize as none


@triton.jit
def quantize_kernel(
    x,
    seed,
    QUANT: tl.constexpr,
    USE_SR: tl.constexpr,
):
    if QUANT == "bfloat16":
        return bfloat16(x, seed, USE_SR)
    elif QUANT == "e4m3":
        return e4m3(x, seed, USE_SR)
    elif QUANT == "e5m2":
        return e5m2(x, seed, USE_SR)
    elif QUANT == "float16":
        return float16(x, seed, USE_SR)
    elif QUANT == "int8":
        return int8(x, seed, USE_SR)
    elif QUANT == "mx8":
        return mx8(x, seed, USE_SR)
    elif QUANT == "none":
        return none(x, seed, USE_SR)


@triton.jit
def in_place_quantize_kernel(
    x_ptr,
    col_elements,
    seed,
    QUANT: tl.constexpr,
    USE_SR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = pid * col_elements + col_offsets
    mask = col_offsets < col_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    x = quantize_kernel(x, seed, QUANT, USE_SR)

    tl.store(x_ptr + offsets, x, mask=mask)


def quantize_(x):
    assert x.stride(-1) == 1

    n_elements = x.numel()
    col_elements = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(col_elements)
    seed = random.randint(10, 10000)

    in_place_quantize_kernel[(triton.cdiv(n_elements, col_elements),)](
        x,
        col_elements,
        seed,
        QUANT=configs.QUANT,
        USE_SR=configs.USE_SR,
        BLOCK_SIZE=BLOCK_SIZE,
    )
