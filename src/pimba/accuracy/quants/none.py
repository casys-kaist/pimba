import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, use_sr: tl.constexpr):
    return x
