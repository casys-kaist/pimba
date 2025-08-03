import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, SR: tl.constexpr):
    MAX = tl.constexpr(65504.0)
    MIN = tl.constexpr(-65504.0)

    x = tl.where(x > MAX, MAX, x)
    x = tl.where(x < MIN, MIN, x)
    x = x.to(tl.float16)
    x = x.to(tl.float32)

    return x
