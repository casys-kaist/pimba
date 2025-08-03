import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, SR: tl.constexpr):
    x = x.to(tl.bfloat16)
    x = x.to(tl.float32)

    return x
