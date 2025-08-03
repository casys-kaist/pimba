import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, SR: tl.constexpr):
    MAX: tl.constexpr = 448
    MIN: tl.constexpr = -448
    M: tl.constexpr = 3
    NORMAL_MIN_EXP: tl.constexpr = -6
    SUB_NORMAL_MIN: tl.constexpr = 2**-9
    SR_WIDTH: tl.constexpr = 8

    R_MASK: tl.constexpr = (2**SR_WIDTH - 1) << (23 - SR_WIDTH - M)
    RTN_MASK: tl.constexpr = 1 << (23 - 1 - M)
    M_MASK: tl.constexpr = (1 << 23) - 1
    E_MASK: tl.constexpr = ((1 << 8) - 1) << 23
    SHIFT: tl.constexpr = 23 - M

    # manipulate bits
    x = x.to(tl.int32, bitcast=True)
    if SR:
        r = tl.randint(seed, tl.arange(0, x.numel))
        r = tl.reshape(r, x.shape)
        r = r & R_MASK
        mantissa = x & M_MASK
        mantissa = ((mantissa + r) >> SHIFT) << SHIFT
    else:
        r = tl.where(x & RTN_MASK != 0, 1, 0)
        mantissa = x & M_MASK
        mantissa = ((mantissa >> SHIFT) + r) << SHIFT

    # for subnormal
    exponent = (x & E_MASK) >> 23
    c = (exponent - 127) < NORMAL_MIN_EXP
    x = x.to(tl.float32, bitcast=True)
    x = tl.where(c, (x / SUB_NORMAL_MIN).to(tl.int32) * SUB_NORMAL_MIN, x)

    # for normal
    x = x.to(tl.int32, bitcast=True)
    x = tl.where(c, x, (x & (~M_MASK)) + mantissa)

    # saturate
    x = x.to(tl.float32, bitcast=True)
    x = tl.where(x > MAX, MAX, x)
    x = tl.where(x < MIN, MIN, x)

    return x
