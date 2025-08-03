import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, SR: tl.constexpr):
    M: tl.constexpr = 6
    SR_WIDTH: tl.constexpr = 8

    E_MASK: tl.constexpr = ((1 << 8) - 1) << 23
    M_MASK: tl.constexpr = (1 << 23) - 1
    SHIFT: tl.constexpr = 23 - M + 1
    RTN_MASK: tl.constexpr = 1 << (SHIFT - 1)
    OUTPUT_M_MASK: tl.constexpr = (1 << M) - 1
    R_MASK: tl.constexpr = (2**SR_WIDTH - 1) << (SHIFT - SR_WIDTH)

    x = x.to(tl.int32, bitcast=True)
    ori_x = x
    x = tl.reshape(x, (x.numel // 16, 8, 2))

    exp = (x & E_MASK) >> 23
    sub_max = tl.max(exp, axis=2)
    max = tl.max(sub_max, axis=1)

    sub_max = tl.reshape(sub_max, (x.numel // 16, 8, 1))
    max = tl.reshape(max, (x.numel // 16, 1, 1))
    compensation = tl.where(max - sub_max >= 1, 1, 0)
    diff = max - exp - compensation

    m = (x & M_MASK) | (2**23)  # type: ignore
    m = m >> diff
    if SR:
        r = tl.randint(seed, tl.arange(0, x.numel))
        r = tl.reshape(r, (x.numel // 16, 8, 2))
        r = r & R_MASK
        m = m + r
    else:
        m = m + RTN_MASK
    m = m >> SHIFT
    m = m << SHIFT
    m = m << diff

    x = x & (~M_MASK)
    x = x | (m & M_MASK)

    # exponent handling (for rounded-up values)
    x = x + ((m & (1 << (23 + 1))) >> 1)  # type: ignore

    # handling underflow
    x = tl.where(m < (1 << 23), 0, x)  # type: ignore

    x = tl.reshape(x, ori_x.shape)
    x = x.to(tl.float32, bitcast=True)

    return x
