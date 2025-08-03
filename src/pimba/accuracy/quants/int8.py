import triton
import triton.language as tl


@triton.jit
def quantize(x, seed, SR: tl.constexpr):
    ori_x = x
    x = tl.reshape(x, (x.numel // 32, 32))
    max = tl.max(tl.abs(x), axis=1)

    max = max.to(tl.bfloat16)
    max = max.to(tl.float32)

    max = tl.reshape(max, (x.numel // 32, 1))

    x = x / max * 127
    if SR:
        r = tl.rand(seed, tl.arange(0, x.numel))
        r = tl.reshape(r, (x.numel // 32, 32))
        x += r - 0.5
    x = tl.where(x < 0, x - 0.5, x + 0.5)
    x = x.to(tl.int32)
    x = tl.where(x > 127, 127, x)
    x = tl.where(x < -127, -127, x)
    x = x * max / 127

    x = tl.reshape(x, ori_x.shape)
    return x
