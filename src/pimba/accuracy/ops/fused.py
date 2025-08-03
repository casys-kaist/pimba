import torch
import torch.nn.functional as F


@torch.compile()
def fused_y_update(x, y, d):
    y += x * d[..., None, None]
    return y


@torch.compile()
def fused_a_discretization(dt, dt_bias, A):
    _dt = F.softplus(dt + dt_bias)
    _dA = _dt * A
    _dA = _dA.exp()

    return _dt, _dA
