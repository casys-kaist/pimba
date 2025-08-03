# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from ....ops import fused_a_discretization, fused_y_update, state_update


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(
                self.d_model, d_in_proj, bias=bias, **factory_kwargs
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        if self.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

    def forward(
        self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        batch, seqlen, _ = u.shape

        zxbcdt = self.in_proj(u)  # (B L Ds)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        ).transpose(1, 2)
        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        # SSM
        _dt, _dA = fused_a_discretization(dt, self.dt_bias, A)
        _dA = rearrange(_dA, "b l h -> b h l")
        _dA = repeat(_dA, "b h l -> b h l n", n=self.d_state)
        _x = rearrange(x, "b l (h p) -> b h l p", p=self.headdim)
        _dB = torch.einsum("blh,bln->bhln", _dt, B)
        _C = repeat(C, "b l n -> b h l n", h=self.nheads)
        state = u.new_zeros(batch, _C.shape[1], _x.shape[-1], _dB.shape[-1])
        _y, _ = state_update(_C, _dB, _x, _dA, None, state)
        _y = fused_y_update(_x, _y, self.D)
        y = rearrange(_y, "b h l n -> b l (h n)")

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
