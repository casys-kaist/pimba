import math

from .. import configs, layers
from .base import Model


class Zamba2(Model):
    def __init__(
        self,
        num_mamba_decoders: int,
        num_transformer_decoders: int,
        hidden_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_mamba_heads: int,
        mamba_head_dim: int,
        mamba_state_dim: int,
        dbyte: int,
        batch: int,
        lin: int,
        lout: int,
        hetero_system: bool,
        tp: int,
        state_dbyte: int,
    ):
        super().__init__()

        configs.ATTENTION_RANGE = (lin + 1, lin + lout)
        for step in range(1, lout):
            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm",
                    batch,
                    hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "qkv",
                    batch,
                    int(4.1 * (hidden_dim // tp)),
                    hidden_dim,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            self.layers.append(
                layers.ACT(
                    "gen",
                    "causal_conv1d",
                    "causal_conv1d",
                    batch,
                    int(8.4 * hidden_dim),
                    1,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            self.layers.append(
                layers.ACT(
                    "gen",
                    "discretization",
                    "discretization",
                    batch,
                    num_mamba_heads,
                    1,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            if hetero_system:
                self.layers.append(
                    layers.SU(
                        "gen",
                        "state_update",
                        "state_update",
                        math.ceil(num_mamba_heads / tp),
                        mamba_head_dim,
                        mamba_state_dim,
                        batch,
                        state_dbyte,
                        num_mamba_decoders,
                    )
                )
            else:
                self.layers.append(
                    layers.MATMUL(
                        "gen",
                        "state_update",
                        "attention_kv",
                        mamba_head_dim,
                        mamba_state_dim,
                        1,
                        math.ceil(num_mamba_heads / tp) * batch,
                        state_dbyte,
                        num_mamba_decoders,
                    )
                )
                self.layers.append(
                    layers.MATMUL(
                        "gen",
                        "state_update",
                        "attention_qk",
                        1,
                        mamba_state_dim,
                        mamba_head_dim,
                        math.ceil(num_mamba_heads / tp) * batch,
                        state_dbyte,
                        num_mamba_decoders,
                    )
                )
            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm",
                    batch,
                    2 * hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            self.layers.append(
                layers.ACT(
                    "gen",
                    "others",
                    "act",
                    batch,
                    2 * (hidden_dim // tp),
                    1,
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "o",
                    batch,
                    hidden_dim,
                    (2 * hidden_dim // tp),
                    1,
                    dbyte,
                    num_mamba_decoders,
                )
            )
            if tp > 1:
                self.layers.append(
                    layers.G2G(
                        "gen",
                        "comm",
                        "g2g_1",
                        batch,
                        hidden_dim,
                        1,
                        1,
                        dbyte,
                        num_mamba_decoders,
                    )
                )

            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm",
                    batch,
                    2 * hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "qkv",
                    batch,
                    6 * (hidden_dim // tp),
                    2 * hidden_dim,
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "gemv",
                    "attention_qk",
                    1,
                    lin + step,
                    attention_head_dim,
                    (num_attention_heads // tp) * batch,
                    state_dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.SOFTMAX(
                    "gen",
                    "others",
                    "attention_softmax",
                    1,
                    lin + step,
                    1,
                    (num_attention_heads // tp) * batch,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "gemv",
                    "attention_kv",
                    1,
                    attention_head_dim,
                    lin + step,
                    (num_attention_heads // tp) * batch,
                    state_dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "proj",
                    batch,
                    hidden_dim,
                    (2 * hidden_dim // tp),
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            if tp > 1:
                self.layers.append(
                    layers.G2G(
                        "gen",
                        "comm",
                        "g2g_1",
                        batch,
                        hidden_dim,
                        1,
                        1,
                        dbyte,
                        num_transformer_decoders,
                    )
                )
            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm",
                    batch,
                    hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "ffn_1",
                    batch,
                    8 * (hidden_dim // tp),
                    hidden_dim,
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.ACT(
                    "gen",
                    "others",
                    "act",
                    batch,
                    4 * (hidden_dim // tp),
                    1,
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "ffn_2",
                    batch,
                    hidden_dim,
                    4 * (hidden_dim // tp),
                    1,
                    dbyte,
                    num_transformer_decoders,
                )
            )
            if tp > 1:
                self.layers.append(
                    layers.G2G(
                        "gen",
                        "comm",
                        "g2g_2",
                        batch,
                        hidden_dim,
                        1,
                        1,
                        dbyte,
                        num_transformer_decoders,
                    )
                )
