import math

from .. import layers
from .base import Model


class Mamba2(Model):
    def __init__(
        self,
        num_decoders: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        state_dim: int,
        dbyte: int,
        batch: int,
        lin: int,
        lout: int,
        hetero_system: bool,
        tp: int,
        state_dbyte: int,
    ):
        super().__init__()

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
                num_decoders * (lout - 1),
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
                num_decoders * (lout - 1),
            )
        )
        self.layers.append(
            layers.ACT(
                "gen",
                "discretization",
                "discretization",
                batch,
                num_heads,
                1,
                1,
                dbyte,
                num_decoders * (lout - 1),
            )
        )
        if hetero_system:
            self.layers.append(
                layers.SU(
                    "gen",
                    "state_update",
                    "state_update",
                    math.ceil(num_heads / tp),
                    head_dim,
                    state_dim,
                    batch,
                    state_dbyte,
                    num_decoders * (lout - 1),
                )
            )
        else:
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "state_update",
                    "attention_kv",
                    head_dim,
                    state_dim,
                    1,
                    math.ceil(num_heads / tp) * batch,
                    state_dbyte,
                    num_decoders * (lout - 1),
                )
            )
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "state_update",
                    "attention_qk",
                    1,
                    state_dim,
                    head_dim,
                    math.ceil(num_heads / tp) * batch,
                    state_dbyte,
                    num_decoders * (lout - 1),
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
                num_decoders * (lout - 1),
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
                num_decoders * (lout - 1),
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
                num_decoders * (lout - 1),
            )
        )
        if tp > 1:
            self.layers.append(
                layers.G2G(
                    "gen",
                    "comm",
                    "g2g",
                    batch,
                    hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_decoders * (lout - 1),
                )
            )
