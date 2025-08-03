from .. import configs, layers
from .base import Model


class OPT(Model):
    def __init__(
        self,
        num_decoders: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
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
            # gen/qkv
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "qkv",
                    batch,
                    3 * (hidden_dim // tp),
                    hidden_dim,
                    1,
                    dbyte,
                    num_decoders,
                )
            )
            # gen/attention_qk
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "gemv",
                    "attention_qk",
                    1,
                    lin + step,
                    head_dim,
                    (num_heads // tp) * batch,
                    state_dbyte,
                    num_decoders,
                )
            )
            # gen/attention_softmax
            self.layers.append(
                layers.SOFTMAX(
                    "gen",
                    "others",
                    "attention_softmax",
                    1,
                    lin + step,
                    1,
                    (num_heads // tp) * batch,
                    dbyte,
                    num_decoders,
                )
            )
            # gen/attention_kv
            self.layers.append(
                layers.MATMUL(
                    "gen",
                    "gemv",
                    "attention_kv",
                    1,
                    head_dim,
                    lin + step,
                    (num_heads // tp) * batch,
                    state_dbyte,
                    num_decoders,
                )
            )
            # gen/down
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "proj",
                    batch,
                    hidden_dim,
                    (hidden_dim // tp),
                    1,
                    dbyte,
                    num_decoders,
                )
            )
            # gen?g2g_1
            if tp > 1:
                self.layers.append(
                    layers.G2G(
                        "gen",
                        "comm",
                        "g2g-1",
                        batch,
                        hidden_dim,
                        1,
                        1,
                        dbyte,
                        num_decoders,
                    )
                )
            # gen/norm_1
            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm_1",
                    batch,
                    hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_decoders,
                )
            )
            # gen/ffn_1
            self.layers.append(
                layers.FC(
                    "gen",
                    "gemm",
                    "ffn_1",
                    batch,
                    4 * (hidden_dim // tp),
                    hidden_dim,
                    1,
                    dbyte,
                    num_decoders,
                )
            )
            # gen/act
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
                    num_decoders,
                )
            )
            # gen/ffn_2
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
                    num_decoders,
                )
            )
            # gen/g2g_2
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
                        num_decoders,
                    )
                )
            # gen/norm_2
            self.layers.append(
                layers.NORM(
                    "gen",
                    "others",
                    "norm_2",
                    batch,
                    hidden_dim,
                    1,
                    1,
                    dbyte,
                    num_decoders,
                )
            )
