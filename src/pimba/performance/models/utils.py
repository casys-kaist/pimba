import re
from pathlib import Path

from ruamel.yaml import YAML

from .base import Model as Model
from .gla import GLA
from .hgrn2 import HGRN2
from .mamba2 import Mamba2
from .opt import OPT
from .retnet import RetNet
from .zamba2 import Zamba2

yaml = YAML(typ="safe")
models_configs = yaml.load(Path(__file__).parent / "models.yaml")


def create_model(
    model: str,
    batch: int,
    lin: int,
    lout: int,
    hetero_system: bool,
    tp: int,
    state_dbyte: int,
) -> Model:
    architecture, preset = re.findall(r"(.*)-(.*)", model)[0]

    match architecture:
        case "opt":
            model_cls = OPT
        case "retnet":
            model_cls = RetNet
        case "gla":
            model_cls = GLA
        case "hgrn2":
            model_cls = HGRN2
        case "mamba2":
            model_cls = Mamba2
        case "zamba2":
            model_cls = Zamba2
        case _:
            raise RuntimeError(f"Unsupported model architecture: {architecture}")

    model_config = models_configs[architecture][preset]

    return model_cls(
        **model_config,
        batch=batch,
        lin=lin,
        lout=lout,
        hetero_system=hetero_system,
        tp=tp,
        state_dbyte=state_dbyte,
    )
