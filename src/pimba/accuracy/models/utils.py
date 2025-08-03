from dataclasses import dataclass
from typing import Type

from huggingface_hub import snapshot_download
from lm_eval.models.huggingface import HFLM

from .gla.model import GLA
from .hgrn2.model import HGRN2
from .llama.model import LLaMA
from .mamba2.model import Mamba2
from .opt.model import OPT
from .retnet.model import RetNet
from .zamba2.model import Zamba2


@dataclass
class ModelInfo:
    arch: Type[HFLM]
    name: str
    revision: str


MODELS = {
    # GLA
    "gla-2.7b": ModelInfo(
        GLA, "fla-hub/gla-2.7B-100B", "55c00820ebf3bdd482575bdd50dc6bf85e7526ff"
    ),
    # HGRN2
    "hgrn2-2.7b": ModelInfo(
        HGRN2, "fla-hub/hgrn2-2.7B-100B", "30c023a24acb3ec7b9d6a5e2d91649c533814588"
    ),
    # LLaMA
    "llama-2.7b": ModelInfo(
        LLaMA,
        "princeton-nlp/Sheared-LLaMA-2.7B",
        "2f157a0306b75d37694ae05f6a4067220254d540",
    ),
    # Mamba2
    "mamba2-2.7b": ModelInfo(
        Mamba2, "state-spaces/mamba2-2.7b", "99b226cc377d131cccc610ed4346db564f381f1e"
    ),
    # OPT
    "opt-2.7b": ModelInfo(
        OPT, "facebook/opt-2.7b", "905a4b602cda5c501f1b3a2650a4152680238254"
    ),
    "opt-7b": ModelInfo(
        OPT, "facebook/opt-6.7b", "a45aa65bbeb77c1558bc99bedc6779195462dab0"
    ),
    # RetNet
    "retnet-2.7b": ModelInfo(
        RetNet,
        "fla-hub/retnet-2.7B-100B",
        "8b60668df2cefce9584d39bd12aff6c6927a8628",
    ),
    # Zamba2
    "zamba2-7b": ModelInfo(
        Zamba2,
        "Zyphra/Zamba2-7B-Instruct-v2",
        "54438e69d6391c8664cbf8a0c6789477393ad56f",
    ),
}


def create_model(name: str) -> HFLM:
    if name not in MODELS.keys():
        raise RuntimeError(f"{name} is not in the model list!")

    model_arch = MODELS[name].arch
    model_name = MODELS[name].name
    revision = MODELS[name].revision
    model = model_arch(model_name, revision=revision)

    return model
