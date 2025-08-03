from pathlib import Path

from ruamel.yaml import YAML

from ..devices import GPU, PIM
from .system import System

yaml = YAML(typ="safe")
gpu_configs = yaml.load(Path(__file__).parent / "gpu.yaml")
pim_configs = yaml.load(Path(__file__).parent / "pim.yaml")


def create_system(system_name: str, gpu_name: str, num_gpus: int):
    match system_name:
        case "GPU":
            state_dbyte = 2
            pim = None
        case "GPU+Q":
            state_dbyte = 1
            pim = None
        case "GPU+PIM":
            state_dbyte = 2
            pim = "HBM-PIM"
        case "Pimba":
            state_dbyte = 1
            pim = "Pimba"
        case "Pipelined":
            state_dbyte = 2
            pim = "Pipelined"
        case "Time-multiplexed":
            state_dbyte = 2
            pim = "Time-multiplexed"
        case _:
            raise RuntimeError(f"{system_name} is not in the system list!")

    gpu = GPU(
        **gpu_configs[gpu_name] | {"state_dbyte": state_dbyte, "num_gpus": num_gpus}
    )
    if pim is not None:
        pim = PIM(
            **pim_configs[pim]
            | {"num_channels": gpu.num_hbm_stacks * 8, "hbm_freq": gpu.hbm_freq}
        )

    return System(gpu, pim)
