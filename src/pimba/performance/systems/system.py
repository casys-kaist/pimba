from dataclasses import dataclass

import pandas as pd

from ..devices import GPU, PIM, Device
from ..layers import Layer
from ..models import Model


@dataclass
class Result:
    phase: str
    category: str
    name: str
    time: float
    power_1: float
    power_2: float


class System:
    def __init__(self, gpu: GPU, pim: PIM | None):
        self.gpu = gpu
        self.pim = pim

    @property
    def tp(self) -> int:
        return self.gpu.num_gpus

    @property
    def hetero_system(self) -> bool:
        if self.pim is not None:
            return True
        else:
            return False

    def mapping(self, layer: Layer) -> Device:
        if (
            self.pim is not None
            and layer.phase == "gen"
            and (layer.category == "state_update" or layer.category == "gemv")
        ):
            return self.pim
        else:
            return self.gpu

    def simulate(self, model: Model):
        res: list[Result] = []
        for layer in model:
            device = self.mapping(layer)
            time, power_1, power_2 = device.simulate(layer)
            time, power_1, power_2 = (
                time * layer.num_layers,
                power_1 * layer.num_layers,
                power_2 * layer.num_layers,
            )
            res.append(
                Result(
                    layer.phase,
                    layer.category,
                    layer.name,
                    time,
                    power_1 * self.gpu.num_gpus,
                    power_2 * self.gpu.num_gpus,
                )
            )
        res_pd = pd.DataFrame(res)
        return res_pd
