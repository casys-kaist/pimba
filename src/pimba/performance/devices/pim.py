import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

from ruamel.yaml import YAML

from .. import configs, layers
from ..utils import register
from ..utils.file import get_simulator_bin_path
from ..utils.trace import Trace
from .base import Device, Handler

yaml = YAML(typ="safe")
yaml.default_flow_style = False


@dataclass(frozen=True)
class LayerInfo:
    hw: str
    op_type: str
    num_op: int
    head_dim: int
    state_dim: int
    state_dbyte: int


@dataclass
class Result:
    time: float
    power_1: float
    power_2: float


@dataclass
class PIM(Device):
    hw: str
    use_command_scheduling: bool
    use_chunk_group: bool
    state_dbyte: int
    energy: dict[str, float]
    num_channels: int = 1
    hbm_freq: int = 1000

    _cache: dict[LayerInfo, Result] = field(default_factory=dict)


def interpolate_result(a: Result, b: Result, k: float):
    return Result(
        time=a.time + k * (b.time - a.time),
        power_1=a.power_1 + k * (b.power_1 - a.power_1),
        power_2=a.power_2 + k * (b.power_2 - a.power_2),
    )


def run_ramulator2(cls, device: PIM, layer_info: LayerInfo):
    # generate trace
    trace = Trace(
        layer_info.op_type,
        layer_info.num_op,
        (layer_info.head_dim, layer_info.state_dim),
        device.hw,
        device.num_channels,
        layer_info.state_dbyte,
        device.use_chunk_group,
        device.use_command_scheduling,
    )

    with (
        NamedTemporaryFile("w", delete=True) as trace_file,
        NamedTemporaryFile("w", delete=True) as pim_config_file,
        NamedTemporaryFile("r", delete=True) as output_file,
    ):
        trace_file.write(trace.generate())  # type: ignore

        # get base pim config
        pim_config = yaml.load(Path(__file__).parent / "base_pim_config.yaml")

        # set channels
        pim_config["Frontend"]["path"] = trace_file.name
        pim_config["MemorySystem"]["DRAM"]["org"]["channel"] = device.num_channels

        # save config file
        yaml.dump(pim_config, Path(pim_config_file.name))

        # run ramulator2
        bin_path = get_simulator_bin_path()
        res = subprocess.run(
            f"{bin_path} -f {pim_config_file.name} -o {output_file.name}",
            shell=True,
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Ramulator2 failed to run: {res.stderr.decode()}")

        # result
        res = yaml.load(Path(output_file.name))

    time = res["cycles"] / (device.hbm_freq * 1000 * 1000)
    num_act = res["num_act"]
    num_rdwr = res["num_rdwr"]
    num_comp = res["num_comp"]
    power_1 = num_rdwr * device.energy["rdwr"]
    power_2 = num_act * device.energy["act"] + num_comp * device.energy["comp"]

    res = Result(time=time, power_1=power_1, power_2=power_2)
    device._cache[layer_info] = res


@register(PIM, layers.SU)
class PIM_SU(Handler):
    @classmethod
    def time(cls, device: PIM, layer: layers.SU) -> float:
        layer_info = LayerInfo(
            device.hw, "SU", layer.num_op * layer.m, layer.n, layer.k, layer.dbyte
        )
        if layer_info not in device._cache:
            run_ramulator2(cls, device, layer_info)
        return device._cache[layer_info].time

    @classmethod
    def power(cls, device: PIM, layer: layers.SU) -> tuple[float, float]:
        layer_info = LayerInfo(
            device.hw, "SU", layer.num_op * layer.m, layer.n, layer.k, layer.dbyte
        )
        if layer_info not in device._cache:
            run_ramulator2(cls, device, layer_info)
        return device._cache[layer_info].power_1, device._cache[layer_info].power_2


@register(PIM, layers.MATMUL)
class PIM_MATMUL(Handler):
    @classmethod
    def time(cls, device: PIM, layer: layers.MATMUL) -> float:
        if layer.name == "attention_qk":
            layer_info = LayerInfo(
                device.hw, "SCORE", layer.num_op, layer.k, layer.n, layer.dbyte
            )
        else:
            layer_info = LayerInfo(
                device.hw, "ATTEND", layer.num_op, layer.n, layer.k, layer.dbyte
            )

        info = {k: v for k, v in layer_info.__dict__.items() if k != "state_dim"}
        layer_start = LayerInfo(state_dim=configs.ATTENTION_RANGE[0], **info)
        if layer_start not in device._cache:
            run_ramulator2(cls, device, layer_start)

        layer_end = LayerInfo(state_dim=configs.ATTENTION_RANGE[1], **info)
        if layer_end not in device._cache:
            run_ramulator2(cls, device, layer_end)

        k = (layer_info.state_dim - layer_start.state_dim) / (
            layer_end.state_dim - layer_start.state_dim
        )
        res = interpolate_result(
            device._cache[layer_start], device._cache[layer_end], k
        )

        return res.time

    @classmethod
    def power(cls, device: PIM, layer: layers.MATMUL) -> tuple[float, float]:
        if layer.name == "attention_qk":
            layer_info = LayerInfo(
                device.hw, "SCORE", layer.num_op, layer.k, layer.n, layer.dbyte
            )
        else:
            layer_info = LayerInfo(
                device.hw, "ATTEND", layer.num_op, layer.n, layer.k, layer.dbyte
            )

        info = {k: v for k, v in layer_info.__dict__.items() if k != "state_dim"}
        layer_start = LayerInfo(state_dim=configs.ATTENTION_RANGE[0], **info)
        if layer_start not in device._cache:
            run_ramulator2(cls, device, layer_start)

        layer_end = LayerInfo(state_dim=configs.ATTENTION_RANGE[1], **info)
        if layer_end not in device._cache:
            run_ramulator2(cls, device, layer_end)

        k = (layer_info.state_dim - layer_start.state_dim) / (
            layer_end.state_dim - layer_start.state_dim
        )
        res = interpolate_result(
            device._cache[layer_start], device._cache[layer_end], k
        )

        return res.power_1, res.power_2
