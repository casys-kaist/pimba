import math
from dataclasses import dataclass, field

from .. import layers
from ..layers import Layer
from ..utils import register
from .base import Device, Handler


@dataclass
class GPU(Device):
    num_gpus: int
    num_hbm_stacks: int
    hbm_freq: int
    num_cores: int
    l1_cache_size: int
    l2_cache_size: int
    peak_flops: int
    peak_memory_bandwidth: int
    max_interface_bandwidth: int
    max_compute_util: float
    max_memory_util: float
    state_dbyte: int
    energy: dict[str, float]

    _table_tiles: dict = field(default_factory=dict)

    def get_traffic_for_tile(self, tm, tn, layer: Layer):
        m, n, k, num_op, dbyte = layer.get_infos()
        traffic = [math.ceil(n / tn) * m * k, math.ceil(m / tm) * n * k, m * n]
        traffic = [i * dbyte * num_op for i in traffic]
        return traffic

    def get_optimal_tile(self, layer: Layer, tb_scale):
        config = layer.get_infos()
        (m, n, k, num_op, dbyte) = config

        if config in self._table_tiles:
            return self._table_tiles[config]

        trange = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]

        # find L1 tile size
        l1_tk = 32
        opt_config = [0, 0]
        min_cost = float("inf")
        for l1_tm in trange:
            for l1_tn in trange:
                l1_tm = min(l1_tm, m)
                l1_tn = min(l1_tn, n)
                required_capacity = (
                    l1_tm + l1_tn
                ) * l1_tk * dbyte + l1_tm * l1_tn * dbyte
                if required_capacity > self.l1_cache_size:
                    continue
                l2_access = sum(self.get_traffic_for_tile(l1_tm, l1_tn, layer))

                ## applying SM underutilization to cost function
                num_threadblock = tb_scale(m, l1_tm, n, l1_tn) * num_op

                tmp = math.ceil(num_threadblock / self.num_cores) * self.num_cores
                core_utilization = num_threadblock / tmp
                cost = l2_access * pow((1 / core_utilization), 2)
                if cost < min_cost:
                    min_cost = cost
                    opt_config = [l1_tm, l1_tn]

        l1_tm, l1_tn = opt_config

        # find L2 tile size
        ## experimentally found L2 tile_k size
        l2_tk = k / 64

        min_access = float("inf")
        opt_config = [0, 0]
        for l2_tm in [l1_tm * i for i in range(1, int(m / l1_tm) + 1)] + [m]:
            for l2_tn in [l1_tn * i for i in range(1, int(n / l1_tn) + 1)] + [n]:
                l2_tm = min(l2_tm, m)
                l2_tn = min(l2_tn, n)
                required_capacity = (
                    l2_tm + l2_tn
                ) * l2_tk * dbyte + l2_tm * l2_tn * dbyte
                if required_capacity > self.l2_cache_size:
                    if l2_tm != l1_tm or l2_tn != l1_tn:
                        continue

                access = (
                    math.ceil(m / l2_tm) * n * k * dbyte
                    + math.ceil(n / l2_tn) * m * k * dbyte
                    + m * n * dbyte
                )

                if access < min_access:
                    min_access = access
                    opt_config = [l2_tm, l2_tn]

        l2_tm, l2_tn = opt_config
        out_tiles = [l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk]
        self._table_tiles[config] = out_tiles
        return out_tiles

    def compute_time(self, layer: Layer, tb_scale):
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self.get_optimal_tile(
            layer, tb_scale
        )
        m, n, k, num_op, dbyte = layer.get_infos()
        flops = self.peak_flops * self.max_compute_util

        num_threadblock = tb_scale(m, l1_tm, n, l1_tn) * num_op

        tmp = math.ceil(num_threadblock / self.num_cores) * self.num_cores
        core_utilization = num_threadblock / tmp

        flops = flops * core_utilization

        ## e.g., peak flops of FP8  is twice that of FP16
        flops *= int(2 / dbyte)

        return layer.get_flops() / flops

    def get_energy(self, layer: Layer, mem, l2, l1, reg):
        e_off = mem * self.energy["mem"]
        e_l2 = l2 * self.energy["l2"]
        e_l1 = l1 * self.energy["l1"]
        e_reg = reg * self.energy["reg"]
        e_flop = layer.get_flops() / 2 * self.energy["alu"]

        energy_1 = e_off
        energy_2 = sum([e_l2, e_l1, e_reg, e_flop])

        return energy_1, energy_2


def fc_scale(m, l1_tm, n, l1_tn):
    return math.ceil(m / l1_tm) * math.ceil(n / l1_tn)


def no_scale(a, b, c, d):
    return 1


@register(GPU, layers.FC)
class GPU_FC(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.FC) -> float:
        # compute time
        compute_time = device.compute_time(layer, fc_scale)

        # memory time
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = device.get_optimal_tile(
            layer, fc_scale
        )
        m, n, k, num_op, dbyte = layer.get_infos()

        off_data = device.get_traffic_for_tile(l2_tm, l2_tn, layer)

        mem_bw = device.peak_memory_bandwidth * device.max_memory_util
        num_threadblock = math.ceil(m / l1_tm) * math.ceil(n / l1_tn) * num_op
        tmp = math.ceil(num_threadblock / device.num_cores) * device.num_cores
        core_utilization = num_threadblock / tmp
        mem_bw = mem_bw * core_utilization

        memory_time = sum(off_data) / mem_bw

        return max(compute_time, memory_time)

    @classmethod
    def power(cls, device: GPU, layer: layers.FC) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = device.get_optimal_tile(
            layer, fc_scale
        )
        reg_tm, reg_tn = 16, 16

        mem = sum(device.get_traffic_for_tile(l2_tm, l2_tn, layer))
        l2 = sum(device.get_traffic_for_tile(l1_tm, l1_tn, layer))
        l1 = sum(device.get_traffic_for_tile(reg_tm, reg_tn, layer))
        reg = sum([m * n * k, m * n * k, m * n * k])

        return device.get_energy(layer, mem, l2, l1, reg)


@register(GPU, layers.MATMUL)
class GPU_MATMUL(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.MATMUL) -> float:
        # compute time
        compute_time = device.compute_time(layer, no_scale)

        # memory time
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = device.get_optimal_tile(
            layer, no_scale
        )
        m, n, k, num_op, dbyte = layer.get_infos()

        off_data = device.get_traffic_for_tile(l2_tm, l2_tn, layer)

        mem_bw = device.peak_memory_bandwidth * device.max_memory_util
        num_threadblock = num_op
        tmp = math.ceil(num_threadblock / device.num_cores) * device.num_cores
        core_utilization = num_threadblock / tmp
        mem_bw = mem_bw * core_utilization

        memory_time = sum(off_data) / mem_bw

        return max(compute_time, memory_time)

    @classmethod
    def power(cls, device: GPU, layer: layers.MATMUL) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = device.get_optimal_tile(
            layer, fc_scale
        )
        reg_tm, reg_tn = 16, 16

        mem = sum(device.get_traffic_for_tile(l2_tm, l2_tn, layer))
        l2 = sum(device.get_traffic_for_tile(l1_tm, l1_tn, layer))
        l1 = sum(device.get_traffic_for_tile(reg_tm, reg_tn, layer))
        reg = sum([m * n * k, m * n * k, m * n * k])

        return device.get_energy(layer, mem, l2, l1, reg)


@register(GPU, layers.TO_PIM)
class GPU_TO_PIM(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.TO_PIM) -> float:
        m, n, k, num_op, dbyte = layer.get_infos()
        traffic = m * n * num_op * dbyte
        interface_bw = device.peak_memory_bandwidth / 2
        return traffic / interface_bw

    @classmethod
    def power(cls, device: GPU, layer: layers.TO_PIM) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()
        traffic = m * n * num_op * dbyte

        return traffic * device.energy["mem"], 0.0


@register(GPU, layers.SOFTMAX)
class GPU_SOFTMAX(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.SOFTMAX) -> float:
        # compute time
        compute_time = device.compute_time(layer, no_scale)

        # memory time
        m, n, k, num_op, dbyte = layer.get_infos()

        off_data = num_op * m * n * dbyte * 2

        # from: https://github.com/upmem/upmem_llm_framework
        memory_time = (
            (
                0.000000615
                * (1555 * 1000 * 1000 * 1000 / device.peak_memory_bandwidth)
                * off_data
                + 6.87
            )
            / 1000
            / 1000
        )

        return max(compute_time, memory_time)

    @classmethod
    def power(cls, device: GPU, layer: layers.SOFTMAX) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()

        mem = m * n * num_op * dbyte * 2
        return device.get_energy(layer, mem, mem, mem, mem)


@register(GPU, layers.G2G)
class GPU_G2G(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.G2G) -> float:
        def get_nvlink_time(size):
            approx_ns_time = 6060 + 0.009 * size * (
                600 * 1000 * 1000 * 1000 / device.max_interface_bandwidth
            )
            approx_time = approx_ns_time / 1000 / 1000 / 1000
            return max(approx_time, size / (device.max_interface_bandwidth / 2))

        m, n, k, num_op, dbyte = layer.get_infos()
        traffic = m * n * num_op * dbyte
        exec_time = get_nvlink_time(traffic) * math.log2(device.num_gpus)
        return exec_time

    @classmethod
    def power(cls, device: GPU, layer: layers.G2G) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()
        traffic = m * n * num_op * dbyte

        return traffic * device.energy["mem"], 0.0


@register(GPU, layers.NORM)
class GPU_NORM(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.NORM) -> float:
        # compute time
        compute_time = device.compute_time(layer, no_scale)

        # memory time
        m, n, k, num_op, dbyte = layer.get_infos()

        off_data = num_op * m * n * dbyte * 3

        memory_time = (
            (
                0.0000016
                * (1555 * 1000 * 1000 * 1000 / device.peak_memory_bandwidth)
                * off_data
                + 6.87
            )
            / 1000
            / 1000
        )

        return max(compute_time, memory_time)

    @classmethod
    def power(cls, device: GPU, layer: layers.NORM) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()

        mem = m * n * num_op * dbyte * 3
        return device.get_energy(layer, mem, mem, mem, mem)


@register(GPU, layers.ACT)
class GPU_ACT(Handler):
    @classmethod
    def time(cls, device: GPU, layer: layers.ACT) -> float:
        # compute time
        compute_time = device.compute_time(layer, no_scale)

        # memory time
        m, n, k, num_op, dbyte = layer.get_infos()

        off_data = num_op * m * n * dbyte * 2

        memory_time = (
            (
                0.000000447
                * (1555 * 1000 * 1000 * 1000 / device.peak_memory_bandwidth)
                * off_data
                + 8.29
            )
            / 1000
            / 1000
        )

        return max(compute_time, memory_time)

    @classmethod
    def power(cls, device: GPU, layer: layers.ACT) -> tuple[float, float]:
        m, n, k, num_op, dbyte = layer.get_infos()

        mem = m * n * num_op * dbyte * 2
        return device.get_energy(layer, mem, mem, mem, mem)
