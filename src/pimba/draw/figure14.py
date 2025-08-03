from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator

from .utils import get_output_path, get_perf_res

MODEL_ORDERS = ["retnet", "gla", "hgrn2", "mamba2", "zamba2", "opt"]
SYSTEM_ORDERS = ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]
NUM_SYSTEMS = 4


def draw_figure14():
    def f(x):
        for model, num_gpus in [
            ("retnet-70b", 8),
            ("gla-70b", 8),
            ("hgrn2-70b", 8),
            ("mamba2-70b", 8),
            ("zamba2-70b", 8),
            ("opt-70b", 8),
        ]:
            for system in ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]:
                if x["exp"] == {
                    "model": model,
                    "num_gpus": num_gpus,
                    "system_name": system,
                    "batch_size": 128,
                    "gpu": "A100",
                }:
                    return True
        return False

    def sort_key(x):
        x = x["exp"]

        big_or_small = 0 if x["num_gpus"] == 1 else 1

        model: str = x["model"]
        model_number = -1
        for idx, name in enumerate(MODEL_ORDERS):
            if name in model:
                model_number = idx

        system: str = x["system_name"]
        system_number = -1
        for idx, name in enumerate(SYSTEM_ORDERS):
            if system == name:
                system_number = idx

        return (big_or_small, model_number, x["batch_size"], system_number)

    def get_breakdown(x):
        mem = defaultdict(float, x["power_1"])
        comp = defaultdict(float, x["power_2"])

        state_update_mem = mem["state_update"]
        state_update_comp = comp["state_update"]
        attention_mem = mem["gemv"]
        attention_comp = comp["gemv"]
        gemm = mem["gemm"] + comp["gemm"]
        others = 0
        for key in ["discretization", "causal_conv1d", "comm", "others"]:
            others += mem[key] + comp[key]

        return (
            state_update_mem,
            state_update_comp,
            attention_mem,
            attention_comp,
            gemm,
            others,
        )

    data = get_perf_res()
    data = list(filter(f, data))
    data.sort(key=sort_key)
    data = [get_breakdown(d["res"]) for d in data]
    data = np.array(data).T
    norm = data[:, ::NUM_SYSTEMS].sum(axis=0).repeat(4)
    data = data / norm[None, :]

    # config
    colors = [
        "#185952",
        "#2a9d8f",
        "#6aaa83",
        "#a9b776",
        "#e9c46a",
        "#F07354",
        "#AD4343",
    ]
    bar_width = 0.5
    line_width = 1.5

    # variables
    count = len(data[0])
    iter = len(data)
    bias = -0.5

    fig, ax = plt.subplots(figsize=(7, 2))
    assert isinstance(ax, Axes)

    for i in range(iter):
        ax.bar(
            np.arange(count),
            data[i],
            bar_width,
            bottom=data[:i].sum(axis=0),
            color=colors[i],
            edgecolor="black",
            linewidth=line_width,
            zorder=3,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(line_width)
        spine.set_zorder(4)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, zorder=0)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(bias, count + bias)
    ax.xaxis.set_major_locator(FixedLocator([bias + i for i in range(0, count + 1, 4)]))
    ax.xaxis.set_minor_locator(
        FixedLocator([bias + i for i in range(0, count + 1) if i % 4 != 0])
    )
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.4, 0.6, 0.8, 1.0]))

    ax.tick_params("x", which="major", length=80, width=line_width)
    ax.tick_params("x", which="minor", length=6, width=line_width)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig(
        get_output_path("figure14"), format="pdf", bbox_inches="tight", pad_inches=0.025
    )
