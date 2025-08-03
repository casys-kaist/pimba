import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .utils import get_output_path, get_perf_res

MODEL_ORDERS = ["retnet", "gla", "hgrn2", "mamba2", "zamba2"]
SYSTEM_ORDERS = ["GPU", "Time-multiplexed", "Pipelined"]
NUM_SYSTEMS = 3


def draw_figure5():
    def f(x):
        for model in [
            "retnet-2.7b",
            "gla-2.7b",
            "hgrn2-2.7b",
            "mamba2-2.7b",
            "zamba2-7b",
        ]:
            for system in ["GPU", "Time-multiplexed", "Pipelined"]:
                if x["exp"] == {
                    "model": model,
                    "num_gpus": 1,
                    "system_name": system,
                    "batch_size": 128,
                    "gpu": "A100",
                }:
                    return True
        return False

    def sort_key(x):
        x = x["exp"]

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

        return (model_number, system_number)

    data = get_perf_res()
    data = list(filter(f, data))
    data.sort(key=sort_key)
    data = [d["res"]["time"]["state_update"] for d in data]
    data = np.array(data).reshape(-1, NUM_SYSTEMS).T
    datas = data[0:1, :] / data

    # data
    area = [17.79, 32.4]

    # config
    bar_width = 0.22
    line_width = 1.5
    ax2_bar_width = 0.5
    ax2_bias = 0.8

    # variables
    count = len(datas[0])
    iter = len(datas)
    bias = -(1 - (iter - 1) * bar_width) / 2

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7, 2.4),
        gridspec_kw={
            "width_ratios": [7, 2],
        },
    )  # type: ignore
    assert isinstance(ax1, Axes)
    assert isinstance(ax2, Axes)

    fig.subplots_adjust(wspace=0.07)

    colors = [
        "#3D9D8F",
        "#F9D98D",
        "#E76F52",
    ]

    for i in range(iter):
        ax1.bar(
            np.arange(count) + i * bar_width,
            datas[i],
            bar_width,
            edgecolor="black",
            linewidth=line_width,
            color=colors[i],
            zorder=3,
        )
    ax2.bar(
        [0, 1],
        area,
        ax2_bar_width,
        edgecolor="black",
        linewidth=line_width,
        color=colors[1:3],
        zorder=3,
    )

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=1, zorder=0)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax1.set_xticks([bias + i for i in range(count + 1)])
    ax1.set_xlim(bias, count + bias)
    ax1.tick_params("x", length=20, width=line_width)

    ax2.set_xlim(-ax2_bias, 1 + ax2_bias)
    ax2.set_ylim(0, 35)
    ax2.set_xticks([])
    ax2.yaxis.tick_right()

    fig.savefig(
        get_output_path("figure5"), format="pdf", bbox_inches="tight", pad_inches=0.025
    )
