import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator

from .utils import get_output_path, get_perf_res

MODEL_ORDERS = ["retnet", "gla", "hgrn2", "mamba2", "zamba2"]
SYSTEM_ORDERS = ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]
NUM_SYSTEMS = 4


def draw_figure3():
    def f(x):
        for model in [
            "retnet-2.7b",
            "gla-2.7b",
            "hgrn2-2.7b",
            "mamba2-2.7b",
            "zamba2-7b",
        ]:
            for batch_size in [32, 64, 128]:
                if x["exp"] == {
                    "model": model,
                    "num_gpus": 1,
                    "system_name": "GPU",
                    "batch_size": batch_size,
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
        state_update = 0.0 if "state_update" not in x else x["state_update"]
        gemv = 0.0 if "gemv" not in x else x["gemv"]
        discretization = 0.0 if "discretization" not in x else x["discretization"]
        causal_conv1d = 0.0 if "causal_conv1d" not in x else x["causal_conv1d"]
        gemm = 0.0 if "gemm" not in x else x["gemm"]
        others = 0.0 if "others" not in x else x["others"]

        return (state_update, gemv, discretization, causal_conv1d, gemm, others)

    data = get_perf_res()
    data = list(filter(f, data))
    data.sort(key=sort_key)
    data = [get_breakdown(d["res"]["time"]) for d in data]
    data = np.array(data).T
    data = data / data.sum(axis=0)[None, :]

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
    bar_width = 0.6
    line_width = 1.5

    # variables
    count = len(data[0])
    iter = len(data)
    bias = -0.5

    fig, ax = plt.subplots(figsize=(7, 2.4))
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

    ax.set_ylim(0, 1.0)
    ax.set_xlim(bias, count + bias)
    ax.xaxis.set_major_locator(FixedLocator([bias + i for i in range(0, count + 1, 3)]))
    ax.xaxis.set_minor_locator(
        FixedLocator([bias + i for i in range(0, count + 1) if i % 3 != 0])
    )
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.4, 0.6, 0.8, 1.0]))

    ax.tick_params("x", which="major", length=37, width=line_width)
    ax.tick_params("x", which="minor", length=6, width=line_width)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig(
        get_output_path("figure3"), format="pdf", bbox_inches="tight", pad_inches=0.025
    )
