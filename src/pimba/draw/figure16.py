import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator

from .utils import get_output_path, get_perf_res

MODEL_ORDERS = ["retnet", "gla", "hgrn2", "mamba2", "zamba2", "opt"]
SYSTEM_ORDERS = ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]
NUM_SYSTEMS = 4


def draw_figure16():
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
                for batch_size in [32, 64, 128]:
                    if x["exp"] == {
                        "model": model,
                        "num_gpus": num_gpus,
                        "system_name": system,
                        "batch_size": batch_size,
                        "gpu": "H100",
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

    data = get_perf_res()
    data = list(filter(f, data))
    data.sort(key=sort_key)
    data = [sum(d["res"]["time"].values()) for d in data]
    data = np.array(data)
    data = data.reshape(-1, NUM_SYSTEMS).T
    data = data[0:1, :] / data

    # config
    colors = [
        # "#AFDAD3",
        "#2a9d8f",
        "#FFECC0",
        "#e9c46a",
        "#e76f51",
    ]
    bar_width = 0.165
    line_width = 2

    # variables
    count = len(data[0])
    iter = len(data)
    bias = -(1 - (iter - 1) * bar_width) / 2

    fig, ax = plt.subplots(figsize=(10, 2.5))  # type: ignore
    assert isinstance(ax, Axes)

    for i in range(iter):
        ax.bar(
            np.arange(count) + i * bar_width,
            data[i],
            bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=line_width,
            zorder=3,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(line_width)
        spine.set_zorder(4)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=1, zorder=0)

    ax.set_xlim(bias, count + bias)

    ax.xaxis.set_major_locator(FixedLocator([bias + i for i in range(0, count + 1, 3)]))
    ax.xaxis.set_minor_locator(
        FixedLocator([bias + i for i in range(0, count + 1) if i % 3 != 0])
    )
    ax.yaxis.set_major_locator(FixedLocator([1, 2]))

    ax.set_xticklabels([])
    ax.tick_params("x", which="major", length=40, width=line_width)
    ax.tick_params("x", which="minor", length=12, width=line_width)
    ax.set_yticklabels([])

    fig.savefig(
        get_output_path("figure16"), format="pdf", bbox_inches="tight", pad_inches=0.025
    )
