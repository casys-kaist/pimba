import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

from .utils import get_acc_res, get_output_path


def draw_figure4():
    def f(data):
        res = []

        for model in [
            "llama-2.7b",
            "opt-2.7b",
            "retnet-2.7b",
            "gla-2.7b",
            "mamba2-2.7b",
        ]:
            d = []
            for quant, use_sr in [
                ("none", False),
                ("int8", False),
                ("int8", True),
                ("e4m3", False),
                ("e4m3", True),
                ("e5m2", False),
                ("e5m2", True),
                ("mx8", False),
                ("mx8", True),
            ]:
                for x in data:
                    if x["exp"] == {"model": model, "quant": quant, "use_sr": use_sr}:
                        d.append(x["result"]["wikitext"])
            res.append(d)
        return res

    data = get_acc_res()
    data = f(data)
    data = np.array(data)

    num_models = len(data)
    count = len(data[0])

    # Create a figure with two subplots
    fig, axs = plt.subplots(
        1, num_models, figsize=(16, 1.85), gridspec_kw={"wspace": 0.22}
    )  # type: ignore

    colors = [
        "#BAD8FF",
        "#AEDAD3",
        "#3D9D8F",
        "#FFF8E7",
        "#FFECC0",
        "#E1BE6B",
        "#C8A242",
        "#E7BAAB",
        "#D55C3D",
    ]

    bar_width = 1.0
    line_width = 1.5

    for i, ax in enumerate(axs):
        ax.bar(
            np.arange(count),
            data[i],
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=line_width,
            zorder=3,
        )
        ax.set_xlim(-2, count + 1)
        ax.set_ylim(0, 30)
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)
            spine.set_zorder(4)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=1, zorder=0)
        ax.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25, 30]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params("x", which="major", width=0)
        ax.tick_params("y", which="major", width=line_width)

    fig.savefig(get_output_path("figure4"), format="pdf", bbox_inches="tight")
