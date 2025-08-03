import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .utils import get_acc_res, get_output_path


def draw_figure6():
    def f(data):
        res = []

        for quant, use_sr in [
            ("mx8", False),
            ("mx8", True),
            ("e4m3", False),
            ("e4m3", True),
            ("e5m2", False),
            ("e5m2", True),
            ("int8", False),
            ("int8", True),
            ("none", False),
        ]:
            for x in data:
                if x["exp"] == {
                    "model": "mamba2-2.7b",
                    "quant": quant,
                    "use_sr": use_sr,
                }:
                    res.append(x["result"]["wikitext"])
        return res

    # NOTE: orders: mx8, mx8_sr, e4m3, e4m3_sr, e5m2, e5m2_sr, int8, int8_sr, fp16
    data = get_acc_res()
    perplexity = f(data)

    # NOTE: we derived these numbers using Synopsys Design Compiler tool.
    area = [
        19.91075707,
        21.00365576,
        22.54227987,
        23.63517856,
        21.14093776,
        22.23383645,
        65.92677316,
        67.01967185,
        32.44301734,
    ]

    line_width = 1.5

    colors = [
        "#E76F52",
        "#E9C46A",
        "#3D9D8F",
        "#3a86ff",
    ]

    color = [
        colors[0],
        colors[0],
        colors[1],
        colors[1],
        colors[1],
        colors[1],
        colors[2],
        colors[2],
        colors[3],
    ]

    markers = [
        "o",
        "o",
        "X",
        "X",
        "X",
        "X",
        "P",
        "P",
        "D",
    ]

    # brokenaxes 설정
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2,
        2,
        figsize=(8, 3.5),
        height_ratios=[1, 5],
        width_ratios=[4, 1],
    )  # type: ignore
    assert isinstance(ax1, Axes)
    assert isinstance(ax2, Axes)
    assert isinstance(ax3, Axes)
    assert isinstance(ax4, Axes)

    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for ax in (ax1, ax2, ax3, ax4):
        for i in range(len(perplexity)):
            ax.scatter(
                area[i],
                perplexity[i],
                color=color[i],
                marker=markers[i],
                s=150,
                zorder=3,
                linewidths=1.25,
                edgecolors="black",
            )

    for ax in (ax1, ax2):
        ax.set_ylim(60, 65)
        ax.set_yticks([60, 65])
    for ax in (ax3, ax4):
        ax.set_ylim(5, 30.05)
    for ax in (ax1, ax3):
        ax.set_xlim(18, 34.05)
    for ax in (ax2, ax4):
        ax.set_xlim(64, 68)

    for ax in (ax1, ax2, ax3, ax4):
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)
            spine.set_zorder(4)
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=1, zorder=0)
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=1, zorder=0)

    ax1.set_xticklabels([])
    ax1.tick_params("x", which="major", length=0)
    for s in ("bottom", "right"):
        ax1.spines[s].set_visible(False)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params("both", which="major", length=0)
    for s in ("bottom", "left"):
        ax2.spines[s].set_visible(False)

    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    ax4.set_yticklabels([])
    ax4.tick_params("y", which="major", length=0)
    for s in ("top", "left"):
        ax4.spines[s].set_visible(False)

    d = 1.0  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=line_width,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 0], transform=ax2.transAxes, **kwargs)
    ax3.plot([0, 1], [1, 0], transform=ax3.transAxes, **kwargs)
    ax4.plot([0, 1], [0, 1], transform=ax4.transAxes, **kwargs)

    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    ax1.tick_params("y", which="major", width=line_width)
    ax3.tick_params("y", which="major", width=line_width)
    ax3.tick_params("x", which="major", width=line_width)
    ax4.tick_params("x", which="major", width=line_width)

    fig.savefig(get_output_path("figure6"), format="pdf", bbox_inches="tight")
