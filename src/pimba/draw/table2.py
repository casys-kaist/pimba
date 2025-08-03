import re

import pandas as pd
from scipy.stats import gmean

from .utils import get_acc_res, get_output_path


def draw_table2():
    def f(data):
        res = []

        for model in [
            "retnet-2.7b",
            "gla-2.7b",
            "hgrn2-2.7b",
            "mamba2-2.7b",
            "zamba2-7b",
            "opt-7b",
        ]:
            architecture, _ = re.findall(r"(.*)-(.*)", model)[0]
            d_gpu = [architecture, "GPU"]
            d_pimba = [architecture, "Pimba"]
            for quant, use_sr in [("none", False), ("mx8", True)]:
                # NOTE: We found that "bfloat16" kernel shows much lower perplexity than "none" kernel in HGRN2.
                # Therefore, we used the "bfloat16" kernel for the HGRN2 model, for stronger baseline.
                # Note that, this is favorable to GPU baseline.
                if model == "hgrn2-2.7b" and quant == "none":
                    quant = "bfloat16"
                    use_sr = False

                for x in data:
                    if x["exp"] == {"model": model, "quant": quant, "use_sr": use_sr}:
                        if quant == "mx8":
                            d = d_pimba
                        else:
                            d = d_gpu
                        for data_name in [
                            "wikitext",
                            "piqa",
                            "lambada_standard",
                            "hellaswag",
                            "arc_easy",
                            "arc_challenge",
                            "winogrande",
                        ]:
                            d.append(x["result"][data_name])
            for d in [d_gpu, d_pimba]:
                d.append(float(gmean(d[3:])))
                d[3:] = [_d * 100 for _d in d[3:]]
                d[2:3] = [round(_d, 2) for _d in d[2:3]]
                d[3:] = [round(_d, 1) for _d in d[3:]]
            res.extend([d_gpu, d_pimba])
        return res

    data = get_acc_res()
    data = f(data)

    header = [
        "Model",
        "Method",
        "WikiText-2",
        "Piqa",
        "Lambada",
        "Hellaswag",
        "ARC-E",
        "ARC-C",
        "Winogrande",
        "Geomean",
    ]

    df = pd.DataFrame(data, columns=header)
    df.to_csv(get_output_path("table2", ext="csv"), index=False)
