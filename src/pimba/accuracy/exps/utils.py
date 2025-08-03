from collections import defaultdict
from dataclasses import dataclass

from ..datasets.utils import DEFAULT_DATASETS


@dataclass
class ExpData:
    model: str
    quant: str
    use_sr: bool
    datasets: list[str]


@dataclass(frozen=True)
class ExpKey:
    model: str
    quant: str
    use_sr: bool


def merge(exps: list[ExpData]) -> list[ExpData]:
    merged: dict[ExpKey, set] = defaultdict(set)

    for exp in exps:
        merged[ExpKey(exp.model, exp.quant, exp.use_sr)].update(exp.datasets)

    return [
        ExpData(model=k.model, quant=k.quant, use_sr=k.use_sr, datasets=sorted(v))
        for k, v in merged.items()
    ]


def create_accuracy_exps(exp_name_list: list[str]) -> list[ExpData]:
    exps = []

    for name in exp_name_list:
        match name:
            case "table2":
                for model in [
                    "retnet-2.7b",
                    "gla-2.7b",
                    "hgrn2-2.7b",
                    "mamba2-2.7b",
                    "zamba2-7b",
                    "opt-7b",
                ]:
                    for quant in [("none", False), ("mx8", True)]:
                        # NOTE: We found that "bfloat16" kernel shows much lower perplexity than "none" kernel in HGRN2.
                        # Therefore, we used the "bfloat16" kernel for the HGRN2 model, for stronger baseline.
                        # Note that, this is favorable to GPU baseline.
                        if model == "hgrn2-2.7b" and quant == ("none", False):
                            quant = ("bfloat16", False)

                        exps.append(
                            ExpData(
                                model=model,
                                quant=quant[0],
                                use_sr=quant[1],
                                datasets=DEFAULT_DATASETS,
                            )
                        )
            case "figure4":
                for model in [
                    "retnet-2.7b",
                    "gla-2.7b",
                    "mamba2-2.7b",
                    "opt-2.7b",
                    "llama-2.7b",
                ]:
                    for quant in [
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
                        exps.append(
                            ExpData(
                                model=model,
                                quant=quant[0],
                                use_sr=quant[1],
                                datasets=["wikitext"],
                            )
                        )
            case "figure6":
                for quant in [
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
                    exps.append(
                        ExpData(
                            model="mamba2-2.7b",
                            quant=quant[0],
                            use_sr=quant[1],
                            datasets=["wikitext"],
                        )
                    )

    return merge(exps)
