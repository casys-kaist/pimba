from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpData:
    model: str
    system: str
    num_gpus: int
    batch_size: int
    gpu: str


def create_performance_exps(exp_name_list: list[str]) -> list[ExpData]:
    exps = []

    for name in exp_name_list:
        match name:
            case "figure3":
                for model in [
                    "retnet-2.7b",
                    "gla-2.7b",
                    "hgrn2-2.7b",
                    "mamba2-2.7b",
                    "zamba2-7b",
                ]:
                    for batch_size in [32, 64, 128]:
                        exps.append(
                            ExpData(
                                model=model,
                                system="GPU",
                                num_gpus=1,
                                batch_size=batch_size,
                                gpu="A100",
                            )
                        )
            case "figure5":
                for model in [
                    "retnet-2.7b",
                    "gla-2.7b",
                    "hgrn2-2.7b",
                    "mamba2-2.7b",
                    "zamba2-7b",
                ]:
                    for system in ["GPU", "Time-multiplexed", "Pipelined"]:
                        exps.append(
                            ExpData(
                                model=model,
                                system=system,
                                num_gpus=1,
                                batch_size=128,
                                gpu="A100",
                            )
                        )
            case "figure12":
                for model, num_gpus in [
                    ("retnet-2.7b", 1),
                    ("gla-2.7b", 1),
                    ("hgrn2-2.7b", 1),
                    ("mamba2-2.7b", 1),
                    ("zamba2-7b", 1),
                    ("opt-7b", 1),
                    ("retnet-70b", 8),
                    ("gla-70b", 8),
                    ("hgrn2-70b", 8),
                    ("mamba2-70b", 8),
                    ("zamba2-70b", 8),
                    ("opt-70b", 8),
                ]:
                    for system in ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]:
                        for batch_size in [32, 64, 128]:
                            exps.append(
                                ExpData(
                                    model=model,
                                    system=system,
                                    num_gpus=num_gpus,
                                    batch_size=batch_size,
                                    gpu="A100",
                                )
                            )
            case "figure13":
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
                            exps.append(
                                ExpData(
                                    model=model,
                                    system=system,
                                    num_gpus=num_gpus,
                                    batch_size=batch_size,
                                    gpu="A100",
                                )
                            )
            case "figure14":
                for model, num_gpus in [
                    ("retnet-70b", 8),
                    ("gla-70b", 8),
                    ("hgrn2-70b", 8),
                    ("mamba2-70b", 8),
                    ("zamba2-70b", 8),
                    ("opt-70b", 8),
                ]:
                    for system in ["GPU", "GPU+Q", "GPU+PIM", "Pimba"]:
                        exps.append(
                            ExpData(
                                model=model,
                                system=system,
                                num_gpus=num_gpus,
                                batch_size=128,
                                gpu="A100",
                            )
                        )
            case "figure16":
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
                            exps.append(
                                ExpData(
                                    model=model,
                                    system=system,
                                    num_gpus=num_gpus,
                                    batch_size=batch_size,
                                    gpu="H100",
                                )
                            )

    return list(set(exps))
