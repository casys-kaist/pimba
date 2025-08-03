from dataclasses import dataclass


@dataclass
class DatasetInfo:
    name: str
    output: str


DATASETS = {
    "wikitext": DatasetInfo("wikitext", "word_perplexity,none"),
    "piqa": DatasetInfo("piqa", "acc,none"),
    "lambada": DatasetInfo("lambada_standard", "acc,none"),
    "hellaswag": DatasetInfo("hellaswag", "acc,none"),
    "arc_easy": DatasetInfo("arc_easy", "acc,none"),
    "arc_challenge": DatasetInfo("arc_challenge", "acc,none"),
    "winogrande": DatasetInfo("winogrande", "acc,none"),
}

DEFAULT_DATASETS = [
    "wikitext",
    "piqa",
    "lambada",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
]


def create_datasets(datasets: list[str] | None):
    if datasets is None:
        datasets = DEFAULT_DATASETS

    tasks: list[DatasetInfo] = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise RuntimeError(f"{dataset} is not in the dataset list!")

        tasks.append(DATASETS[dataset])

    return tasks
