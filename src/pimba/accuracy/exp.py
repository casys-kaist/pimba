import random
from contextlib import contextmanager

import numpy as np
import torch
from lm_eval import simple_evaluate
from rich import print, tree

from . import configs
from .datasets import create_datasets
from .models import create_model


class Accuracy:
    def __init__(
        self,
        model: str,
        quant: str,
        use_sr: bool,
        datasets: list[str] | None,
        seed: int,
    ):
        config = tree.Tree(
            "Config",
            style="bold blue",
        )

        # quant
        self.quant_name = quant
        self.use_sr = use_sr

        quant_config = tree.Tree("Quant")
        quant_config.add(f"mode: {quant}", style="white")
        quant_config.add(f"use_sr: {use_sr}", style="white")
        config.add(quant_config, style="yellow")

        # seed
        self.seed = seed

        seed_config = tree.Tree("Seed")
        seed_config.add(f"seed: {seed}", style="white")
        config.add(seed_config, style="yellow")

        # model
        self.model_name = model
        self.model = create_model(model)

        model_config = tree.Tree("Model")
        model_config.add(f"model: {model}", style="white")
        config.add(model_config, style="yellow")

        # datasets
        self.datasets = create_datasets(datasets)

        datasets_config = tree.Tree("Datasets")
        for dataset in self.datasets:
            datasets_config.add(dataset.name, style="white")
        config.add(datasets_config, style="yellow")

        # config
        self.config = config

    def print_config(self):
        print(self.config)

    def run(self):
        res = simple_evaluate(
            model=self.model,
            tasks=[dataset.name for dataset in self.datasets],
            random_seed=self.seed,
            numpy_random_seed=self.seed,
            torch_random_seed=self.seed,
            fewshot_random_seed=self.seed,
        )
        res: dict[str, float] = {
            dataset.name: res["results"][dataset.name][dataset.output]
            for dataset in self.datasets
        }

        return {
            "exp": {
                "model": self.model_name,
                "quant": self.quant_name,
                "use_sr": self.use_sr,
            },
            "result": res,
        }

    @classmethod
    @contextmanager
    def setup(
        cls,
        model: str,
        quant: str,
        use_sr: bool,
        datasets: list[str] | None = None,
        seed=0,
    ):
        # quant
        original_quant = configs.QUANT
        original_use_sr = configs.USE_SR
        configs.QUANT = quant
        configs.USE_SR = use_sr

        # seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        yield Accuracy(model, quant, use_sr, datasets, seed)

        configs.QUANT = original_quant
        configs.USE_SR = original_use_sr
