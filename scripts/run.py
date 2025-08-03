from multiprocessing import Pool
from pathlib import Path

from ruamel.yaml import YAML
from tqdm import tqdm

from pimba import (
    Accuracy,
    Performance,
    create_accuracy_exps,
    create_performance_exps,
    get_root_path,
)

EXPS = [
    "table2",
    "figure3",
    "figure4",
    "figure5",
    "figure6",
    "figure12",
    "figure13",
    "figure14",
    "figure16",
]


def perf_exp(exp_data):
    with Performance.setup(
        model=exp_data.model,
        system=exp_data.system,
        num_gpus=exp_data.num_gpus,
        batch_size=exp_data.batch_size,
        gpu=exp_data.gpu,
    ) as exp:
        return exp.run()


def acc_exp(exp_data):
    with Accuracy.setup(
        model=exp_data.model,
        quant=exp_data.quant,
        use_sr=exp_data.use_sr,
        datasets=exp_data.datasets,
    ) as exp:
        return exp.run()


if __name__ == "__main__":
    # run accuracy experiments
    acc_res = [acc_exp(exp_data) for exp_data in create_accuracy_exps(EXPS)]

    # run performance experiments
    exp_list = create_performance_exps(EXPS)
    with Pool(maxtasksperchild=1) as pool:
        perf_res = [
            result
            for result in tqdm(
                pool.imap_unordered(perf_exp, exp_list),
                desc="performance experiments",
                total=len(exp_list),
            )
        ]

    # dump results
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False

    result_path = get_root_path() / "res"
    result_path.mkdir(exist_ok=True, parents=True)

    yaml.dump(acc_res, result_path / "accuracy_result.yaml")
    yaml.dump(perf_res, result_path / "performance_result.yaml")

    print("All experiments done!")
