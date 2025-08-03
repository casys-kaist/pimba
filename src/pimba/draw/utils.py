from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def get_root_path():
    for parent in Path(__file__).parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot find root directory. Did you remove .git file?")


def get_perf_res():
    root = get_root_path()
    res_path = root / "res" / "performance_result.yaml"
    assert res_path.exists()
    return yaml.load(res_path)


def get_acc_res():
    root = get_root_path()
    res_path = root / "res" / "accuracy_result.yaml"
    assert res_path.exists()
    return yaml.load(res_path)


def get_output_path(name: str, ext: str = "pdf"):
    summary_path = get_root_path() / "summary"
    summary_path.mkdir(exist_ok=True, parents=True)
    return summary_path / f"{name}.{ext}"
