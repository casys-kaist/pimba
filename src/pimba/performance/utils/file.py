from pathlib import Path


def get_root_path():
    for parent in Path(__file__).parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot find root directory. Did you remove .git file?")


def get_simulator_bin_path():
    root = get_root_path()
    bin_path = root / "build" / "PimbaSim"
    assert bin_path.exists()
    return bin_path
