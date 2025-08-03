# NOTE: triton autotune monkey patch
# Due to triton's autotune function intorudces non-deterministic behavior,
# it interupts reproducibility. So we disable it.
import triton


def _dummy_tuner(**kwargs):
    def decorator(fn):
        return fn

    return decorator


triton.autotune = _dummy_tuner

# disable logging
import logging

logging.disable()

from .accuracy.exp import Accuracy as Accuracy
from .accuracy.exps.utils import create_accuracy_exps as create_accuracy_exps
from .draw.exps import draw as draw
from .performance.exp import Performance as Performance
from .performance.exps import create_performance_exps as create_performance_exps
from .performance.utils.file import get_root_path as get_root_path
