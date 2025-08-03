from typing import Type

from ..devices import Device
from ..layers import Layer


def register(device: Type[Device], layer: Type[Layer]):
    def decorator(cls):
        device.LAYER_HANDLERS[layer] = cls
        return cls

    return decorator
