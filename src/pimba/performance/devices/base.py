from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Generic, Tuple, Type, TypeVar

from ..layers import Layer

DeviceT = TypeVar("DeviceT", bound="Device")


class Handler(ABC, Generic[DeviceT]):
    @classmethod
    def simulate(cls, device: DeviceT, layer) -> Tuple[float, float, float]:
        return cls.time(device, layer), *cls.power(device, layer)

    @classmethod
    @abstractmethod
    def time(cls, device: DeviceT, layer) -> float: ...

    @classmethod
    @abstractmethod
    def power(cls, device: DeviceT, layer) -> Tuple[float, float]: ...


class Device(ABC):
    LAYER_HANDLERS: ClassVar[Dict[Type[Layer], Type[Handler]]] = {}

    def __init_subclass__(cls) -> None:
        cls.LAYER_HANDLERS = {}

    def simulate(self, layer: Layer):
        if type(layer) not in self.LAYER_HANDLERS:
            raise RuntimeError(
                f"Model {type(self)} does not support layer {type(layer)}"
            )

        handler = self.LAYER_HANDLERS[type(layer)]
        return handler.simulate(self, layer)
