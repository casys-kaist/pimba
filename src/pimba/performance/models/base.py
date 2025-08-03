from abc import ABC, abstractmethod

from ..layers import Layer


class Model(ABC):
    @abstractmethod
    def __init__(self):
        self.layers: list[Layer] = []

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)
