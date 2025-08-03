from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(
        self,
        phase: str,
        category: str,
        name: str,
        m: int,
        n: int,
        k: int,
        num_op: int,
        dbyte: int,
        num_layers: int,
    ):
        self.phase = phase
        self.category = category
        self.name = name
        self.m = m
        self.n = n
        self.k = k
        self.num_op = num_op
        self.dbyte = dbyte
        self.num_layers = num_layers

    def get_infos(self):
        return self.m, self.n, self.k, self.num_op, self.dbyte

    @abstractmethod
    def get_flops(self) -> int: ...


class FC(Layer):
    def get_flops(self) -> int:
        return 2 * self.m * self.n * self.k * self.num_op


class MATMUL(Layer):
    def get_flops(self) -> int:
        return 2 * self.m * self.n * self.k * self.num_op


class TO_PIM(Layer):
    def get_flops(self) -> int:
        return 0


class SOFTMAX(Layer):
    def get_flops(self) -> int:
        return 5 * self.m * self.n * self.num_op


class G2G(Layer):
    def get_flops(self) -> int:
        return 0


class NORM(Layer):
    def get_flops(self) -> int:
        return 5 * self.m * self.n * self.num_op


class ACT(Layer):
    def get_flops(self) -> int:
        return 8 * self.m * self.n * self.num_op


class SU(Layer):
    def get_flops(self) -> int:
        return 0
