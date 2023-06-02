from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from numba import njit


class Material(Enum):
    scattering = auto()
    tranparent = auto()


@dataclass
class Layer():
    material: Material = Material.scattering
    mu_a: float = None
    mu_s: float = None
    g: float = None
    n: float = None
    start: float = None
    end: float = None

    def get_func_is_insite(self):
        start = self.start
        end = self.end

        @njit(fastmath=True)
        def is_insite(z):
            if start < z < end:
                return True
            else:
                return False
        return is_insite


class Sample:
    def __init__(self, layers: list[Layer]):
        for i in range(len(layers)-1):
            assert layers[i].end == layers[i+1].start
        self.layers = layers
        self.boundaries_list = self.calc_boundaries_list()

    def calc_boundaries_list(self):
        layers = self.layers
        boundaries_list = np.array([layer.start for layer in layers])
        return np.append(boundaries_list, layers[-1].end)

    def get_func_layer_index(self):
        boundaries_list = self.boundaries_list

        @njit(fastmath=True)
        def layer_index(z):
            assert isinstance(z, float)
            for i, bound in enumerate(boundaries_list):
                if z < bound:
                    return i - 1
            return -2
        return layer_index
