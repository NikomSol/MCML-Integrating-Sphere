import numpy as np
from numba import njit

from .layer import Layer


class Sample:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        if layers == []:
            self.boundaries_list = []
            return

        for i in range(len(layers) - 1):
            assert layers[i].end == layers[i + 1].start
        self.boundaries_list = self.calc_boundaries_list()

    def calc_boundaries_list(self):
        layers = self.layers
        boundaries_list = np.array([layer.start for layer in layers])
        return np.append(boundaries_list, layers[-1].end)

    def get_func_layer_index(self):
        boundaries_list = self.boundaries_list

        @njit(fastmath=True)
        def layer_index(z):
            for i, bound in enumerate(boundaries_list):
                if z <= bound:
                    return i - 1
            return -2

        return layer_index
