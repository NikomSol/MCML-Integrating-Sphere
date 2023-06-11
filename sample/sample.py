import numpy as np
from numba import njit

from .layer import Layer


class Sample:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        if not layers:
            self.boundaries_list = []
            self.g_table = []
            self.mu_a_table = []
            self.mu_s_table = []
            self.n_table = []
            return

        for i in range(len(layers) - 1):
            assert layers[i].end == layers[i + 1].start

        self.calc_boundaries_list()
        self.calc_property_tables()

    def calc_boundaries_list(self):
        layers = self.layers
        boundaries_list = np.array([layer.start for layer in layers])
        self.boundaries_list = np.append(boundaries_list, layers[-1].end)

    def get_func_layer_index(self):
        boundaries_list = self.boundaries_list

        @njit(fastmath=True)
        def layer_index(z):
            if z <= boundaries_list[0]:
                return np.NINF
            for i, bound in enumerate(boundaries_list[1:]):
                if z <= bound:
                    return i
            return np.PINF

        return layer_index

    def calc_property_tables(self):
        layers = self.layers
        n = len(layers)
        g_table = np.zeros(n)
        mu_a_table = np.zeros(n)
        mu_s_table = np.zeros(n)
        n_table = np.zeros(n)

        for i, layer in enumerate(layers):
            g_table[i] = layer.g
            mu_a_table[i] = layer.mu_a
            mu_s_table[i] = layer.mu_s
            n_table[i] = layer.n

        self.g_table = g_table
        self.mu_a_table = mu_a_table
        self.mu_s_table = mu_s_table
        self.n_table = n_table
