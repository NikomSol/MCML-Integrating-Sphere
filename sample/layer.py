from dataclasses import dataclass

from numba import njit

from .material import Material


@dataclass
class Layer:
    material: Material = Material.scattering
    mu_a: float = None
    mu_s: float = None
    g: float = None
    n: float = None
    start: float = None
    end: float = None

    def get_func_is_inside(self):
        start = self.start
        end = self.end

        @njit(fastmath=True)
        def is_inside(z):
            return bool(start < z < end)

        return is_inside
