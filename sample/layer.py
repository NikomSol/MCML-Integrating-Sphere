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
