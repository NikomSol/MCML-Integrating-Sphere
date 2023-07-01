from dataclasses import dataclass

from numba import njit

from .material import Material


@dataclass
class Layer:
    material: Material = None
    mu_a: float = None
    mu_s: float = None
    g: float = None
    n: float = None
    depth: float = None
