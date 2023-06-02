from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class Material(Enum):
    scattering = auto()
    tranparent = auto()


@dataclass
class Domain:
    material: Material = Material.scattering
    mu_a: float
    mu_s: float
    g: float
    n: float


@dataclass
class Layer(Domain):
    normal = np.array([0, 0, 1])
    start: float
    end: float


@dataclass
class Sample:
    domains: list[Domain]
