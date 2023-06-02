# TODO Move mods to separate file (import conflict)

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class Dimension(Enum):
    point = 1
    surface = 2
    volume = 3


class SpatialDistribution(Enum):
    gauss = auto()
    cyrcle = auto()


class AngularDistribution(Enum):
    collimated = auto()
    diffuse = auto()


@dataclass
class SourceCfg:
    dimension: Dimension = Dimension.surface
    spatial_distribution: SpatialDistribution = SpatialDistribution.gauss
    angular_distribution:  AngularDistribution = AngularDistribution.collimated
    beam_center: np.ndarray = np.array([0, 0, 0])
    beam_diameter: float = 1

    def validate(self):
        if self.dimension not in Dimension:
            raise ValueError('dimension = {dimension} not available')
        if self.spatial_distribution not in SpatialDistribution:
            raise ValueError('spatial_distribution = {spatial_distribution} not available')
        if self.angular_distribution not in AngularDistribution:
            raise ValueError('angular_distribution = {angular_distribution} not available')
