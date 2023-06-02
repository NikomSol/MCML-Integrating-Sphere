from dataclasses import dataclass, field

import numpy as np

from .angular_distribution import AngularDistribution
from .dimention import Dimension
from .spatial_distribution import SpatialDistribution


@dataclass
class SourceCfg:
    dimension: Dimension = Dimension.surface
    spatial_distribution: SpatialDistribution = SpatialDistribution.gauss
    angular_distribution: AngularDistribution = AngularDistribution.collimated
    beam_center: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    beam_diameter: float = 1

    def validate(self):
        if not self.dimension:
            raise ValueError(f'dimension = None in SourceCfg')
        if self.dimension not in Dimension:
            raise ValueError(f'dimension = {self.dimension} not available')

        if not self.spatial_distribution:
            raise ValueError(f'spatial_distribution = None in SourceCfg')
        if self.spatial_distribution not in SpatialDistribution:
            raise ValueError(f'spatial_distribution = {self.spatial_distribution} not available')

        if not self.angular_distribution:
            raise ValueError(f'angular_distribution = None in SourceCfg')
        if self.angular_distribution not in AngularDistribution:
            raise ValueError(f'angular_distribution = {self.angular_distribution} not available')
