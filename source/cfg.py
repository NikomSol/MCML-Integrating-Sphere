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
    beam_diameter: float = 1.0

    def validate(self):
        if not self.dimension:
            raise ValueError('dimension = None in SourceCfg')
        if not isinstance(self.dimension, Dimension):
            raise ValueError('dimension is not of type Dimension in SourceCfg')
        if self.dimension not in Dimension:
            raise ValueError(f'dimension = {self.dimension} not available in SourceCfg')

        if not self.spatial_distribution:
            raise ValueError('spatial_distribution = None in SourceCfg')
        if not isinstance(self.spatial_distribution, SpatialDistribution):
            raise ValueError('spatial_distribution is not of type SpatialDistribution in SourceCfg')
        if self.spatial_distribution not in SpatialDistribution:
            raise ValueError(f'spatial_distribution = {self.spatial_distribution} not available in SourceCfg')

        if not self.angular_distribution:
            raise ValueError('angular_distribution = None in SourceCfg')
        if not isinstance(self.angular_distribution, AngularDistribution):
            raise ValueError('angular_distribution is not of type AngularDistribution in SourceCfg')
        if self.angular_distribution not in AngularDistribution:
            raise ValueError(f'angular_distribution = {self.angular_distribution} not available in SourceCfg')

        if self.beam_center is None:
            raise ValueError('beam_center = None in SourceCfg')
        if not isinstance(self.beam_center, np.ndarray):
            raise ValueError('beam_center is not of type np.ndarray in SourceCfg')

        if not self.beam_diameter:
            raise ValueError('beam_diameter = None in SourceCfg')
        if not isinstance(self.beam_diameter, float):
            raise ValueError('beam_diameter is not of type float in SourceCfg')
        if self.beam_diameter < 0:
            raise ValueError(f'beam_diameter = {self.beam_diameter} out of range [0, +inf) in SourceCfg')
