import numpy as np

from .LigntSource import Dimension, SpatialDistribution, AxialDistribution


class GeneratorCfg:
    mode_source_dimension: Dimension = Dimension.surface
    mode_spatial_distribution: SpatialDistribution = SpatialDistribution.gauss
    mode_angular_distribution: AxialDistribution = AxialDistribution.collimated
    beam_center: np.ndarray = np.array([0, 0, 0])
    beam_diameter: float = 1
