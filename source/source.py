import numpy as np
from numba import njit

from sample import Sample
from .angular_distribution import AngularDistribution
from .cfg import SourceCfg
from .dimention import Dimension
from .spatial_distribution import SpatialDistribution


class Source:
    def __init__(self, cfg: SourceCfg, sample: Sample):
        self.cfg = cfg
        self.sample = sample
        self.func_generator = self.get_func_generator()

    def get_func_generator(self):
        layer_index = self.sample.get_func_layer_index()
        spatial_distribution = self.get_func_spatial_distribution()
        angular_distribution = self.get_func_angular_distribution()

        @njit(fastmath=True)
        def generator():
            p = np.zeros(8)

            p[0], p[1], p[2] = spatial_distribution()
            p[3], p[4], p[5] = angular_distribution()

            p[6] = 1
            p[7] = layer_index(p[:2])

            return p

        return generator

    # spatial_distribution block
    def get_func_spatial_distribution(self):
        mode_dimension = self.cfg.dimension
        mode_spatial_distribution = self.cfg.spatial_distribution

        if mode_dimension is Dimension.surface:
            if mode_spatial_distribution is SpatialDistribution.gauss:
                spatial_distribution = self.get_func_surface_gauss()
            elif mode_spatial_distribution is SpatialDistribution.circle:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return spatial_distribution

    def get_func_surface_gauss(self):
        x0, y0, z0 = self.cfg.beam_center
        w = self.cfg.beam_diameter

        @njit(fastmath=True)
        def surface_gauss():
            r0 = np.random.rand()
            r1 = np.random.rand()
            ph = 2 * np.pi * r0
            radius = w * np.sqrt(-np.log(r1))
            return x0 + np.cos(ph) * radius, y0 + np.sin(ph) * radius, z0

        return surface_gauss

    # angular_distribution block
    def get_func_angular_distribution(self):
        mode_dimension = self.cfg.dimension
        mode_angular_distribution = self.cfg.angular_distribution

        if mode_dimension is Dimension.surface:
            if mode_angular_distribution is AngularDistribution.collimated:
                angular_distribution = self.get_func_surface_collimated()
            elif mode_angular_distribution is AngularDistribution.diffuse:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return angular_distribution

    def get_func_surface_collimated(self):

        @njit(fastmath=True)
        def surface_collimated():
            return 0, 0, 1

        return surface_collimated
