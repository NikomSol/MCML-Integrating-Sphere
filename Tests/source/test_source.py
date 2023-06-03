import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
import numpy as np

from sample import Sample, Layer, Material
from source import AngularDistribution, Dimension, SpatialDistribution
from source import Source, SourceCfg


@pytest.fixture
def classic_cfg():
    return SourceCfg(
        dimension=Dimension.surface,
        spatial_distribution=SpatialDistribution.gauss,
        angular_distribution=AngularDistribution.collimated,
        beam_center=np.array([0, 0, 0]),
        beam_diameter=float(1)
    )


@pytest.fixture
def classic_sample():
    return Sample([
        Layer(material=Material.transparent,
              start=0., end=1.,
              mu_a=0., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.scattering,
              start=1., end=2.,
              mu_a=1., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.transparent,
              start=2., end=3.,
              mu_a=1., mu_s=1., g=0.9, n=1.5)
        ])


@pytest.fixture
def classic_source(classic_cfg, classic_sample):
    return Source(classic_cfg, classic_sample)


def test_classic_source(classic_cfg, classic_source):
    func_generator = classic_source.func_generator

    assert not np.array_equal(func_generator(),
                              func_generator())

    assert np.array_equal(func_generator()[2:],
                          np.array([0, 0, 0, 1, 1, -1, 0]))

    def norm_func(_):
        return np.linalg.norm(func_generator()[:2])

    N = 100
    assert sum(map(norm_func, range(N))) < 2 * N * classic_cfg.beam_diameter