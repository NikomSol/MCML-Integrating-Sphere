import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import numpy as np
import pytest

from sample import Sample, Layer, Material
from source import AngularDistribution, Dimension, SpatialDistribution, Source, SourceCfg


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
        Layer(material=Material.transparent, depth=1.,
              mu_a=0., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.scattering, depth=1.,
              mu_a=1., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.transparent, depth=1.,
              mu_a=1., mu_s=1., g=0.9, n=1.5)
    ])


@pytest.fixture
def classic_source(classic_cfg, classic_sample):
    return Source(classic_cfg, classic_sample)


def test_classic_source(classic_cfg, classic_source):
    func_generator = classic_source.func_generator

    assert not np.array_equal(func_generator(),
                              func_generator())
    assert np.array_equal(np.zeros((3, 3)), np.zeros((3, 3)))

    assert np.array_equal(func_generator()[1:],
                          np.array([[0, 0, 1], [1, np.NINF, 0]]))

    def norm_func(_):
        return np.linalg.norm(func_generator()[0, :2])

    n = 100
    assert sum(map(norm_func, range(n))) < 2 * n * classic_cfg.beam_diameter
