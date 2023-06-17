import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import numpy as np
import pytest

from detector import DetectorCfg, Measurement
from direct_problem import DirectProblem, DirectProblemCfg
from sample import Sample, Layer, Material
from source import Source, AngularDistribution, Dimension, SpatialDistribution, SourceCfg


# Generate gauss collimated beam
# Control Collimated, Diffuse and Absorbtion radiation
def collamated_parameters(l_mu_t, r):
    e = np.exp(-l_mu_t)
    trans = (1-r)**2 * e / (1 - (r * e)**2)
    ref = r + r * ((1 - r) * e)**2 / (1 - (r * e)**2)
    losses = (1 - r) * (1 - e) / (1 - r * e)
    assert trans + ref + losses == 1
    return trans, ref, losses


@pytest.fixture
def source_cfg():
    return SourceCfg(
        dimension=Dimension.surface,
        spatial_distribution=SpatialDistribution.gauss,
        angular_distribution=AngularDistribution.collimated,
        beam_center=np.array([0, 0, 0]),
        beam_diameter=float(1)
    )


@pytest.fixture
def detector_cfg():
    return DetectorCfg(
        measurement=Measurement.CollimatedDiffuse,
    )


@pytest.fixture
def detector(detector_cfg):
    return detector_cfg.get_detector()


@pytest.fixture
def direct_problem_cfg():
    return DirectProblemCfg(
        N=10000,
        threads=1
    )


# Bouguer-Lambert-Beer law (blb)

# Absorbtion layer - a

@pytest.fixture
def sample_blb_a():
    return Sample([Layer(material=Material.transparent,
                         start=0., end=1., n=1.5,
                         mu_a=1., mu_s=0., g=1)])


@pytest.fixture
def source_blb_a(source_cfg, sample_blb_a):
    return Source(source_cfg, sample_blb_a)


@pytest.fixture
def direct_problem_blb_a(direct_problem_cfg, sample_blb_a,
                         source_blb_a, detector):
    return DirectProblem(direct_problem_cfg, sample_blb_a,
                         source_blb_a, detector)


def test_direct_problem_blb_a(direct_problem_blb_a):
    N = direct_problem_blb_a.cfg.N
    relative_error = 10**(-2)
    absolute_error = relative_error * N

    solution = direct_problem_blb_a.solve()

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < absolute_error

    layer = direct_problem_blb_a.sample.layers[0]
    mu_a = layer.mu_a
    mu_s = layer.mu_s
    z = layer.end - layer.start
    n = layer.n

    trans, refl, losses = collamated_parameters(z * (mu_a + mu_s), ((n - 1) / (n + 1))**2)
    assert (solution[1] - losses * mu_a / (mu_a + mu_s) * N) / solution[1] < relative_error
    assert (solution[0][0] - trans * N) / solution[0][0] < relative_error
    assert (solution[0][1] - 0) < absolute_error
    assert (solution[0][2] - 0) < absolute_error
    assert (solution[0][3] - refl * N) / solution[0][3] < relative_error
