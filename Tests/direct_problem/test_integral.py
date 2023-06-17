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
# Absorbtion layer - a, scattering - s

@pytest.fixture
def sample_blb_a():
    return Sample([Layer(material=Material.transparent,
                         start=0., end=1.,
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
    error = 10**(-5)

    n = direct_problem_blb_a.cfg.N
    solution = direct_problem_blb_a.solve()

    assert np.abs(n - np.sum(solution[0]) + solution[1]) < error
