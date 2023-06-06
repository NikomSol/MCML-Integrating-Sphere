import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
import numpy as np

from sample import Sample, Layer, Material

from source import AngularDistribution, Dimension, SpatialDistribution
from source import Source, SourceCfg

from detector import DetectorCfg, Measurement, Probe
from detector import Detector

from direct_problem import DirectProblemCfg
from direct_problem import DirectProblem

@pytest.fixture
def classic_sample():
    return Sample([
        Layer(material=Material.transparent,
              start=0., end=1.,
              mu_a=0.1, mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.scattering,
              start=1., end=2.,
              mu_a=1., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.transparent,
              start=2., end=3.,
              mu_a=1., mu_s=1., g=0.9, n=1.5)
        ])


@pytest.fixture
def classic_source_cfg():
    return SourceCfg(
        dimension=Dimension.surface,
        spatial_distribution=SpatialDistribution.gauss,
        angular_distribution=AngularDistribution.collimated,
        beam_center=np.array([0, 0, 0]),
        beam_diameter=float(1)
    )


@pytest.fixture
def classic_source(classic_source_cfg, classic_sample):
    return Source(classic_source_cfg, classic_sample)


@pytest.fixture
def cfg_ALL():
    return DetectorCfg(
        measurement=Measurement.ALL,
    )


@pytest.fixture
def detector_ALL(cfg_ALL):
    return Detector(cfg_ALL)


@pytest.fixture
def direct_problem_cfg():
    return DirectProblemCfg(
        N=10,
        threads=1
    )


@pytest.fixture
def direct_problem(direct_problem_cfg, classic_sample, classic_source, detector_ALL):
    return DirectProblem(direct_problem_cfg,
                         classic_sample, classic_source, detector_ALL)


def test_direct_problem_get_funcs(direct_problem):
    p1 = np.array([[0, 0, 1.5],
                  [0, 0.6, 0.8],
                  [10 ** -5, 1, 0]])

    p2 = p1 * 1.
    p2[2, 0] = 0

    p5 = p1 * 1.
    p5[2, 0] = 10 * p5[2, 0]

    term = direct_problem.get_func_term()
    p_term = term(p1)
    assert np.array_equal(p_term, p2) or np.array_equal(p_term, p5)

    turn = direct_problem.get_func_turn()
    p_turn = turn(p1)
    assert np.array_equal(p_turn[0], p1[0])
    assert np.array_equal(p_turn[2], p1[2])
    assert not np.array_equal(p_turn[1], p1[1])

    reflection = direct_problem.get_func_reflection()
    p3 = p1 * 1.
    p3[1, 2] = -p1[1, 2]
    p4 = p1 * 1.
    p4[2, 1] = p1[2, 1] + 1
    p_refl = reflection(p1)
    assert np.array_equal(p_refl, p3) or np.array_equal(p_refl, p4)

    move = direct_problem.get_func_move()
    p_move = move(p1)
    assert not np.array_equal(p_move[0], p1[0])
    assert np.array_equal(p_move[1, :-1], p1[1, :-1])
    assert (p_move[1, 2] == p1[1, 2]) or (p_move[1, 2] == -p1[1, 2])
    assert p_move[2, 0] != p1[2, 0]


def test_direct_problem_get_trace(direct_problem):
    trace = direct_problem.get_func_trace()
    storage = trace()
    assert np.sum(storage) == 1
