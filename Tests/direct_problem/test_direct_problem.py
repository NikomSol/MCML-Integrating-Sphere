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
              mu_a=0.1, mu_s=1., g=0.9, n=1.4),
        Layer(material=Material.scattering,
              start=1., end=2.,
              mu_a=1., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.transparent,
              start=2., end=3.,
              mu_a=1., mu_s=1., g=0.9, n=1.4)
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

@pytest.fixture
def base_p():
    return np.array([[0, 0, 1.5],
                     [0, 0.6, 0.8],
                     [10 ** -5, 0, 0]])


def test_direct_problem_get_func_term(direct_problem, base_p):

    term = direct_problem.get_func_term()

    p2 = base_p * 1.
    p2[2, 0] = 0

    p3 = base_p * 1.
    p3[2, 0] = 10 * p3[2, 0]

    p_term = term(base_p)
    assert np.array_equal(p_term, p2) or np.array_equal(p_term, p3)


def test_direct_problem_get_func_turn(direct_problem, base_p):

    turn = direct_problem.get_func_turn()

    p_turn = turn(base_p)
    assert np.array_equal(p_turn[0], base_p[0])
    assert np.array_equal(p_turn[2], base_p[2])
    assert not np.array_equal(p_turn[1], base_p[1])


def test_direct_problem_get_func_reflection(direct_problem, base_p):

    reflection = direct_problem.get_func_reflection()

    # layer - layer
    p2 = base_p * 1.
    p2[1, 2] = - p2[1, 2]
    p3 = base_p * 1.
    p3[2, 1] = p3[2, 1] + 1

    p_refl = reflection(base_p)
    assert np.array_equal(p_refl, p2) or (np.array_equal(p_refl[0], p3[0]) and
                                          np.array_equal(p_refl[2], p3[2]) and
                                          not np.array_equal(p_refl[1], p3[1]))
    assert np.abs(np.linalg.norm(p_refl[1]) - 1.) < 10 ** (-5)

    # down -> layer
    start_p = np.array([[0, 0, 0],
                        [0, 0.6, 0.8],
                        [10 ** -5, np.NINF, 0]])
    p2 = start_p * 1.
    p2[1, 2] = - p2[1, 2]
    p3 = start_p * 1.
    p3[2, 1] = 0

    p_refl = reflection(start_p)
    assert np.array_equal(p_refl, p2) or (np.array_equal(p_refl[0], p3[0]) and
                                          np.array_equal(p_refl[2], p3[2]) and
                                          not np.array_equal(p_refl[1], p3[1]))
    assert np.abs(np.linalg.norm(p_refl[1]) - 1.) < 10 ** (-5)

    # layer -> down
    start_p = np.array([[0, 0, 0],
                        [0, 0.6, -0.8],
                        [10 ** -5, 0, 0.5]])
    p2 = start_p * 1.
    p2[1, 2] = - p2[1, 2]
    p3 = start_p * 1.
    p3[2, 1] = np.NINF

    p_refl = reflection(start_p)
    assert np.array_equal(p_refl, p2) or (np.array_equal(p_refl[0], p3[0]) and
                                          np.array_equal(p_refl[2], p3[2]) and
                                          not np.array_equal(p_refl[1], p3[1]))
    assert np.abs(np.linalg.norm(p_refl[1]) - 1.) < 10 ** (-5)

    # top -> layer
    start_p = np.array([[0, 0, 0],
                        [0, 0.6, -0.8],
                        [10 ** -5, np.PINF, 4]])
    p2 = start_p * 1.
    p2[1, 2] = - p2[1, 2]
    p3 = start_p * 1.
    p3[2, 1] = len(direct_problem.sample.layers) - 1

    p_refl = reflection(start_p)
    assert np.array_equal(p_refl, p2) or (np.array_equal(p_refl[0], p3[0]) and
                                          np.array_equal(p_refl[2], p3[2]) and
                                          not np.array_equal(p_refl[1], p3[1]))
    assert np.abs(np.linalg.norm(p_refl[1]) - 1.) < 10 ** (-5)

    # layer -> top
    start_p = np.array([[0, 0, 0],
                        [0, 0.6, 0.8],
                        [10 ** -5, len(direct_problem.sample.layers) - 1, 2.5]])
    p2 = start_p * 1.
    p2[1, 2] = - p2[1, 2]
    p3 = start_p * 1.
    p3[2, 1] = np.PINF

    p_refl = reflection(start_p)
    assert np.array_equal(p_refl, p2) or (np.array_equal(p_refl[0], p3[0]) and
                                          np.array_equal(p_refl[2], p3[2]) and
                                          not np.array_equal(p_refl[1], p3[1]))
    assert np.abs(np.linalg.norm(p_refl[1]) - 1.) < 10 ** (-5)


def test_direct_problem_get_func_move(direct_problem, base_p):

    move = direct_problem.get_func_move()

    p_move = move(base_p)
    assert p_move[2, 0] != base_p[2, 0]
    assert np.abs(np.linalg.norm(p_move[1]) - 1.) < 10 ** (-5)
    assert np.equal(p_move[2, 1], direct_problem.sample.get_func_layer_index()(p_move[0, 2]))


def test_direct_problem_get_trace(direct_problem):
    trace = direct_problem.get_func_trace()
    storage = trace()
    assert np.sum(storage) == 1
