import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import numpy as np
import pytest

from cfg import Cfg
from detector import DetectorCfg, Measurement
from direct_problem import DirectProblem, DirectProblemCfg
from sample import Sample, Layer, Material
from source import Source, AngularDistribution, Dimension, SpatialDistribution, SourceCfg


@pytest.fixture
def cfg():
    source_cfg = SourceCfg(dimension=Dimension.surface,
                           spatial_distribution=SpatialDistribution.gauss,
                           angular_distribution=AngularDistribution.collimated,
                           beam_center=np.array([0, 0, 0]),
                           beam_diameter=float(1))

    detector_cfg = DetectorCfg(measurement=Measurement.CollimatedDiffuse)

    direct_problem_cfg = DirectProblemCfg(N=10000, threads=1)

    cfg = Cfg(
        source=source_cfg,
        detector=detector_cfg,
        direct_problem=direct_problem_cfg
        )

    return cfg


@pytest.fixture
def layers_tissue():
    return [
        Layer(material=Material.scattering, depth=1., n=1.5, mu_a=0.1, mu_s=0.9, g=0.9),
        Layer(material=Material.scattering, depth=1., n=1.3, mu_a=0.02, mu_s=0.08, g=0.5),
        Layer(material=Material.scattering, depth=10., n=1.6, mu_a=0.7, mu_s=0.3, g=0.0)
        ]


@pytest.fixture
def layers_glass():
    return [
        Layer(material=Material.transparent, depth=1., n=1.6, mu_a=0., mu_s=0., g=0.),
        Layer(material=Material.transparent, depth=1., n=1.4, mu_a=0., mu_s=0., g=0.),
        Layer(material=Material.transparent, depth=1., n=1.65, mu_a=0., mu_s=0., g=0.)
        ]


@pytest.fixture
def samples(layers_tissue, layers_glass):
    return [Sample([layers_glass[i], layers_tissue[i], layers_glass[i]]) for i in range(len(layers_tissue))]


@pytest.fixture
def direct_problems(cfg, samples):
    detector = cfg.detector.get_detector()
    return [DirectProblem(cfg.direct_problem, sample, Source(cfg.source, sample), detector) for sample in samples]


def test_direct_problem_0(direct_problems):
    direct_problem = direct_problems[0]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    total_transmition = solution[0][0] + solution[0][1]
    total_reflection = solution[0][2] + solution[0][3]

    reference_total_transmition = (0.7394) * N
    reference_total_reflection = (0.05419 + 0.05813) * N

    # raise ValueError(solution, [reference_total_transmition, reference_total_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(total_transmition - reference_total_transmition) < error_coefficient * np.sqrt(total_transmition)
    assert np.abs(total_reflection - reference_total_reflection) < error_coefficient * np.sqrt(total_reflection)


def test_direct_problem_1(direct_problems):
    direct_problem = direct_problems[1]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    total_transmition = solution[0][0] + solution[0][1]
    total_reflection = solution[0][2] + solution[0][3]

    reference_total_transmition = (0.8987) * N
    reference_total_reflection = (0.02907 + 0.03695) * N

    # raise ValueError(solution, [reference_total_transmition, reference_total_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(total_transmition - reference_total_transmition) < error_coefficient * np.sqrt(total_transmition)
    assert np.abs(total_reflection - reference_total_reflection) < error_coefficient * np.sqrt(total_reflection)


def test_direct_problem_2(direct_problems):
    direct_problem = direct_problems[2]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    total_transmition = solution[0][0] + solution[0][1]
    total_reflection = solution[0][2] + solution[0][3]

    reference_total_transmition = (0.0000541) * N
    reference_total_reflection = (0.06037 + 0.01718) * N

    # raise ValueError(solution, [reference_total_transmition, reference_total_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(total_transmition - reference_total_transmition) < error_coefficient * np.sqrt(total_transmition)
    assert np.abs(total_reflection - reference_total_reflection) < error_coefficient * np.sqrt(total_reflection)
