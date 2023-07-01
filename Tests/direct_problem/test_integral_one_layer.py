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


# Control Collimated, Diffuse and Absorbtion radiation
def reference_parameters(l_mu_t, r):
    e = np.exp(-l_mu_t)
    trans = (1-r)**2 * e / (1 - (r * e)**2)
    ref = r + r * ((1 - r) * e)**2 / (1 - (r * e)**2)
    losses = (1 - r) * (1 - e) / (1 - r * e)
    assert np.abs(trans + ref + losses - 1) < 10**(-12)
    return trans, ref, losses


def layers_reference_parameters(layer):
    mu_a = layer.mu_a
    mu_s = layer.mu_s
    z = layer.depth
    n = layer.n
    return reference_parameters(z * (mu_a + mu_s), ((n - 1) / (n + 1))**2)


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
def layers():
    return [
        Layer(material=Material.scattering, depth=1., n=1.5, mu_a=0.1, mu_s=0.9, g=0.9),
        Layer(material=Material.scattering, depth=1., n=1.3, mu_a=0.02, mu_s=0.08, g=0.5),
        Layer(material=Material.scattering, depth=10., n=1.6, mu_a=0.7, mu_s=0.3, g=0.0)
        ]


@pytest.fixture
def samples(layers):
    return [Sample([layer]) for layer in layers]


@pytest.fixture
def direct_problems(cfg, samples):
    detector = cfg.detector.get_detector()
    return [DirectProblem(cfg.direct_problem, sample, Source(cfg.source, sample), detector) for sample in samples]


def test_direct_problem_0(direct_problems):
    direct_problem = direct_problems[0]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    collimated_transmition = solution[0][0]
    diffuse_transmition = solution[0][1]
    diffuse_reflection = solution[0][2]
    collimated_reflection = solution[0][3]

    trans, ref, losses = layers_reference_parameters(direct_problem.sample.layers[0])

    reference_collimated_transmition = trans * N
    reference_diffuse_transmition = (0.767 - trans) * N
    reference_diffuse_reflection = (0.0435 + 0.04 - ref) * N
    reference_collimated_reflection = ref * N

    # raise ValueError(solution, [reference_collimated_transmition, reference_diffuse_transmition,
    #                             reference_diffuse_reflection, reference_collimated_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(collimated_transmition - reference_collimated_transmition) < error_coefficient * np.sqrt(collimated_transmition)
    assert np.abs(diffuse_transmition - reference_diffuse_transmition) < error_coefficient * np.sqrt(diffuse_transmition)
    assert np.abs(diffuse_reflection - reference_diffuse_reflection) < error_coefficient * np.sqrt(diffuse_reflection)
    assert np.abs(collimated_reflection - reference_collimated_reflection) < error_coefficient * np.sqrt(collimated_reflection)


def test_direct_problem_1(direct_problems):
    direct_problem = direct_problems[1]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    collimated_transmition = solution[0][0]
    diffuse_transmition = solution[0][1]
    diffuse_reflection = solution[0][2]
    collimated_reflection = solution[0][3]

    trans, ref, losses = layers_reference_parameters(direct_problem.sample.layers[0])

    reference_collimated_transmition = trans * N
    reference_diffuse_transmition = (0.9206 - trans) * N
    reference_diffuse_reflection = (0.01701 + 0.0272 - ref) * N
    reference_collimated_reflection = ref * N

    # raise ValueError(solution, [reference_collimated_transmition, reference_diffuse_transmition,
    #                             reference_diffuse_reflection, reference_collimated_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(collimated_transmition - reference_collimated_transmition) < error_coefficient * np.sqrt(collimated_transmition)
    assert np.abs(diffuse_transmition - reference_diffuse_transmition) < error_coefficient * np.sqrt(diffuse_transmition)
    assert np.abs(diffuse_reflection - reference_diffuse_reflection) < error_coefficient * np.sqrt(diffuse_reflection)
    assert np.abs(collimated_reflection - reference_collimated_reflection) < error_coefficient * np.sqrt(collimated_reflection)


def test_direct_problem_2(direct_problems):
    direct_problem = direct_problems[2]
    N = direct_problem.cfg.N

    error_coefficient = 3

    solution = direct_problem.solve()

    collimated_transmition = solution[0][0]
    diffuse_transmition = solution[0][1]
    diffuse_reflection = solution[0][2]
    collimated_reflection = solution[0][3]

    trans, ref, losses = layers_reference_parameters(direct_problem.sample.layers[0])

    reference_collimated_transmition = trans * N
    reference_diffuse_transmition = (0.0000549 - trans) * N
    reference_diffuse_reflection = (0.05325 + 0.0175 - ref) * N
    reference_collimated_reflection = ref * N

    # raise ValueError(solution, [reference_collimated_transmition, reference_diffuse_transmition,
    #                             reference_diffuse_reflection, reference_collimated_reflection])

    assert np.abs(np.sum(solution[0]) + solution[1] - N) < error_coefficient * np.sqrt(N)

    assert np.abs(collimated_transmition - reference_collimated_transmition) < error_coefficient * np.sqrt(collimated_transmition)
    assert np.abs(diffuse_transmition - reference_diffuse_transmition) < error_coefficient * np.sqrt(diffuse_transmition)
    assert np.abs(diffuse_reflection - reference_diffuse_reflection) < error_coefficient * np.sqrt(diffuse_reflection)
    assert np.abs(collimated_reflection - reference_collimated_reflection) < error_coefficient * np.sqrt(collimated_reflection)
