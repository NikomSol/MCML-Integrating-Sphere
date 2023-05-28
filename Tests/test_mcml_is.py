import numpy as np
import pytest

from MCML.IS import MCML


@pytest.fixture()
def obj():
    return [
        {'z_start': 0, 'z_end': 1, 'mu_a': 0, 'mu_s': 0.01, 'g': 1, 'n': 1},
        {'z_start': 1, 'z_end': 2, 'mu_a': 1.568, 'mu_s': 12.72, 'g': 0.9376, 'n': 1.4},
        {'z_start': 2, 'z_end': 3, 'mu_a': 0, 'mu_s': 0.01, 'g': 1, 'n': 1}
    ]


@pytest.fixture()
def cfg():
    return {
        'N': 1000,  # in one thread
        'threads': 1,  # max cpu_count()-1

        'mode_generator': 'Surface',  # Surface // Volume (todo)
        'mode_spatial_distribution': 'Gauss',  # Gauss // Circle (todo)
        'mode_angular_distribution': 'Collimated',  # Collimated // Diffuse (todo) // HG (todo)

        'Surface_beam_diameter': 1,
        'Surface_beam_center': np.array([0, 0, 0]),
        'Surface_anisotropy_factor': 0.8,

        'mode_save': 'FIS',  # MIS (todo) // FIS (todo)
        'FIS_collimated_cosine': 0.99,
        'MIS_sphere_type': 'Thorlabs_IS200',
        'MIS_positions_table': np.linspace(0, 200, 10)
    }


@pytest.fixture()
def mcml(obj, cfg):
    return MCML(cfg, obj)


@pytest.mark.parametrize(
    'key, value, exc', [
        ('mode_generator', 'Volume', 'Volume key is absent in modes'),
        ('mode_generator', 'KWA', 'KWA key is absent in modes'),
        ('mode_spatial_distribution', 'Circle', 'Circle key is absent in modes[Surface]'),
        ('mode_spatial_distribution', 'KWA', 'KWA key is absent in modes[Surface]'),
        ('mode_angular_distribution', 'Diffuse', 'Diffuse key is absent in modes[Surface]'),
        ('mode_angular_distribution', 'HG', 'HG key is absent in modes[Surface]'),
        ('mode_angular_distribution', 'KWA', 'KWA key is absent in modes[Surface]'),
        ('mode_save', 'MIS', 'MIS key is absent in modes')
    ]
)
def test_modes_errors(obj, cfg, key, value, exc):
    cfg[key] = value
    with pytest.raises(KeyError) as exc_info:
        MCML(cfg, obj)
    assert str(exc_info.value) == f"'{exc}'"


def test_generator(mcml):
    assert np.array_equal(
        mcml.generator()[2:],
        np.array([0, 0, 0, 1, 1, -1])
    )


def test_turn(mcml):
    p0 = np.array([0, 0, 0.5, 0, 0, 1, 1, 0])
    p1 = mcml.get_func_turn()(p0)

    assert np.array_equal(p0, p1)

    p0 = np.array([0, 0, 1.5, 0, 0, 1, 1, 1])
    p1 = mcml.get_func_turn()(p0)

    assert np.array_equal(p0, p1) is False
    assert np.array_equal(p0[:3], p1[:3])
    assert np.array_equal(p0[-2:], p1[-2:])
    assert p0[3] ** 2 + p0[4] ** 2 + p0[5] ** 2 == 1


def test_move(mcml):
    p0 = np.array([0, 0, 0.5, 0, 0, 1, 10 ** -5, 1])
    p1 = mcml.get_func_term()(p0)
    p2 = np.array([0, 0, 0.5, 0, 0, 1, 0, 1])
    p3 = np.array([0, 0, 0.5, 0, 0, 1, 10 ** -4, 1])

    assert np.array_equal(p3, p1) or np.array_equal(p2, p1)
