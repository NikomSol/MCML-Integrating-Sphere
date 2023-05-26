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
        'N': 100000,
        'threads': 1,

        'mode_generator': 'Surface',
        'Surface_spatial_distribution': 'Gauss',
        'Surface_beam_diameter': 1,
        'Surface_beam_center': np.array([0, 0, 0]),
        'Surface_angular_distribution': 'Collimated',

        'mode_save': 'FIS',
        'FIS_collimated_cosinus': 0.99,
    }


@pytest.fixture()
def mcml(obj, cfg):
    return MCML(cfg, obj)


@pytest.mark.parametrize(
    'key, value, exc', [
        ('Surface_spatial_distribution', 'Circle', 'todo get_func_circle_distribution'),
        ('Surface_spatial_distribution', 'KWA', 'Unknown Surface_spatial_distribution'),
        ('Surface_angular_distribution', 'Diffuse', 'todo Surface_angular_distribution == Diffuse'),
        ('Surface_angular_distribution', 'HG', 'todo Surface_angular_distribution == HG'),
        ('Surface_angular_distribution', 'KWA', 'Unknown Surface_angular_distribution'),
        ('mode_generator', 'Volume', 'todo mode_generator == Volume'),
        ('mode_generator', 'KWA', 'Unknown mode_generator'),
        ('mode_save', 'MIS', 'todo MIS')
    ]
)
def test_modes_errors(obj, cfg, key, value, exc):
    cfg[key] = value
    with pytest.raises(ValueError) as exc_info:
        MCML(cfg, obj)
    assert str(exc_info.value) == exc


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
    assert 1. == p0[3] ** 2 + p0[4] ** 2 + p0[5] ** 2


def test_move(mcml):
    p0 = np.array([0, 0, 0.5, 0, 0, 1, 10 ** -5, 1])
    p1 = mcml.get_func_term()(p0)
    p2 = np.array([0, 0, 0.5, 0, 0, 1, 0, 1])
    p3 = np.array([0, 0, 0.5, 0, 0, 1, 10 ** -4, 1])

    assert np.array_equal(p3, p1) or np.array_equal(p2, p1)
