import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import numpy as np
import pytest

from source import SourceCfg, AngularDistribution, Dimension, SpatialDistribution


@pytest.fixture
def empty_cfg():
    return SourceCfg(
        dimension=None,
        spatial_distribution=None,
        angular_distribution=None,
        beam_center=None,
        beam_diameter=None,
    )


def test_empty_cfg(empty_cfg):
    with pytest.raises(ValueError):
        empty_cfg.validate()


@pytest.fixture
def cfg():
    return SourceCfg(
        dimension=Dimension.surface,
        spatial_distribution=SpatialDistribution.gauss,
        angular_distribution=AngularDistribution.collimated,
        beam_center=np.array([0, 0, 0]),
        beam_diameter=float(1)
    )


def test_cfg_is_ok(cfg):
    cfg.validate()


def test_cfg_all_dimension(cfg):
    for variant in Dimension:
        cfg.dimension = variant
        cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'dimension = None in SourceCfg'),
        (dict(), 'dimension = None in SourceCfg'),
    ]
)
def test_cfg_with_wrong_dimension(cfg, value, error):
    cfg.dimension = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


def test_cfg_all_spatial_distribution(cfg):
    for variant in SpatialDistribution:
        cfg.spatial_distribution = variant
        cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'spatial_distribution = None in SourceCfg'),
        (dict(), 'spatial_distribution = None in SourceCfg'),
    ]
)
def test_cfg_with_wrong_spatial_distribution(cfg, value, error):
    cfg.spatial_distribution = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


def test_cfg_all_angular_distribution(cfg):
    for variant in AngularDistribution:
        cfg.angular_distribution = variant
        cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'angular_distribution = None in SourceCfg'),
        (dict(), 'angular_distribution = None in SourceCfg'),
    ]
)
def test_cfg_with_wrong_angular_distribution(cfg, value, error):
    cfg.angular_distribution = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


@pytest.mark.parametrize(
    'value, error', [
        (None, 'beam_center = None in SourceCfg'),
        (dict(), 'beam_center is not of type np.ndarray in SourceCfg'),
    ]
)
def test_cfg_with_wrong_beam_center(cfg, value, error):
    cfg.beam_center = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


@pytest.mark.parametrize(
    'value, error', [
        (None, 'beam_diameter = None in SourceCfg'),
        (dict(), 'beam_diameter = None in SourceCfg',),
        (float(-1), 'beam_diameter = -1.0 out of range [0, +inf) in SourceCfg')
    ]
)
def test_cfg_with_wrong_beam_diameter(cfg, value, error):
    cfg.beam_diameter = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error
