import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest

from detector import DetectorCfg, Measurement, Probe


@pytest.fixture
def empty_cfg():
    return DetectorCfg(
        measurement=None,
        probe=None,
        collimated_cosine=None,
    )


def test_empty_cfg(empty_cfg):
    with pytest.raises(ValueError):
        empty_cfg.validate()


@pytest.fixture
def cfg():
    return DetectorCfg(
        measurement=Measurement.FIS,
        probe=Probe.IS_Ideal,
        collimated_cosine=0.99,
    )


def test_cfg_is_ok(cfg):
    cfg.validate()


def test_cfg_all_measurement(cfg):
    for variant in Measurement:
        cfg.measurement = variant
        cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'measurement = None in DetectorCfg'),
        (dict(), 'measurement = None in DetectorCfg'),
    ]
)
def test_cfg_with_wrong_measurement(cfg, value, error):
    cfg.measurement = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


def test_cfg_all_probe(cfg):
    for variant in Probe:
        cfg.probe = variant
        cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'probe = None in DetectorCfg'),
        (dict(), 'probe = None in DetectorCfg'),
    ]
)
def test_cfg_with_wrong_probe(cfg, value, error):
    cfg.probe = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == 'probe = None in DetectorCfg'


@pytest.mark.parametrize(
    'value, error', [
        (None, 'collimated_cosine = None in DetectorCfg'),
        (dict(), 'collimated_cosine = None in DetectorCfg'),
        (-1.0, 'collimated_cosine = -1.0 out of range (0, 1) in DetectorCfg'),
        (1.0, 'collimated_cosine = 1.0 out of range (0, 1) in DetectorCfg'),
        (2.0, 'collimated_cosine = 2.0 out of range (0, 1) in DetectorCfg'),
    ]
)
def test_cfg_with_wrong_collimated_cosine(cfg, value, error):
    cfg.collimated_cosine = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error
