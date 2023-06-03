import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
from multiprocess import cpu_count

from direct_problem import DirectProblemCfg


@pytest.fixture
def empty_cfg():
    return DirectProblemCfg(
        N=None,
        threads=None,
    )


def test_empty_cfg(empty_cfg):
    with pytest.raises(ValueError):
        empty_cfg.validate()


@pytest.fixture
def cfg():
    return DirectProblemCfg(
        N=1,
        threads=1,
    )


def test_cfg_is_ok(cfg):
    cfg.validate()


@pytest.mark.parametrize(
    'value, error', [
        (None, 'N = None in DirectProblemCfg'),
        (dict(), 'N = None in DirectProblemCfg'),
        (0, 'N = None in DirectProblemCfg'),
        (-1, 'N = -1 out of range [1, +inf) in DirectProblemCfg'),
    ]
)
def test_cfg_with_wrong_n(cfg, value, error):
    cfg.N = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error


@pytest.mark.parametrize(
    'value, error', [
        (None, 'threads = None in DirectProblemCfg'),
        (dict(), 'threads = None in DirectProblemCfg'),
        (0, 'threads = None in DirectProblemCfg'),
        (-1, f'threads = -1 out of range [1, {cpu_count()}] in DirectProblemCfg'),
        (10000, f'threads = 10000 out of range [1, {cpu_count()}] in DirectProblemCfg'),
    ]
)
def test_cfg_with_wrong_threads(cfg, value, error):
    cfg.threads = value
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    assert str(exc_info.value) == error
