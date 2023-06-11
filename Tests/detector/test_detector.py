import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
import numpy as np

from detector import DetectorCfg, Measurement
from detector import Detector


@pytest.fixture
def cfg_ALL():
    return DetectorCfg(
        measurement=Measurement.ALL,
    )


def test_cfg_ALL(cfg_ALL):
    cfg_ALL.validate()


@pytest.fixture
def detector_ALL(cfg_ALL):
    return cfg_ALL.get_detector()


def test_detector_ALL_storage(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))

    storage += np.ones(2)
    new_storage = get_storage()
    assert np.array_equal(storage, np.ones(2))
    assert np.array_equal(new_storage, np.zeros(2))


def test_detector_ALL_save_in_medium_with_mass(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_move = np.ones((3, 3))

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))
    storage = save_ending(p_move,storage)
    assert np.array_equal(storage, np.zeros(2))
   


def test_detector_ALL_save_in_medium_without_mass(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_move = np.ones((3, 3))

    # photon terminate
    p_move[2, 0] = 0

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))
    storage = save_ending(p_move, storage)
    assert np.array_equal(storage, np.zeros(2))


def test_detector_ALL_save_out_medium_with_mass(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_move = np.ones((3, 3))

    # photon out bottom
    p_move[2, 1] = np.NINF
    p_move[2, 0] = 0.3

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))
    storage = save_ending(p_move, storage)
    assert np.array_equal(storage, np.array([0, 0.3]))

    # photon out top
    p_move[2, 1] = np.PINF
    p_move[2, 0] = 0.3

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))
    storage = save_ending(p_move, storage)
    assert np.array_equal(storage, np.array([0.3, 0]))
