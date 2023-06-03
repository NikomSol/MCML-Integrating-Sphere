import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
import numpy as np

from detector import DetectorCfg, Measurement, Probe
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
    return Detector(cfg_ALL)


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
    save_progress = detector_ALL.get_func_save_progress_ALL()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_gen = np.ones((3, 3))
    p_move = np.ones((3, 3))
    p_term = np.ones((3, 3))
    p_turn = np.ones((3, 3))

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))

    storage = save_progress(p_gen, p_move, p_term, p_turn, storage)
    assert np.array_equal(storage, np.zeros(2))

    with pytest.raises(ValueError) as exc_info:
        storage = save_ending(p_gen, p_move, p_term, p_turn, storage)
    assert str(exc_info.value) == 'Photon ending with non-zero mass and without leaving medium'


def test_detector_ALL_save_in_medium_without_mass(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    save_progress = detector_ALL.get_func_save_progress_ALL()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_gen = np.ones((3, 3))
    p_move = np.ones((3, 3))
    p_term = np.ones((3, 3))
    p_turn = np.ones((3, 3))

    # photon terminate
    p_term[2, 0] = 0

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))

    storage = save_progress(p_gen, p_move, p_term, p_turn, storage)
    assert np.array_equal(storage, np.zeros(2))

    storage = save_ending(p_gen, p_move, p_term, p_turn, storage)
    assert np.array_equal(storage, np.array([0, 1]))


def test_detector_ALL_save_out_medium_with_mass(detector_ALL):
    get_storage = detector_ALL.get_func_get_storage()
    save_progress = detector_ALL.get_func_save_progress_ALL()
    save_ending = detector_ALL.get_func_save_ending_ALL()

    # photon in medium with mass
    p_gen = np.ones((3, 3))
    p_move = np.ones((3, 3))
    p_term = np.ones((3, 3))
    p_turn = np.ones((3, 3))

    # photon out
    p_move[2, 1] = -1
    p_move[2, 0] = 0.3

    storage = get_storage()
    assert np.array_equal(storage, np.zeros(2))

    storage = save_progress(p_gen, p_move, p_term, p_turn, storage)
    assert np.array_equal(storage, np.zeros(2))

    storage = save_ending(p_gen, p_move, p_term, p_turn, storage)
    assert np.array_equal(storage, np.array([0.3, 0.7]))
