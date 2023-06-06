import sys

# Well, you'll have to deal with that for a while
sys.path.append(".")

import pytest
import numpy as np

from sample import Sample, Layer, Material


@pytest.fixture
def empty_sample():
    return Sample([])


def test_empty_sample(empty_sample):
    assert empty_sample.boundaries_list == []
    assert empty_sample.g_table == []


@pytest.fixture
def scattering_sample():
    return Sample([Layer(material=Material.scattering,
                         start=0., end=1.,
                         mu_a=1., mu_s=1., g=0.9, n=1.5)])


def test_scattering_sample(scattering_sample):
    assert np.array_equal(scattering_sample.boundaries_list,
                          np.array([0, 1]))


@pytest.fixture
def classic_sample():
    return Sample([
        Layer(material=Material.transparent,
              start=0., end=1.,
              mu_a=0., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.scattering,
              start=1., end=2.,
              mu_a=1., mu_s=1., g=0.9, n=1.5),
        Layer(material=Material.transparent,
              start=2., end=3.,
              mu_a=1., mu_s=1., g=0.9, n=1.5)
        ])


def test_classic_sample(classic_sample):
    assert np.array_equal(classic_sample.boundaries_list,
                          np.array([0, 1, 2, 3]))
    assert np.array_equal(classic_sample.g_table,
                          np.array([0.9, 0.9, 0.9]))


def test_layer_index(classic_sample):
    layer_index = classic_sample.get_func_layer_index()
    assert layer_index(-1.) == -1
    assert layer_index(0) == -1
    assert layer_index(0.5) == 0
    assert layer_index(1) == 0
    assert layer_index(1.5) == 1
    assert layer_index(2) == 1
    assert layer_index(2.5) == 2
    assert layer_index(3) == 2
    assert layer_index(3.5) == -2
