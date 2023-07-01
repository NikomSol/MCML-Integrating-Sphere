# import sys

# # Well, you'll have to deal with that for a while
# sys.path.append(".")

# import numpy as np
# import pytest

# from detector import DetectorCfg, Measurement
# from direct_problem import DirectProblem, DirectProblemCfg
# from sample import Sample, Layer, Material
# from source import Source, AngularDistribution, Dimension, SpatialDistribution, SourceCfg

# @pytest.fixture
# def layers():
#     return {'tissue1':Layer(material=Material.transparent, start=0., end=1., n=1.5, mu_a=1., mu_s=0., g=1)}

# @pytest.fixture
# def tissue1():
#     return Sample([Layer(material=Material.transparent,
#                          start=0., end=1., n=1.5,
#                          mu_a=1., mu_s=0., g=1)])