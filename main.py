from cfg import Cfg
from source import Source, SourceCfg, Dimension, SpatialDistribution, AngularDistribution
from detector import Detector, DetectorCfg, Measurement
from sample import Sample, Layer, Material
from direct_problem import DirectProblem, DirectProblemCfg

import numpy as np


source_cfg = SourceCfg(dimension=Dimension.surface,
                       spatial_distribution=SpatialDistribution.gauss,
                       angular_distribution=AngularDistribution.collimated,
                       beam_center=np.array([0, 0, 0]),
                       beam_diameter=float(1))

detector_cfg = DetectorCfg(measurement=Measurement.ALL)

direct_problem_cfg = DirectProblemCfg(N=3,
                                      threads=1)

cfg = Cfg(
    source=source_cfg,
    detector=detector_cfg,
    direct_problem=direct_problem_cfg
    )

cfg.validate()

sample = Sample([
    Layer(material=Material.transparent,
          start=0., end=1.,
          mu_a=0.1, mu_s=1., g=0.9, n=1.5),
    Layer(material=Material.scattering,
          start=1., end=2.,
          mu_a=1., mu_s=1., g=0.9, n=1.5),
    Layer(material=Material.transparent,
          start=2., end=3.,
          mu_a=1., mu_s=1., g=0.9, n=1.5)
          ])

detector = Detector(cfg.detector)
source = Source(cfg.source, sample)

direct_problem = DirectProblem(cfg.direct_problem,
                               sample, source, detector)
# print('OK')
# print(direct_problem.solve())


p = np.array([[0, 0, 0],
              [0, 0.6, 0.8],
              [10 ** -5, np.NINF, 0]])
print(direct_problem.get_func_reflection()(p))
