from cfg import Cfg

from source import SourceCfg, Dimension, SpatialDistribution, AngularDistribution
from source import Source

from detector import DetectorCfg, Measurement

from sample import Sample, Layer, Material

from direct_problem import DirectProblemCfg
from direct_problem import DirectProblem


from support_functions import timeit, KW

import numpy as np


source_cfg = SourceCfg(dimension=Dimension.surface,
                       spatial_distribution=SpatialDistribution.gauss,
                       angular_distribution=AngularDistribution.collimated,
                       beam_center=np.array([0, 0, 0]),
                       beam_diameter=float(1))

detector_cfg = DetectorCfg(measurement=Measurement.CollimatedDiffuse)

direct_problem_cfg = DirectProblemCfg(N=100,
                                      threads=1)

cfg = Cfg(
    source=source_cfg,
    detector=detector_cfg,
    direct_problem=direct_problem_cfg
    )

cfg.validate()

sample = Sample([
    Layer(material=Material.transparent, depth=1.,
          mu_a=0., mu_s=0., g=0., n=1.45),
    Layer(material=Material.scattering, depth=1.,
          mu_a=1., mu_s=1., g=0.9, n=1.35),
    Layer(material=Material.transparent, depth=1.,
          mu_a=0., mu_s=0., g=0.0, n=1.45)
          ])

detector = cfg.detector.get_detector()
source = Source(cfg.source, sample)

direct_problem = DirectProblem(cfg.direct_problem,
                               sample, source, detector)

# direct_problem.solve()
result = timeit(direct_problem.solve)()
print(result)
print(np.sum(result[0]) + result[1])
print(KW)
