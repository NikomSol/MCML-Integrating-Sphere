from cfg import Cfg
from source import Source, SourceCfg, Dimension, SpatialDistribution, AngularDistribution
from detector import Detector, DetectorAll, IntegratingSphereIdeal, IntegratingSphereThorlabs, DetectorCfg, Measurement
from sample import Sample, Layer, Material
from direct_problem import DirectProblem, DirectProblemCfg

from support_functions import timeit, KW

import numpy as np


source_cfg = SourceCfg(dimension=Dimension.surface,
                       spatial_distribution=SpatialDistribution.gauss,
                       angular_distribution=AngularDistribution.collimated,
                       beam_center=np.array([0, 0, 0]),
                       beam_diameter=float(1))

detector_cfg = DetectorCfg(measurement=Measurement.MIS)

direct_problem_cfg = DirectProblemCfg(N=100000,
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
          mu_a=0.1, mu_s=1., g=0.9, n=1.4),
    Layer(material=Material.scattering,
          start=1., end=2.,
          mu_a=1., mu_s=1., g=0.9, n=1.5),
    Layer(material=Material.transparent,
          start=2., end=3.,
          mu_a=1., mu_s=1., g=0.9, n=1.4)
          ])

detector = cfg.detector.get_detector()
source = Source(cfg.source, sample)

direct_problem = DirectProblem(cfg.direct_problem,
                               sample, source, detector)

# direct_problem.solve()
print(timeit(direct_problem.solve)())
print(KW)
