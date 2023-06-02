from direct_problem.cfg import DirectProblemCfg
from source.cfg import SourceCfg
from detector.cfg import DetectorCfg
from cfg import Cfg

from sample.sample import Sample, Layer
from source.source import Source

# from direct_problem.direct_problem import DirectProblem


source_cfg = SourceCfg()
detector_cfg = DetectorCfg()
dir_prob_cfg = DirectProblemCfg()

cfg = Cfg(source=source_cfg,
          detector=detector_cfg,
          direct_problem=dir_prob_cfg)

sample = Sample([
    Layer(start=0., end=1.,
          mu_a=1., mu_s=1., g=0.9, n=1.5),
    Layer(start=1., end=2.,
          mu_a=1., mu_s=1., g=0.9, n=1.5)
])

layer_index = sample.get_func_layer_index()
assert (layer_index(1.5) == 1)
assert (layer_index(0.5) == 0)
assert (layer_index(2.5) == -2)
assert (layer_index(-1.) == -1)

source = Source(cfg.source, sample)