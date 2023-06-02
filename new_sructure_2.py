#var 1.
sample_cfg = SampleCfg()
source_cfg = SourceCfg()
detector_cfg = DetectorCfg()
conditions_cfg = ConditionsCfg()

dir_prob_cfg = DirectProblemCfg()

cfg = Cfg(source = source_cfg,
          detector = detector_cfg,
          conditions = conditions_cfg,
          dir_prob = dir_prob_cfg,
          sample = sample_cfg)

#var 2.
cfg = Cfg('file_name')

source = Source(cfg.source)
detector = Detector(cfg.detector)
conditions = Conditions(cfg.conditions)
sample = Sample(cfg.sample)

dir_prob = DirectProblem(cfg = cfg.dir_prob, 
                         sample = sample,
                         source = source,
                         conditions = conditions,
                         detector = detector)

dir_prob.solve()
dir_prob.print_result()


new_sample_cfg = SampleCfg()
new_sample = Sample(new_sample_cfg)
dir_prob.sample = new_sample

dir_prob.solve()
dir_prob.print_result()


new_detector_cfg = DetectorCfg()
new_detector = Detector(new_detector_cfg)
dir_prob.detector = new_detector

dir_prob.solve()
dir_prob.print_result()

