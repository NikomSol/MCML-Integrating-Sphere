from dataclasses import dataclass

from direct_problem.cfg import DirectProblemCfg
from source.cfg import SourceCfg
from detector.cfg import DetectorCfg


@dataclass
class Cfg:
    direct_problem: DirectProblemCfg = DirectProblemCfg()
    source: SourceCfg = SourceCfg()
    detector: DetectorCfg = DetectorCfg()

    def validate(self):
        self.direct_problem.validate()
        self.source.validate()
        self.detector.validate()
