from dataclasses import dataclass, field

from detector.cfg import DetectorCfg
from direct_problem.cfg import DirectProblemCfg
from source.cfg import SourceCfg


@dataclass
class Cfg:
    direct_problem: DirectProblemCfg = field(default_factory=DirectProblemCfg())
    source: SourceCfg = field(default_factory=SourceCfg())
    detector: DetectorCfg = field(default_factory=DetectorCfg())

    def validate(self):
        self.direct_problem.validate()
        self.source.validate()
        self.detector.validate()
