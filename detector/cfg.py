# TODO Move mods to separate file (import conflict)

from dataclasses import dataclass
from enum import Enum, auto


class Measurement(Enum):
    ALL = auto()
    FIS = auto()
    MIS = auto()


class Probe(Enum):
    cyrcle_port = auto()
    IS_Ideal = auto()
    IS_Thorlabs_200_4 = auto()


@dataclass
class DetectorCfg:
    measurment: Measurement = Measurement.FIS
    probe: Probe = Probe.IS_Ideal
    collimated_cosine: float = 0.99

    def validate(self):
        if self.measurment not in Measurement:
            raise ValueError('measurment = {measurment} not available')
        if self.probe not in Probe:
            raise ValueError('probe = {probe} not available')
        if not (0 < self.collimated_cosine < 1):
            raise ValueError('collimated_cosine = {self.collimated_cosine} out of range (0,1)')
