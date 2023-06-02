# TODO Move mods to separate file (import conflict)

from dataclasses import dataclass
from enum import Enum, auto


class Measurement(Enum):
    ALL = auto()
    FIS = auto()
    MIS = auto()


class Probe(Enum):
    circle_port = auto()
    IS_Ideal = auto()
    IS_Thorlabs_200_4 = auto()


@dataclass
class DetectorCfg:
    measurement: Measurement = Measurement.FIS
    probe: Probe = Probe.IS_Ideal
    collimated_cosine: float = 0.99

    def validate(self) -> None:
        if not self.measurement:
            raise ValueError(f'measurement = None in DetectorCfg')
        if self.measurement not in Measurement:
            raise ValueError(f'measurement = {self.measurement} not available')

        if not self.probe:
            raise ValueError(f'probe = None in DetectorCfg')
        if self.probe not in Probe:
            raise ValueError(f'probe = {self.probe} not available')

        if not self.collimated_cosine:
            raise ValueError(f'collimated_cosine = None in DetectorCfg')
        if not (0 < self.collimated_cosine < 1):
            raise ValueError(f'collimated_cosine = {self.collimated_cosine} out of range (0, 1)')
