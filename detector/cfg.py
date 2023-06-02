from dataclasses import dataclass

from .measurement import Measurement
from .probe import Probe


@dataclass
class DetectorCfg:
    measurement: Measurement = Measurement.FIS
    probe: Probe = Probe.IS_Ideal
    collimated_cosine: float = 0.99

    def validate(self) -> None:
        if not self.measurement:
            raise ValueError('measurement = None in DetectorCfg')
        if not isinstance(self.measurement, Measurement):
            raise ValueError('measurement is not of type Measurement in DetectorCfg')
        if self.measurement not in Measurement:
            raise ValueError(f'measurement = {self.measurement} not available in DetectorCfg')

        if not self.probe:
            raise ValueError('probe = None in DetectorCfg')
        if not isinstance(self.probe, Probe):
            raise ValueError('probe is not of type Probe in DetectorCfg')
        if self.probe not in Probe:
            raise ValueError(f'probe = {self.probe} not available in DetectorCfg')

        if not self.collimated_cosine:
            raise ValueError('collimated_cosine = None in DetectorCfg')
        if not isinstance(self.collimated_cosine, float):
            raise ValueError('collimated_cosine is not of type float in DetectorCfg')
        if not (0 < self.collimated_cosine < 1):
            raise ValueError(f'collimated_cosine = {self.collimated_cosine} out of range (0, 1) in DetectorCfg')
