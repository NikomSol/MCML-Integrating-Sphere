from dataclasses import dataclass
import numpy as np

from .measurement import Measurement
from .detector import DetectorAll, DetectorCollimatedDiffuse
from .detector import IntegratingSphereIdeal, IntegratingSphereThorlabs


@dataclass
class DetectorCfg:
    measurement: Measurement = None
    positions: np.ndarray = None

    def validate(self) -> None:
        if not self.measurement:
            raise ValueError('measurement = None in DetectorCfg')
        if not isinstance(self.measurement, Measurement):
            raise ValueError('measurement is not of type Measurement in DetectorCfg')
        if self.measurement not in Measurement:
            raise ValueError(f'measurement = {self.measurement} not available in DetectorCfg')

        if (self.measurement is Measurement.MIS_Ideal or self.measurement is Measurement.MIS_Thorlabs):
            if not isinstance(self.positions, np.ndarray):
                raise ValueError('positions is not of type np.ndarray in DetectorCfg')
            if not (0 < len(self.positions)):
                raise ValueError(f'len positions = {len(self.positions)} out of range (1, +inf) in DetectorCfg')

    def get_detector(self):
        measurement = self.measurement
        positions = self.positions

        if measurement is Measurement.ALL:
            return DetectorAll()
        elif measurement is Measurement.CollimatedDiffuse:
            return DetectorCollimatedDiffuse()
        elif measurement is Measurement.MIS_Ideal:
            return IntegratingSphereIdeal(positions)
        elif measurement is Measurement.MIS_Thorlabs:
            return IntegratingSphereThorlabs
        else:
            raise ValueError(f'measurement = {self.measurement} not available in DetectorCfg.get_detector')
