from enum import Enum, auto


class Measurement(Enum):
    ALL = auto()
    CollimatedDiffuse = auto()
    MIS_Ideal = auto()
    MIS_Thorlabs = auto()
