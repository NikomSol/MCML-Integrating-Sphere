from dataclasses import dataclass
import numpy as np
from multiprocess import cpu_count
import enum


@dataclass
class DirectProblemCfg:
    N: int = 1000
    threads: int = 1

    def validate(self):
        N = self.N
        threads = self.threads

        if not isinstance(N, int):
            raise TypeError(f'N = {N} is not int')
        if N < 1:
            raise ValueError(f'N = {N} out of range')

        if not isinstance(threads, int):
            raise TypeError(f'threads = {threads} is not int')
        if not (cpu_count > threads > 0):
            raise TypeError(f'threads = {threads} out of range')


class GeneratorModesDimension(enum.Enum):
    surface = 2


class GeneratorModesSpatialDistribution(enum.Enum):
    gauss = 1


class GeneratorModesAxialDistribution(enum.Enum):
    collimated = 1


@dataclass
class GeneratorCfg:
    mode_source_dimension: GeneratorModesDimension = GeneratorModesDimension.surface
    mode_spatial_distribution: GeneratorModesSpatialDistribution = GeneratorModesSpatialDistribution.gauss
    mode_angular_distribution:  GeneratorModesAxialDistribution = GeneratorModesAxialDistribution.collimated
    beam_center: np.ndarray = np.array([0, 0, 0])
    beam_diameter: float = 1

    def validate(self):
        # TODO GeneratorCfg.validate 
        raise NotImplementedError


class DetectorModesMeasurement(enum.Enum):
    FIS = 1


class DetectorModesDetector(enum.Enum):
    IS_Ideal = 1


@dataclass
class DetectorCfg:
    measurment: DetectorModesMeasurement = DetectorModesMeasurement.FIS
    detector: DetectorModesDetector = DetectorModesDetector.IS_Ideal
    collimated_cosine: float = 0.99

    def validate(self):
        # TODO DetectorCfg.validate 
        raise NotImplementedError


@dataclass
class Cfg:
    direct_problem: DirectProblemCfg()
    generator: GeneratorCfg()
    detector: DetectorCfg()

    def validate(self):
        # TODO Cfg.validate 
        raise NotImplementedError


@dataclass
class Layer:
    thick: float
    mu_a: float
    mu_s: float
    g: float
    n: float


class Sample:
    def __init__(self, layers: list[Layer]):
        self.layers = layers


class InverseProblem:
    """
    Parse inverse problem config:
        Choose direct problem
        Choose optimizer
    Solve inverse problem
    """
    def __init__(self, cfg: Cfg, sample: Sample):
        self.cfg = cfg
        self.samle = sample
        self._set_direct_problem()
        self._set_optimezer()

    def _set_direct_problem(self):
        raise NotImplementedError()

    def _set_optimezer(self):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

    def get_solution(self):
        raise NotImplementedError()


class Optimizer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def optimize(self):
        raise NotImplementedError()


class DirectProblem:
    """
    Parse direct problem config:
        Choose engine
        Choose generator
        Choose detector
    Solve direct problem, save data in Detector objeck?
    """
    def __init__(self, cfg: Cfg, sample: Sample):
        self.cfg = cfg
        self.sample = sample
        self._set_engine()
        self._set_generator()
        self._set_detector()

    def _set_engine(self):
        raise NotImplementedError()

    def _set_generator(self):
        raise NotImplementedError()

    def _set_detector(self):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

    def get_solution(self):
        raise NotImplementedError()


class Generator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def get_generator():
        raise NotImplementedError()


class Detector:
    """
    Save result of Engine / Direct problem work
    Generate empty save object and save function, that can be used by the Engine
    """
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.data = self.get_data_object()

    def get_data_object(self):
        raise NotImplementedError()

    def get_save_progres(self):
        raise NotImplementedError()

    def get_save_end(self):
        raise NotImplementedError()


class FixedIntegratingSphere(Detector):
    # TODO FixedIntegratingSphere
    def __init__(self) -> None:
        raise NotImplementedError()


class Engine:
    """
    Calculate engin function
    Run tracing and calculate detected signal, save data in Detector objeck?
    """
    def __init__(self, cfg: Cfg, sample: Sample, gen: Generator, det: Detector):
        self.cfg = cfg
        self.samle = sample
        self.gen = gen
        self.det = det
        self.trace = self._get_trace()

    def _get_move(self):
        raise NotImplementedError()

    def _get_turn(self):
        raise NotImplementedError()

    def _get_term(self):
        raise NotImplementedError()

    def _get_reflection(self):
        raise NotImplementedError()

    def _get_trace(self):
        raise NotImplementedError()

    def run(self, N=None):
        raise NotImplementedError()


class Plotter:
    def __init__(self) -> None:
        raise NotImplementedError()


class FileManager:
    def __init__(self) -> None:
        raise NotImplementedError()
