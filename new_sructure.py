from dataclasses import dataclass
import numpy as np


@dataclass
class Configuration:
    # DirectProblem cfg
    N: int
    threads: int = 1

    # Generator cfg
    source_dimension: int = 2
    spatial_distribution: str = 'Gauss'
    angular_distribution: str = 'Collimated'
    beam_center: np.ndarray = np.array([0, 0, 0])
    beam_diameter: float = 1

    # Detector cfg
    measurment_type: str = 'FIS'
    detector_type: str = 'IS_Ideal'
    collimated_cosine: float = 0.99


class Sample:
    def __init__(self, layers: list):
        self.layers = layers


@dataclass
class Layer:
    thick: float
    mu_a: float
    mu_s: float
    g: float
    n: float


class InverseProblem:
    """
    Parse inverse problem config:
        Choose direct problem
        Choose optimizer
    Solve inverse problem
    """
    def __init__(self, cfg: Configuration, sample: Sample):
        self.cfg = cfg
        self.samle = sample
        self._set_direct_problem()
        self._set_optimezer()

    def _set_direct_problem(self):
        pass

    def _set_optimezer(self):
        pass

    def solve(self):
        pass

    def get_solution(self):
        pass


class Optimizer:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def optimize(self):
        pass


class DirectProblem:
    """
    Parse direct problem config:
        Choose engine
        Choose generator
        Choose detector
    Solve direct problem, save data in Detector objeck?
    """
    def __init__(self, cfg: Configuration, sample: Sample):
        self.cfg = cfg
        self.sample = sample
        self._set_engine()
        self._set_generator()
        self._set_detector()

    def _set_engine(self):
        pass

    def _set_generator(self):
        pass

    def _set_detector(self):
        pass

    def solve(self):
        pass

    def get_solution(self):
        pass


class Generator:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def get_generator():
        pass


class Detector:
    """
    Save result of Engine / Direct problem work
    Generate empty save object and save function, that can be used by the Engine
    """
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.data = self.get_data_object()

    def get_data_object(self):
        pass

    def get_save_progres(self):
        pass

    def get_save_end(self):
        pass


class Engine:
    """
    Calculate engin function
    Run tracing and calculate detected signal, save data in Detector objeck?
    """
    def __init__(self, cfg: Configuration, sample: Sample, gen: Generator, det: Detector):
        self.cfg = cfg
        self.samle = sample
        self.gen = gen
        self.det = det
        self.trace = self._get_trace()

    def _get_move(self):
        pass

    def _get_turn(self):
        pass

    def _get_term(self):
        pass

    def _get_reflection(self):
        pass

    def _get_trace(self):
        pass

    def run(self, N=None):
        pass


class Plotter:
    pass


class FileManager:
    pass
