import numpy as np
from numba import njit

from sample import Sample
from source import Source
from detector import Detector
from .cfg import DirectProblemCfg


class DirectProblem:
    def __init__(self, cfg: DirectProblemCfg,
                 sample: Sample, source: Source, detector: Detector):
        self.cfg = cfg
        self.sample = sample
        self.source = source
        self.detector = detector

        self.trace = self._get_trace()

    def _get_trace(self):
        source = self.source
        detector = self.detector

        generator = source.get_func_generator

        save_progress = detector.get_func_save_progress
        save_ending = detector.get_func_save_ending
        storage = detector.get_func_get_storage

        move = self.get_func_move()
        term = self.get_func_term()
        turn = self.get_func_turn()

        @njit(fastmath=True)
        def trace():
            raise NotImplementedError()

        return trace

    def solve(self):
        N = self.cfg.N
        trace = self.trace
        storage = self.detector.get_storage()

        for _ in range(N):
            storage += trace()

        return storage

    def get_func_move(self):
        
        @njit(fastmath=True)
        def move():
            raise NotImplementedError()
        
        return move

    def get_func_turn(self):
        
        @njit(fastmath=True)
        def turn():
            raise NotImplementedError()
        
        return turn

    def get_func_term(self):
        
        @njit(fastmath=True)
        def term():
            raise NotImplementedError()
        
        return term