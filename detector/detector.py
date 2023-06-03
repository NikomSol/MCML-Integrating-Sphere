import numpy as np
from numba import njit


from .cfg import DetectorCfg
from .measurement import Measurement
from .probe import Probe


class Detector:
    def __init__(self, cfg: DetectorCfg):
        self.cfg = cfg
        self.parse_cfg()

    def parse_cfg(self):
        cfg = self.cfg
        probe = cfg.probe
        measurment = cfg.measurement

        if measurment is Measurement.ALL:
            self.get_func_save_progress = self.get_func_save_progress_ALL
            self.get_func_save_ending = self.get_func_save_ending_ALL
            self.get_func_get_storage = self.get_func_get_storage_ALL
        else:
            raise NotImplementedError()

    # ALL variant
    def get_func_save_progress_ALL(self):

        @njit(fastmath=True)
        def save_progress(p_gen, p_move, p_term, p_turn, storage):
            # TODO does supression unused variable increase the calculation time?
            _storage = storage * 1.
            _, _, _, _ = p_gen, p_move, p_term, p_turn
            return _storage

        return save_progress

    def get_func_save_ending_ALL(self):

        @njit(fastmath=True)
        def save_ending(p_gen, p_move, p_term, p_turn, storage):
            _storage = storage * 1.
            if p_move[2, 1] in {-1, -2}:
                _storage[0] += p_move[2, 0]
                _storage[1] += p_gen[2, 0] - p_move[2, 0]
                return _storage
            if p_term[2, 0] == 0:
                _storage[1] += p_gen[2, 0]
                return _storage
            raise ValueError('Photon ending with non-zero mass and without leaving medium')

        return save_ending

    def get_func_get_storage_ALL(self):

        @njit(fastmath=True)
        def get_storage():
            return np.zeros(2)

        return get_storage

    # def get_func_save_progress(self):

    #     @njit(fastmath=True)
    #     def save_progress():
    #         raise NotImplementedError()

    #     return save_progress

    # def get_func_save_ending(self):

    #     @njit(fastmath=True)
    #     def save_ending():
    #         raise NotImplementedError()

    #     return save_ending

    # def get_func_get_storage(self):

    #     @njit(fastmath=True)
    #     def get_storage():
    #         raise NotImplementedError()

    #     return get_storage
