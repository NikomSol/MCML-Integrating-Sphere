import numpy as np
from numba import njit


# from .cfg import DetectorCfg
from .measurement import Measurement
from .probe import Probe


class Detector:
    pass


class DetectorAll(Detector):

    def get_func_get_storage(self):

        @njit(fastmath=True)
        def get_storage():
            return np.zeros(2)

        return get_storage

    def get_func_save_ending(self):

        @njit(fastmath=True)
        def save_ending(p_move, storage):
            _storage = storage * 1.

            if np.isposinf(p_move[2, 1]):
                _storage[0] += p_move[2, 0]
            if np.isneginf(p_move[2, 1]):
                _storage[1] += p_move[2, 0]

            return _storage

        return save_ending


class IntegratingSphereIdeal(Detector):
    sphere_diameter = 70
    port_diameter = 10
    gain = None

    def __init__(self, positions):
        self.positions = positions

    def get_func_get_storage(self):
        positions_num = len(self.positions)

        @njit(fastmath=True)
        def get_storage():
            return np.zeros((2, positions_num))

        return get_storage

    def get_func_save_ending(self):
        positions = self.positions
        port_radius = self.port_diameter / 2.
        sphere_diameter = self.sphere_diameter

        @njit(fastmath=True)
        def save_ending(p_move, storage):

            # 1 step: calc z1, z2 of intersept photon trace and port cylinder
            # 2 step: calc which position of port between z1, z2
            _storage = storage * 1.

            x0, y0 = p_move[0, :-1]
            z0 = 0
            cx, cy, cz = p_move[1]

            a = cx * cx + cy * cy
            b = 2 * (x0 * cx + y0 * cy)
            c = x0 * x0 + y0 * y0 - port_radius * port_radius

            # Collimated beam variant
            if a == 0:
                # Increase storage for top sphere if x**2 + y**2 < r**2
                if c < 0 and cz > 0:
                    _storage[0] += p_move[2, 0]
                return _storage

            det = b * b - 4 * a * c

            # Without trace-cylinder interseotions
            if det < 0:
                return _storage

            sq_det = np.sqrt(det)
            z_1 = z0 + cz * (-b - sq_det) / (2 * a)
            z_2 = z0 + cz * (-b + sq_det) / (2 * a)

            # step 2
            if cz > 0:
                for i, z in enumerate(positions):
                    if z_1 < z < z_2:
                        _storage[0, i] += p_move[2, 0]
            else:
                for i, z in enumerate(positions):
                    if (z_2 < z < z_1) and not (z_2 < z + sphere_diameter < z_1):
                        _storage[1, i] += p_move[2, 0]

            return _storage

        return save_ending


class IntegratingSphereThorlabs(Detector):
    sphere_diameter = 70

    port_1_diameter = 12.5
    port_1_black_height = 1
    port_1_white_height = 1

    port_2_diameter = 12.5
    port_2_black_height = 1
    port_2_white_height = 1

    gain = 1

    sphere_diameter = 70

    port_1_diameter = 12.5
    port_2_diameter = 12.5

    gain = 1

    def __init__(self, positions):
        raise NotImplemented('class IntegratingSphereThorlabs')


# class Detector:
#     def __init__(self, cfg: DetectorCfg):
#         self.cfg = cfg
#         self.parse_cfg()

#     def parse_cfg(self):
#         cfg = self.cfg
#         probe = cfg.probe
#         measurment = cfg.measurement

#         if measurment is Measurement.ALL:
#             self.get_func_save_progress = self.get_func_save_progress_ALL
#             self.get_func_save_ending = self.get_func_save_ending_ALL
#             self.get_func_get_storage = self.get_func_get_storage_ALL
#         else:
#             raise NotImplementedError()

#     # ALL variant
#     def get_func_save_progress_ALL(self):

#         @njit(fastmath=True)
#         def save_progress(p_gen, p_move, p_term, p_turn, storage):
#             # TODO does supression unused variable increase the calculation time?
#             _storage = storage * 1.
#             _, _, _, _ = p_gen, p_move, p_term, p_turn
#             return _storage

#         return save_progress

#     def get_func_save_ending_ALL(self):

#         @njit(fastmath=True)
#         def save_ending(p_gen, p_move, p_term, p_turn, storage):
#             _storage = storage * 1.
#             if np.isinf(p_move[2, 1]):
#                 _storage[0] += p_move[2, 0]
#                 _storage[1] += p_gen[2, 0] - p_move[2, 0]
#                 return _storage
#             if p_term[2, 0] == 0:
#                 _storage[1] += p_gen[2, 0]
#                 return _storage
#             raise ValueError('Photon ending with non-zero mass and without leaving medium')

#         return save_ending

#     def get_func_get_storage_ALL(self):

#         @njit(fastmath=True)
#         def get_storage():
#             return np.zeros(2)

#         return get_storage