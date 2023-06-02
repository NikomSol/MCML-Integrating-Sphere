# import numpy as np
# from numba import njit


# class DirectProblem:
#     def __init__(self, cfg, sample, source, detector):
#         self.cfg = cfg
#         self.sample = sample
#         self.source = source
#         self.detector = detector

#         self.trace = self._get_trace()

#     def _get_trace(self):
#         source = self.source
#         detector = self.detector

#         generator = source.func_generator

#         save_progress = detector.func_save_progress
#         save_ending = detector.func_save_ending
#         storage = detector.new_storage()

#         move = self.get_func_move()
#         term = self.get_func_term()
#         turn = self.get_func_turn()

#         @njit(fastmath=True)
#         def trace():
#             raise NotImplementedError()

#         return trace

#     def solve(self):
#         N = self.cfg.N
#         trace = self.trace
#         storage = self.detector.get_storage()

#         for _ in range(N):
#             storage += trace()

#         return storage
