#!/usr/bin/env python
# import subprocess
# import sys
# package = 'some_package'
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numba'])

import numpy as np
from numba import jit


@jit  # (nopython=True)
def turn(old_p, g):
    p = old_p * 1.
    th = np.arccos(1 / 2 / g * (1 + g ** 2 - ((1 - g ** 2) / (1 - g + 2 * g * np.random.random(1)[0])) ** 2))
    ph = 2 * np.pi * np.random.random(1)[0]

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_ph = np.sin(ph)
    cos_ph = np.cos(ph)

    if p[5] != 1. and p[5] != -1.:
        cx = (sin_th * (p[3] * p[5] * cos_ph - p[4] * sin_ph)) / (np.sqrt(1 - p[5] ** 2)) + p[3] * cos_th
        cy = (sin_th * (p[4] * p[5] * cos_ph + p[3] * sin_ph)) / (np.sqrt(1 - p[5] ** 2)) + p[4] * cos_th
        cz = -(np.sqrt(1 - p[5] ** 2)) * sin_th * cos_ph + p[5] * cos_th
    if p[5] == 1.:
        cx = sin_th * cos_ph
        cy = sin_th * sin_ph
        cz = cos_th
    if p[5] == -1.:
        cx = sin_th * cos_ph
        cy = -sin_th * sin_ph
        cz = -cos_th

    p[3] = cx
    # p[4] = cy
    p[5] = cz

    return p


print('type(1)')
print(type(turn(np.array([0, 0, 0, 0, 0, 1, 1, 1]), 0.98) + 1))
