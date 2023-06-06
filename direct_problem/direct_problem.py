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

    def get_func_trace(self):
        source = self.source
        detector = self.detector

        generate = source.get_func_generator()

        save_progress = detector.get_func_save_progress()
        save_ending = detector.get_func_save_ending()
        get_storage = detector.get_func_get_storage()

        move = self.get_func_move()
        term = self.get_func_term()
        turn = self.get_func_turn()

        @njit(fastmath=True)
        def trace():
            storage = get_storage()
            p_gen = generate()  # Save start photon parameters
            p_turn = p_gen * 1.  # First cycle photon parameters
            for _ in range(10 ** 3):
                p_move = move(p_turn)

                # Check leave calc area
                if (p_move[2, 1] == -1 or p_move[2, 1] == -2):
                    break
                # Check small weight
                p_term = term(p_move)
                if p_term[2, 0] == 0:
                    break

                p_turn = turn(p_term)
                storage = save_progress(p_gen, p_move, p_term, p_turn, storage)

            storage = save_ending(p_gen, p_move, p_term, p_turn, storage)
            return storage

        return trace

    def solve(self):
        N = self.cfg.N
        trace = self.get_func_trace()
        storage = self.detector.get_storage()

        for _ in range(N):
            storage += trace()

        return storage

    def get_func_move(self):
        sample = self.sample
        z_end_table = sample.boundaries_list[1:]
        z_start_table = sample.boundaries_list[:-1]
        mu_a_table = sample.mu_a_table
        mu_s_table = self.sample.mu_s_table

        reflection = self.get_func_reflection()

        @njit(fastmath=True)
        def move(old_p):
            # Считаем, что Sum(l*mu_t)=-log(x)=l_rand, и при переходе через границу
            # убавляем l_rand в соответсвии с пройденным путем
            l_rand_tol = 1e-5  # Минимальное значение l_rand
            p = old_p * 1.
            # Создание случайной безразмерной величины длины
            l_rand = -np.log(np.random.random(1))[0]
            while l_rand > l_rand_tol:

                # Определяем текущие границы и свойства
                layer_index = int(p[2, 1])

                z_start = z_start_table[layer_index]
                z_end = z_end_table[layer_index]

                mu_a = mu_a_table[layer_index]
                mu_s = mu_s_table[layer_index]
                mu_t = mu_a + mu_s

                l_free_path = 1 / mu_t
                l_layer = l_rand * l_free_path

                # Расчитываем на какую величину мы должны переместиться
                new_p_z = p[0, 2] + l_layer * p[1, 2]
                # Проверка выхода за границу
                if z_start < new_p_z < z_end:
                    # Без взаимодействия с границей раздела сред
                    new_p_x = p[0, 0] + l_layer * p[1, 0]
                    new_p_y = p[0, 1] + l_layer * p[1, 1]
                    p[0, 0], p[0, 1], p[0, 2] = new_p_x, new_p_y, new_p_z
                    p[2, 0] = p[2, 0] * np.exp(-mu_a * l_layer)
                    break
                    # С взаимодействием с границей раздела сред

                # Расчет на сколько мы переместились до границы
                if p[1, 2] > 0:
                    l_part = (z_end - p[0, 2]) / p[1, 2]
                else:
                    l_part = (z_start - p[0, 2]) / p[1, 2]

                # Уменьшаем случайную l_rand в соответсвии с пройденным путем
                l_rand = l_rand - mu_t * l_part

                # Расчет новой координаты на границе
                new_p_x = p[0, 0] + l_part * p[1, 0]
                new_p_y = p[0, 1] + l_part * p[1, 1]
                new_p_z = p[0, 2] + l_part * p[1, 2]

                p[0, 0], p[0, 1], p[0, 2] = new_p_x, new_p_y, new_p_z
                p[2, 0] = p[2, 0] * np.exp(-mu_a * l_part)

                # Проверка отражения
                p = reflection(p)
                if p[2, 1] == -1 or p[2, 1] == -2:
                    break

            return p

        return move

    def get_func_turn(self):
        g_table = self.sample.g_table

        @njit(fastmath=True)
        def turn(old_p):
            p = old_p * 1.
            g = g_table[int(p[2, 1])]
            if g != 0.:
                th = np.arccos(0.5 / g * (1 + g ** 2 - ((1 - g ** 2) / (1 - g + 2 * g * np.random.rand())) ** 2))
            else:
                th = np.arccos(1 - 2 * np.random.rand())
            ph = 2 * np.pi * np.random.rand()

            sin_th = np.sin(th)
            cos_th = np.cos(th)
            sin_ph = np.sin(ph)
            cos_ph = np.cos(ph)

            cx0 = p[1, 0]
            cy0 = p[1, 1]
            cz0 = p[1, 2]
            if cz0 != 1. and cz0 != -1.:
                cx = (sin_th * (cx0 * cz0 * cos_ph - cy0 * sin_ph)) / (np.sqrt(1 - cz0 ** 2)) + cx0 * cos_th
                cy = (sin_th * (cy0 * cz0 * cos_ph + cx0 * sin_ph)) / (np.sqrt(1 - cz0 ** 2)) + cy0 * cos_th
                cz = -(np.sqrt(1 - cz0 ** 2)) * sin_th * cos_ph + cz0 * cos_th
            if cz0 == 1.:
                cx = sin_th * cos_ph
                cy = sin_th * sin_ph
                cz = cos_th
            if cz0 == -1.:
                cx = sin_th * cos_ph
                cy = -sin_th * sin_ph
                cz = -cos_th
            # cx0, cy0, cz0 = cx, cy, cz
            p[1, 0], p[1, 1], p[1, 2] = cx, cy, cz

            return p

        return turn

    def get_func_term(self):

        @njit(fastmath=True)
        def term(old_p):
            p = old_p * 1.
            mass = p[2, 0]
            threshold_m = 10 ** -4
            threshold_factor = 10
            if (mass < threshold_m) & (mass != 0.):
                if np.random.rand() <= 1 / threshold_factor:
                    mass = mass * threshold_factor
                else:
                    mass = 0.
            p[2, 0] = mass
            return p

        return term

    def get_func_reflection(self):

        n_table = self.sample.n_table
        R_frenel = self.get_func_R_frenel()

        @njit(fastmath=True)
        def reflection(old_p):
            p = old_p * 1.
            layer_number = len(n_table)
            layer_index = int(p[2, 1])
            cz = p[1, 2]
            n1 = n_table[layer_index]

            if (layer_index == layer_number - 1) and (cz > 0):
                n2 = 1
                buf = -2
            elif (layer_index != layer_number - 1) and (cz > 0):
                n2 = n_table[layer_index + 1]
                buf = layer_index + 1
            elif (layer_index == 0) and (cz < 0):
                n2 = 1
                buf = -1
            else:
                n2 = n_table[layer_index - 1]
                buf = layer_index - 1

            if np.random.rand() < R_frenel(np.arccos(cz), n1, n2):
                cz = -cz
            else:
                cz = np.sqrt(1 - (n1 / n2) ** 2 * (1 - cz ** 2)) * np.sign(cz)
                p[2, 1] = buf

            return p

        return reflection

    def get_func_R_frenel(self):
        @njit(fastmath=True)
        def R_frenel(th, n1, n2):
            if th > np.pi / 2:
                th = np.pi - th

            n = n1 / n2
            cos_th1 = np.cos(th)
            cos_th2 = np.cos(np.arcsin(n * np.sin(th)))

            def rs(cos_th1, cos_th2, n1, n2):
                a = n1 * cos_th1 - n2 * cos_th2
                b = n1 * cos_th1 + n2 * cos_th2
                return a / b

            def rp(cos_th1, cos_th2, n1, n2):
                a = n2 * cos_th1 - n1 * cos_th2
                b = n2 * cos_th1 + n1 * cos_th2
                return a / b

            if np.sin(th) >= n2 / n1:
                res = 1.0
            else:
                res = 0.5 * ((rs(cos_th1, cos_th2, n1, n2)) ** 2 + (rp(cos_th1, cos_th2, n1, n2)) ** 2)

            if res < 1e-6:
                res = 0.0
            return res

        return R_frenel