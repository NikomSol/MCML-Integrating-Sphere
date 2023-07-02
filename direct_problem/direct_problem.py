import numpy as np
from numba import njit

from detector import Detector
from sample import Sample
from source import Source
from .cfg import DirectProblemCfg
from support_functions import timeit


class DirectProblem:
    def __init__(self, cfg: DirectProblemCfg, sample: Sample, source: Source, detector: Detector):
        self.cfg = cfg
        self.sample = sample
        self.source = source
        self.detector = detector

    def solve(self):
        N = self.cfg.N
        trace = self.get_func_trace()
        storage_emission = self.detector.get_func_get_storage_emission()()
        storage_absorption = self.get_func_get_storage_absorption()()

        for _ in range(N):
            _storage_emission, _storage_absorption = trace()
            storage_emission += _storage_emission
            storage_absorption += _storage_absorption

        return storage_emission, storage_absorption

    def get_func_trace(self):
        source = self.source
        detector = self.detector

        generate = source.get_func_generator()

        save_emission = detector.get_func_save_emission()
        get_storage_emission = detector.get_func_get_storage_emission()

        save_absorption = self.get_func_save_absorption()
        get_storage_absorption = self.get_func_get_storage_absorption()

        move = self.get_func_move()
        term = self.get_func_term()
        turn = self.get_func_turn()

        # @njit(fastmath=True)
        def trace():
            storage_emission = get_storage_emission()
            storage_absorption = get_storage_absorption()

            p_gen = generate()  # Save start photon parameters
            p_turn = p_gen * 1.  # First cycle photon parameters
            p_term = p_gen * 1.  # First cycle photon parameters
            for _ in range(10 ** 3):
                p_move = move(p_turn)

                # print(_)
                # print(p_move)
                # print(p_term)

                # Check leave z-area
                if np.isinf(p_move[2, 1]):
                    break
                # Check leave xy-area
                if p_move[2, 0] == 0:
                    break
                # Check small weight
                p_term = term(p_move)
                if p_term[2, 0] == 0:
                    break

                p_turn = turn(p_term)

            storage_emission = save_emission(p_move, storage_emission)
            storage_absorption = save_absorption(p_gen, p_move, p_term, storage_absorption)
            # print(storage)
            # print(p_gen)
            # print(p_move)
            # print(p_term)
            # print(p_turn)
            return storage_emission, storage_absorption

        return trace

    def get_func_move(self):
        sample = self.sample
        z_end_table = sample.boundaries_list[1:]
        z_start_table = sample.boundaries_list[:-1]
        mu_a_table = sample.mu_a_table
        mu_s_table = sample.mu_s_table

        reflection = self.get_func_reflection()

        l_rand_tol = 1e-5  # Минимальное значение l_rand

        # @njit(fastmath=True)
        def move(old_p):
            # Считаем, что Sum(l*mu_t)=-log(x)=l_rand, и при переходе через границу
            # убавляем l_rand в соответствии с пройденным путем

            p = old_p * 1.

            # Проверка отражения при входе в среду
            if np.isinf(p[2, 1]):
                if np.isneginf(p[2, 1]):
                    l_part = (z_start_table[0] - p[0, 2]) / p[1, 2]
                else:
                    l_part = (z_end_table[-1] - p[0, 2]) / p[1, 2]

                p[0, 0] = p[0, 0] + l_part * p[1, 0]
                p[0, 1] = p[0, 1] + l_part * p[1, 1]
                p[0, 2] = p[0, 2] + l_part * p[1, 2]

                p = reflection(p)

                if np.isinf(p[2, 1]):
                    return p

            # Создание случайной безразмерной величины длины
            l_rand = -np.log(np.random.random(1))[0]
            while l_rand > l_rand_tol:
                # FIXME Перенести проверку на уход из расчетной области 50х50 сюды

                # Определяем текущие границы и свойства
                layer_index = int(p[2, 1])

                z_start = z_start_table[layer_index]
                z_end = z_end_table[layer_index]

                mu_a = mu_a_table[layer_index]
                mu_s = mu_s_table[layer_index]
                mu_t = mu_a + mu_s

                # В рассеивающей среде мы определяем длину свободного пробега
                # и проверяем не рассеялись ли мы внутри среды
                if mu_s != 0:

                    l_free_path = 1 / mu_s
                    l_scattering = l_rand * l_free_path

                    # Рассчитываем на какую величину мы должны переместиться
                    new_p_z = p[0, 2] + l_scattering * p[1, 2]
                    # Проверка выхода за границу
                    if z_start < new_p_z < z_end:
                        # Без взаимодействия с границей раздела сред
                        new_p_x = p[0, 0] + l_scattering * p[1, 0]
                        new_p_y = p[0, 1] + l_scattering * p[1, 1]
                        p[0, 0], p[0, 1], p[0, 2] = new_p_x, new_p_y, new_p_z
                        p[2, 0] = p[2, 0] * np.exp(-mu_a * l_scattering)
                        break
                        # С взаимодействием с границей раздела сред

                # Расчет на сколько мы переместились до границы
                if p[1, 2] > 0:
                    l_part = (z_end - p[0, 2]) / p[1, 2]
                else:
                    l_part = (z_start - p[0, 2]) / p[1, 2]

                # Уменьшаем случайную l_rand в соответствии с пройденным путем
                l_rand = l_rand - mu_s * l_part

                # Расчет новой координаты на границе
                new_p_x = p[0, 0] + l_part * p[1, 0]
                new_p_y = p[0, 1] + l_part * p[1, 1]
                new_p_z = p[0, 2] + l_part * p[1, 2]

                p[0, 0], p[0, 1], p[0, 2] = new_p_x, new_p_y, new_p_z
                p[2, 0] = p[2, 0] * np.exp(-mu_a * l_part)

                # Проверка отражения
                p = reflection(p)
                if np.isinf(p[2, 1]):
                    break
                
                # Проверка вылета за расчетную XY - область
                if p[0, 1] > 50 or p[0, 2] > 50:
                    p[2, 0] = 0
                    break

            return p

        return move

    def get_func_turn(self):
        g_table = self.sample.g_table

        # @njit(fastmath=True)
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
            if cz0 not in [-1., 1.]:
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

        # @njit(fastmath=True)
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

        # @njit(fastmath=True)
        def reflection(old_p):
            p = old_p * 1.
            layer_index = p[2, 1]
            layer_number = len(n_table)
            cz1 = p[1, 2]

            if np.isinf(layer_index):
                n1 = 1
                if np.isneginf(layer_index):
                    next_layer_index = 0
                else:
                    next_layer_index = layer_number - 1
                n2 = n_table[next_layer_index]
            else:
                layer_index = int(layer_index)
                n1 = n_table[layer_index]
                if layer_index == 0 and cz1 < 0:
                    next_layer_index = np.NINF
                    n2 = 1
                elif layer_index == layer_number - 1 and cz1 > 0:
                    next_layer_index = np.PINF
                    n2 = 1
                else:
                    next_layer_index = int(layer_index + np.sign(cz1))
                    n2 = n_table[next_layer_index]

            inv_n = n1 / n2
            sz1 = np.sqrt(1 - cz1 * cz1)
            sz2 = inv_n * sz1

            # Full back reflection
            if sz2 >= 1:
                p[1, 2] = - cz1
                return p

            cz2 = np.sqrt(1 - sz2 * sz2) * np.sign(cz1)

            if np.random.rand() < R_frenel(cz1, cz2, inv_n):
                p[1, 2] = - cz1
                return p

            p[1, 2] = cz2
            p[1, 0] = p[1, 0] * inv_n
            p[1, 1] = p[1, 1] * inv_n
            p[2, 1] = next_layer_index

            return p

        return reflection

    def get_func_R_frenel(self):

        # @njit(fastmath=True)
        def R_frenel(cz1, cz2, inv_n):
            a_s = inv_n * cz1 - cz2
            b_s = inv_n * cz1 + cz2
            rs = a_s / b_s

            a_p = cz1 - inv_n * cz2
            b_p = cz1 + inv_n * cz2
            rp = a_p / b_p

            rsp = 0.5 * (rs * rs + rp * rp)

            return rsp

        return R_frenel

    # absorption saving
    def get_func_get_storage_absorption(self):

        @njit(fastmath=True)
        def get_storage_absorption():
            return 0

        return get_storage_absorption

    def get_func_save_absorption(self):

        @njit(fastmath=True)
        def save_absorption(p_gen, p_move, p_term, storage_absorption):
            _storage = storage_absorption * 1.

            if np.isinf(p_move[2, 1]):
                _storage += p_gen[2, 0] - p_move[2, 0]
            elif p_move[2, 0] == 0:
                None
            elif p_term[2, 0] == 0:
                _storage += p_gen[2, 0]
            else:
                raise ValueError('Photon in sample with mass left trace cyrcle')
            return _storage

        return save_absorption
