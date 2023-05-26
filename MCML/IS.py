# IS class
# Class functions: parse IS modes, get difference data
# Input cnf, object
# Output data dicts/array and graphs


import numpy as np
from multiprocess import Pool, cpu_count
from numba import njit

"""
Суть процесса
- интерпретировать входные данные cnf, obj
- в зависимости от типа задачи определяем numba функции gen, move, term, turn....
- запускаем run с использованием полученных функций в котором параллелим в каждом потоке считая по N фотонов
"""


class MCML:
    def __init__(self, cnf: dict, obj: list) -> None:
        """
        :param cnf: Настройки решателя (todo через классы)
        :param obj: Геометрия и свойства среды (todo через классы)
        """
        self.cnf = cnf
        self.obj = obj

        self.N = cnf['N']
        self.threads = cnf['threads']
        layers_number = len(obj)
        self.z_start_table = np.array([obj[i]['z_start'] for i in range(layers_number)])
        self.z_end_table = np.array([obj[i]['z_end'] for i in range(layers_number)])
        self.n_table = np.array([obj[i]['n'] for i in range(layers_number)])
        self.mu_a_table = np.array([obj[i]['mu_a'] for i in range(layers_number)])
        self.mu_s_table = np.array([obj[i]['mu_s'] for i in range(layers_number)])
        self.g_table = np.array([obj[i]['g'] for i in range(layers_number)])

        self.parse_modes()

    def parse_modes(self):
        """
        Определяем моды задачи, указанные в cnf
        """
        self.parse_mode_generator()
        self.parse_mode_save()

    # parse_mode_generator
    def parse_mode_generator(self):
        """
        Определяем функцию генератора фотонов
        """
        cnf = self.cnf

        # генерация фотонов на поверхности
        if cnf['mode_generator'] == 'Surface':

            # определяем функции распределения фотонов при генерации по координате и углу
            if cnf['Surface_spatial_distribution'] == 'Gauss':
                spatial_distribution = self.get_func_gauss_distribution()
            elif cnf['Surface_spatial_distribution'] == 'Cyrcle':
                raise ValueError("todo get_func_cyrcle_distribution")
            else:
                raise ValueError("Unknown Surface_spatial_distribution")

            if cnf['Surface_angular_distribution'] == 'Collimated':
                angular_distribution = self.get_func_collimated_distribution()
            elif cnf['Surface_angular_distribution'] == 'Diffuse':
                raise ValueError("todo Surface_angular_distribution == Diffuse")
            elif cnf['Surface_angular_distribution'] == 'HG':
                raise ValueError("todo Surface_angular_distribution == HG")
            else:
                raise ValueError("Unknown Surface_angular_distribution")
            # определяем функцию генерации
            self.generator = self.get_func_generator(spatial_distribution, angular_distribution)

        # генерация фотонов во всем объеме среды (чтобы адекватно замоделировать например гаусс с перетяжкой внутри)
        elif cnf['mode_generator'] == 'Volume':
            raise ValueError("todo mode_generator == Volume")
        else:
            raise ValueError("Unknown mode_generator")

    def get_func_generator(self, spatial_distribution, angular_distribution):
        """
        Возвращает сгенерированное начальное состояние фотона (фотон может сгенериться и на поверхности)
        :param spatial_distribution: Возвращает случайные координаты фотона
        :param angular_distribution: Возвращает случайные направляющие косинусы фотона
        :return: Фотон
        """
        z0 = self.cnf['Surface_beam_center'][2]
        layer_index = self.get_layer_index_from_z(z0)

        @njit(fastmath=True)
        def buf():
            p = np.zeros(8)

            p[0], p[1], p[3] = spatial_distribution()
            p[3], p[4], p[5] = angular_distribution()

            p[6] = 1
            p[7] = layer_index

            return p

        return buf

    def get_func_gauss_distribution(self):
        """
        Возвращаем функцию, которая возвращает случайные координаты фотона, в соответсвии с гауссовым распределением
        """
        cnf = self.cnf
        x0, y0, z0 = cnf['Surface_beam_center']
        w = cnf['Surface_beam_diameter']

        @njit(fastmath=True)
        def buf():
            r0 = np.random.rand()
            r1 = np.random.rand()
            ph = 2 * np.pi * r0
            radius = w * np.sqrt(-np.log(r1))
            return x0 + np.cos(ph) * radius, y0 + np.sin(ph) * radius, z0

        return buf

    def get_func_collimated_distribution(self):
        """
        Возвращаем функцию, которая возвращает случайные (нет) направляющие косинусы фотона
        в соответсвии с коллимированным излучением
        """

        @njit(fastmath=True)
        def buf():
            return 0, 0, 1

        return buf

    # parse_mode_output
    def parse_mode_save(self):
        """
        Определяем мод связанные с сохранением данных

        save_obj - возвращает объект хранения данных (нужно в нескольких местах создавать, todo отдельный класс?)
        save_data - хрень в которой будут храниться результаты
        save_prog - функция записывающая в save_data в процессе распространения фотона
        save_end - функция записывающая в save_data после того как фотон закончил распространяться
        save_interpreter - интерпретатор сохраненных данных (точно нужно вынести это все в отдельный класс)
        """
        cnf = self.cnf
        if cnf['mode_save'] == 'FIS':
            self.save_obj = self.get_obj_save_data_FIS
            self.save_data = self.save_obj()
            self.save_prog = self.get_func_save_prog_FIS()
            self.save_end = self.get_func_save_end_FIS()
            self.save_interpreter = self.get_func_save_interpreter_FIS()
            # raise ValueError("todo FIS")
        elif cnf['mode_save'] == 'MIS':
            raise ValueError("todo MIS")
        else:
            raise ValueError("Unknown mode_save")

    def get_obj_save_data_FIS(self):
        """
        CT - коллимированное пропускание
        DT - диффузное пропускание
        DR - диффузное отражение
        CR - коллимированное отражение
        A - поглощение
        """
        return np.zeros(5)

    def get_func_save_prog_FIS(self):
        """
        p_gen - состояние фотона при генерации
        p_move - состояние фотона после move
        p_term - состояние фотона после term
        p_turn - состояние фотона после turn
        save_data_old - сохраненные на текущий момент данные
        Ничего не пишем по ходу распространения
        """

        @njit(fastmath=True)
        def buf(p_gen, p_move, p_term, p_turn, save_data_old):
            return save_data_old

        return buf

    def get_func_save_end_FIS(self):
        """
        p_gen - состояние фотона при генерации
        p_move - состояние фотона после move
        p_term - состояние фотона после term
        p_turn - состояние фотона после turn
        save_data_old - сохраненные на текущий момент данные

        Записываем в конце сколько куда вылетело и сколько поглотилось
        """

        collimated_cosinus = self.cnf['FIS_collimated_cosinus']

        @njit(fastmath=True)
        def buf(p_gen, p_move, p_term, p_turn, save_data_old):
            save_data = save_data_old * 1.
            save_data[-1] += p_gen[-2] - p_move[-2]
            if p_move[-1] == -2:
                if p_move[5] > collimated_cosinus:
                    save_data[0] += p_move[-2]
                else:
                    save_data[1] += p_move[-2]
            elif p_move[-1] == -1:
                if p_move[5] > - collimated_cosinus:
                    save_data[2] += p_move[-2]
                else:
                    save_data[3] += p_move[-2]

            return save_data

        return buf

    def get_func_save_interpreter_FIS(self):
        """
        CT - коллимированное пропускание
        DT - диффузное пропускание
        DR - диффузное отражение
        CR - коллимированное отражение
        A - поглощение
        Нормируем и упаковываем в словарь сохраненные данные
        """
        N = self.N
        threads = self.threads

        N_all = N * threads

        save_data = self.save_data

        def buf():
            buf = save_data / N_all
            return {'CT': buf[0], 'DT': buf[1], 'DR': buf[2], 'CR': buf[3], 'A': buf[4], 'check_sum': np.sum(buf)}

        return buf

    # get engin jit function trace/move/turn/term/reflection/R_frenel

    def get_func_R_frenel(self):
        @njit(fastmath=True)
        def R_frenel(th, n1, n2):  # Определения коэффициента Френеля с учетом неполяризованного излучения
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

            if np.sin(th) >= n2 / n1:  # Полное внутренне отражение
                res = 1.
            else:
                res = 0.5 * ((rs(cos_th1, cos_th2, n1, n2)) ** 2 + (rp(cos_th1, cos_th2, n1, n2)) ** 2)

            if res < 1e-6:
                res = 0.
            return res

        return R_frenel

    def get_func_reflection(self):

        n_table = self.n_table
        R_frenel = self.get_func_R_frenel()

        @njit(fastmath=True)
        def reflection(old_p):  # Отражение фотонов от границы раздела сред
            p = old_p * 1.
            layer_number = len(n_table)
            layer_index = int(p[-1])
            n1 = n_table[layer_index]

            # Определяем слой в направлении которого мы вылетам buf и его коэффициент преломления n2
            if (layer_index == layer_number - 1) and (p[5] > 0):
                n2 = 1
                buf = -2
            elif (layer_index != layer_number - 1) and (p[5] > 0):
                n2 = n_table[layer_index + 1]
                buf = layer_index + 1
            elif (layer_index == 0) and (p[5] < 0):
                n2 = 1
                buf = -1
            else:
                n2 = n_table[layer_index - 1]
                buf = layer_index - 1

            if np.random.rand() < R_frenel(np.arccos(p[5]), n1, n2):
                p[5] = -p[5]
            else:
                p[5] = np.sqrt(1 - (n1 / n2) ** 2 * (1 - p[5] ** 2)) * np.sign(p[5])
                p[-1] = buf

            return p

        return reflection

    def get_func_move(self):

        z_end_table = self.z_end_table
        z_start_table = self.z_start_table
        mu_a_table = self.mu_a_table
        mu_s_table = self.mu_s_table

        reflection = self.get_func_reflection()

        @njit(fastmath=True)
        def move(old_p):  # Движение фотона в образце
            # Считаем, что Sum(l*mu_t)=-log(x)=l_rand, и при переходе через границу
            # убавляем l_rand в соответсвии с пройденным путем
            l_rand_tol = 1e-5  # Минимальное значение l_rand
            p = old_p * 1.
            # Создание случайной безразмерной величины длины
            l_rand = -np.log(np.random.random(1))[0]
            while l_rand > l_rand_tol:
                # Определяем текущие границы и свойства
                layer_index = int(p[-1])

                z_start = z_start_table[layer_index]
                z_end = z_end_table[layer_index]

                mu_a = mu_a_table[layer_index]
                mu_s = mu_s_table[layer_index]
                mu_t = mu_a + mu_s

                l_free_path = 1 / mu_t
                l_layer = l_rand * l_free_path

                # Расчитываем на какую величину мы должны переместиться
                new_p_z = p[2] + l_layer * p[5]
                # Проверка выхода за границу
                if z_start < new_p_z < z_end:
                    # Без взаимодействия с границей раздела сред
                    new_p_x = p[0] + l_layer * p[3]
                    new_p_y = p[1] + l_layer * p[4]
                    p[0], p[1], p[2] = new_p_x, new_p_y, new_p_z
                    p[6] = p[6] * np.exp(-mu_a * l_layer)
                    break
                else:
                    # С взаимодействием с границей раздела сред

                    # Расчет на сколько мы переместились до границы
                    if p[5] > 0:
                        l_part = (z_end - p[2]) / p[5]
                    else:
                        l_part = (z_start - p[2]) / p[5]

                    # Уменьшаем случайную l_rand в соответсвии с пройденным путем
                    l_rand = l_rand - mu_t * l_part

                    # Расчет новой координаты на границе
                    new_p_x = p[0] + l_part * p[3]
                    new_p_y = p[1] + l_part * p[4]
                    new_p_z = p[2] + l_part * p[5]

                    p[0], p[1], p[2] = new_p_x, new_p_y, new_p_z
                    p[6] = p[6] * np.exp(-mu_a * l_part)

                    # Проверка отражения
                    p = reflection(p)
                    if p[-1] == -1 or p[-1] == -2:
                        break

            return p

        return move

    def get_func_turn(self):
        g_table = self.g_table

        @njit(fastmath=True)
        def turn(old_p):  # Поворот движения фотонов
            p = old_p * 1.
            g = g_table[int(p[-1])]
            if g != 0.:
                th = np.arccos(0.5 / g * (1 + g ** 2 - ((1 - g ** 2) / (1 - g + 2 * g * np.random.rand())) ** 2))
            else:
                th = np.arccos(1 - 2 * np.random.rand())
            ph = 2 * np.pi * np.random.rand()

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

            p[3], p[4], p[5] = cx, cy, cz

            return p

        return turn

    def get_func_term(self):
        @njit(fastmath=True)
        def term(old_p):  # Если масса слишком маленькая, то убиваем, либо увеличиваем массу в 10 раз
            p = old_p * 1.
            threshold_m = 10 ** -4
            threshold_factor = 10
            if (p[6] < threshold_m) & (p[6] != 0.):
                if np.random.rand() <= 1 / threshold_factor:
                    p[6] = threshold_m * threshold_factor
                else:
                    p[6] = 0.
            return p

        return term

    def get_func_trace(self):
        """
        Возвращаем функцию, которая обеспечивает пролет одного фотона
        Она возвращает локальный save_data которые потом можно проссумировать по всем фотонам
        """
        # Тут и используем все созданные нами numba функции
        gen = self.generator
        move = self.get_func_move()
        term = self.get_func_term()
        turn = self.get_func_turn()
        save_prog = self.save_prog
        save_end = self.save_end

        save_obj_local = self.save_obj()

        @njit(nopython=True)
        def trace():
            save_data = save_obj_local
            p_gen = gen()
            p_turn = p_gen * 1.  # Стартовый p_turn создаем так как цикл именно изменяет p_turn
            for j in range(10 ** 3):
                p_move = move(p_turn)

                # Выход из цикла либо из-за вылета либо из-за уменьшения массы до нуля
                if (p_move[-1] == -1 or p_move[-1] == -2):
                    break
                else:
                    p_term = term(p_move)
                    if p_term[-2] == 0:
                        break
                p_turn = turn(p_term)
                save_data = save_prog(p_gen, p_move, p_term, p_turn, save_data)

            save_data = save_end(p_gen, p_move, p_term, p_turn, save_data)
            return save_data

        return trace

    # run
    def run(self):
        N = self.N
        threads = self.threads

        trace = self.get_func_trace()
        # создаем локальную переменную save_data_run в которую будем сохранять данные одного run
        save_data_run = self.save_obj()

        if (type(threads) is int) and (cpu_count() > threads > 1):
            # Параллелим
            # Один таск - N фотонов, результат одного таска сохраняется в локальную переменную save_data_task
            def task(i):
                save_data_task = self.save_obj()
                for i in range(N):
                    save_data_task += trace()
                return save_data_task

            # Я нихрена не понял как параллелить, но такая конструкция заработала
            with Pool(threads) as pool:
                # threads разделаем task
                save_data_task = pool.map_async(task, range(threads))
                # Сохраняем результаты всех таксков в save_data_run
                for save_data_task in save_data_task.get():  # из того, что вернул таск гетом берем данные?
                    save_data_run += save_data_task
                pool.close()
                pool.join()
        elif threads == 1:
            # Если 1 тред, то можно и не параллелись
            for i in range(N):
                save_data_run += trace()
        else:
            raise ValueError("threads out of (1, cpu_count-1)")

        self.save_data += save_data_run

    # help func
    def get_layer_index_from_z(self, z):
        """
        Возвращает номер слоя
        -1 слой воздуха z <= zmin, -2 слой воздуха z >= zmax
        :param z: Текущая координата фотона
        :return: Текущий слой фотона
        """
        z_start_table = self.z_start_table
        z_end_table = self.z_end_table

        "Генерация на границе с воздухом = генерация в воздухе"
        if z <= z_start_table[0]:
            return -1
        if z >= z_end_table[-1]:
            return -2

        for i, z_start in enumerate(z_start_table):
            if z > z_start:
                i += 1
            else:
                return i - 1
        return i - 1

    # output
    def get_output(self):
        # Хе, ну save_interpreter() это типо внутренняя функция, а гет отпут это для людей,
        # забыл как оно делается нормально
        return self.save_interpreter()
