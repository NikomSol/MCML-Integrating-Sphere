from enum import Enum, auto
from dataclasses import dataclass
import numpy as np

class SpaceDimension(Enum):
    Infinity = auto()
    HalfInfinity = auto()
    Multilayer = auto()
    Mesh = auto()
    Function = auto()

class Study(Enum):
    Stationary = auto()
    TimeDependent = auto()
    FrequenceDomain = auto()

@dataclass
class DirectProblemCfg:
    space_dimension: SpaceDimension = None
    study: Study = None
    N: int = None
    threads: int = None

# Sample
class Domain:
    pass

class Layer(Domain):
    pass

class FunctionArea(Domain):
    pass

class Sample():
    def __init__(self, domains: list[Domain]):
        self.domains = domains

class Mesh(Sample):
    def __init__(self):
        pass

# Engine functions
# var1: 1 класс - много реализаций (без наследования). При инициализации по конфигам определяется реализация.
# var2: 1 класс - 1 реализация. В классе конифга есть функция "дай мне реализацию". 
# Классы реализации сами конфиги не кушают, только то, что им нужно. (Я скорее за такой, удобнее добавлять реализации.)
# Move, Turn, Term, End - сейчас под первый вариант накидано, Generator, Save под второй.

class MoveCfg():
    pass

class Move():
    def __init__(move_cfg: MoveCfg, sample: Sample):
        pass
    def get_function(self):
        pass

class TurnCfg():
    pass

class Turn():
    def __init__(turn_cfg: TurnCfg, sample: Sample):
        pass
    def get_function(self):
        pass

class TermCfg:
    pass

class Term():
    def __init__(term_cfg: TermCfg, sample: Sample):
        pass
    def get_function(self):
        pass

class EndCfg:
    pass

class End():
    def __init__(end_cfg: EndCfg, sample: Sample):
        pass
    def get_function(self):
        pass


@dataclass
class GeneratorCfg:
    def get_generator(sample):
        pass

class Generator():
    def __init__(sample: Sample):
        pass
    def get_function(self):
        pass

@dataclass
class SaveCfg:
    def get_saver(sample):
        pass

class Saver():
    def __init__(sample: Sample):
        pass
    def get_function(self):
        pass
    def get_function_get_storage(self):
        pass


class DirectProblem():
    def __init__(self, cfg: DirectProblemCfg, generator: Generator, move: Move, turn: Turn, term: Term, saver: Saver, end: End):
        self.cfg = cfg
        self.generator = generator
        self.move = move
        self.turn = turn
        self.term = term
        self.saver = saver
        self.end = end

    def solve(self):
        N = self.cfg.N
        generator = self.generator.function()
        move = self.move.get_function()
        turn = self.turn.get_function()
        term = self.term.get_function()
        saver = self.saver.get_function()
        end = self.end.get_function()

        get_storage = self.save.get_function_get_storage()

        storage = get_storage()

        for _ in range(N):
            p = generator(p)
            while True:
                p = move(p)
                p = term(p)
                storage = saver(storage, p)
                if end(p):
                    break
                p = turn(p)

        return storage
