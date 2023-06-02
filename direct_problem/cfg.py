from dataclasses import dataclass

from multiprocess import cpu_count


@dataclass
class DirectProblemCfg:
    N: int = None
    threads: int = None

    def validate(self) -> None:
        if not self.N:
            raise ValueError('N = None in DirectProblemCfg')
        if not isinstance(self.N, int):
            raise ValueError('N is not of type int in DirectProblemCfg')
        if self.N < 1:
            raise ValueError(f'N = {self.N} out of range [1, +inf) in DirectProblemCfg')

        if not self.threads:
            raise ValueError('threads = None in DirectProblemCfg')
        if not isinstance(self.threads, int):
            raise ValueError('threads is not of type int in DirectProblemCfg')
        if not (1 <= self.threads <= cpu_count()):
            raise ValueError(f'threads = {self.threads} out of range [1, {cpu_count()}] in DirectProblemCfg')
