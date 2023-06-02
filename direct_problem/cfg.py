from dataclasses import dataclass

from multiprocess import cpu_count


@dataclass
class DirectProblemCfg:
    N: int = None
    threads: int = None

    def validate(self) -> None:

        if not self.N:
            raise ValueError(f'N = None in DirectProblemCfg')
        if self.N < 1:
            raise ValueError(f'N = {self.N} out of range [1, +inf)')

        if not self.threads:
            raise ValueError(f'threads = None in DirectProblemCfg')
        if not (1 <= self.threads <= cpu_count()):
            raise ValueError(f'threads = {self.threads} out of range [1, {cpu_count()}]')


if __name__ == '__main__':
    cfg = DirectProblemCfg()
    try:
        cfg.validate()
        raise RuntimeError('default empty cfg should fail')
    except ValueError:
        print('Everything works fine')
