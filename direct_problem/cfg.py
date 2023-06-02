from dataclasses import dataclass
from multiprocess import cpu_count


@dataclass
class DirectProblemCfg:
    N: int = 1000
    threads: int = 1

    def validate(self):
        N = self.N
        threads = self.threads

        if N not in range(1, 10000000):
            # i can't get int(inf)
            raise ValueError(f'N = {N} out of range')

        if threads not in range(1, cpu_count()):
            raise ValueError(f'threads = {threads} out of range')
