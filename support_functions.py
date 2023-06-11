from time import time

KW = dict({})


def timeit(method):
    def timed(*args, **kwargs):
        # return method(*args, **kwargs)
        ts = time()
        result = method(*args, **kwargs)
        te = time()
        name = method.__name__.upper()
        if name in KW:
            KW[name]['calls'] += 1
            KW[name]['times'] += int((te - ts) * 1000)
        else:
            KW[name] = dict({
                'calls': 1,
                'times': int((te - ts) * 1000)
            })
        return result
    return timed
