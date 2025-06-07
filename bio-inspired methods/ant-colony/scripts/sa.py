import numpy as np


def sa(t_max=1, t_min=1e-9, e_th=1e-5, alpha=0.84):
    np.random.seed(1)
    x, count = _anneal(t_max, t_min, e_th, alpha)


def f(x):
    return x[0]**2 + x[1]**2


def gen():
    return np.random.uniform(-1, size=2)


def _anneal(t, t_min, e_th, alpha):
    x = gen()
    e = f(x)
    n = 0

    while t > t_min and e > e_th:
        dx = gen()
        de = f(dx)

        delta = de-e

        if delta < 0:
            x = dx
            e = de
        else:
            p = np.exp(-delta/t)
            r = np.random.rand()

            if r < p:
                x = dx
                e = de

        t *= alpha
        n += 1

    return x, n


if __name__ == '__main__':
    sa()
