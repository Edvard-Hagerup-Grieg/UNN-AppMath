import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**(n+1) + x - alpha

def df(x):
    return (n + 1) * x**n + 1


def newtons_method(x0, delta):
    xn = x0 - (f(x0) / df(x0))

    xn_arr = [xn]
    difference_arr = [abs(xn-x0)]
    while abs(xn - x0) > delta:
        x0 = xn
        xn = x0 - (f(x0) / df(x0))
        xn_arr.append(xn)
        difference_arr.append(abs(xn-x0))

    return xn


if __name__ == "__main__":

    n_arr = [2, 4, 6]
    srart = [2, 1.5, 1.1]

    total_T = []
    for i in range(3):
        n = n_arr[i]
        T = []
        for alpha in np.arange(srart[i], 4.01, 0.01):
            x = newtons_method(0.0, 0.0001)
            beta = (alpha * n * x ** (n + 1)) / (1 + x ** n) ** 2
            lyambda = np.sqrt(beta ** 2 - 1)
            tau = np.arccos(-1 / beta) / lyambda
            T.append(tau)

        total_T.append(T)


    fig = plt.figure()
    ax = plt.gca()

    for i in range(3):
        ax.plot(np.arange(srart[i], 4.01, 0.01), total_T[i])

    plt.grid(True)
    plt.xlim(0.0, 4.0)
    plt.ylim(0.0, 2.6)
    plt.legend(['n = 2', 'n = 4', 'n = 6'])
    plt.ylabel("t(alpha)")
    plt.xlabel("alpha")
    plt.show()