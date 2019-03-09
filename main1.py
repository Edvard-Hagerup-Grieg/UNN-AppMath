import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**(n+1) + x - alpha

def df(x):
    return (n + 1) * x**n + 1

def f2(x):
    return x**2 - alpha * x + 1

def df2(x):
    return 2*x - alpha


def bisection_method(a, b, delta):
    left = a
    right = b
    middle = (left + right) / 2.

    middle_arr = [middle]
    difference_arr = [abs(right-left)]
    while abs(right - left) > delta:
        if f(left)*f(middle) < 0:
            right = middle
        else: left = middle
        middle = (left + right) / 2.
        middle_arr.append(middle)
        difference_arr.append(abs(right-left))

    return middle, middle_arr, difference_arr

def newtons_method(x0, delta):
    xn = x0 - (f(x0) / df(x0))

    xn_arr = [xn]
    difference_arr = [abs(xn-x0)]
    while abs(xn - x0) > delta:
        x0 = xn
        xn = x0 - (f(x0) / df(x0))
        xn_arr.append(xn)
        difference_arr.append(abs(xn-x0))

    return xn, xn_arr, difference_arr

def newtons_method2(x0, delta):
    xn = x0 - (f(x0) / df(x0))

    xn_arr = [xn]
    difference_arr = [abs(xn-x0)]
    while abs(xn - x0) > delta:
        x0 = xn
        xn = x0 - (f2(x0) / df2(x0))
        xn_arr.append(xn)
        difference_arr.append(abs(xn-x0))

    return xn, xn_arr, difference_arr


if __name__ == "__main__":

# PART 1
    trajectories = []
    for n in np.arange(2,7,2):
        tr = []
        for alpha in np.arange(0,50.1,0.1):
            x,_,_ = newtons_method(1.5, 0.0001)
            tr.append(x)
        trajectories.append(tr)

    fig1 = plt.figure()
    ax1 = plt.gca()

    for tr in trajectories:
        ax1.plot(np.arange(0,50.1,0.1), tr)

    plt.grid(True)
    plt.legend(['n = 2', 'n = 4', 'n = 6'])
    plt.ylabel("x(alpha)")
    plt.xlabel("alpha")
    plt.show()

# PART 2
    n = 4
    alpha = 10

    _, answers1, d1 = bisection_method(0, 5.8, 0.00001)
    _, answers2, d2 = newtons_method(0.6, 0.00001)
    answer = [1.53301] * max(len(answers1), len(answers2))

    fig2 = plt.figure()
    ax2 = plt.gca()

    ax2.plot(answers1, color = 'red', alpha=0.7)
    ax2.plot(answers2, color = 'blue', alpha=0.7)
    ax2.plot(answer, color = 'black', alpha=0.5)

    plt.grid(True)
    plt.legend(['bisection', 'newton', 'exact'])
    plt.ylabel("x(k), m(k)")
    plt.xlabel("k")

    fig3 = plt.figure()
    ax3 = plt.gca()

    ax3.plot(np.log(d1), color='red')
    ax3.plot(np.log(d2), color='blue')

    plt.grid(True)
    plt.legend(['log|a[k] - b[k]|', 'log|x[k+1] - x[k]|'])
    plt.ylabel("log|a[k] - b[k], log|x[k+1] - x[k]|")
    plt.xlabel("k")

    plt.show()

# PART 3
    x1_arr = []
    x2_arr = []
    x3_arr = []
    for alpha in np.arange(0.1, 3.501, 0.001):
        x1, _, _ = newtons_method(np.random.random() * 100, 0.0001)
        if alpha >= 2:
            x2, _, _ = newtons_method2(np.random.random() * 100, 0.0001)
            x3, _, _ = newtons_method2(np.random.random() * 100 - 50, 0.0001)

            iter = 0
            while abs(x2 - x3) < 0.001:
                x3, _, _ = newtons_method2(np.random.random() * 100 - 50, 0.0001)
                iter = iter + 1
                if iter > 100: break

            x3_arr.append(x3)
            x2_arr.append(x2)
        x1_arr.append(x1)

    fig4 = plt.figure()
    ax4 = plt.gca()

    ax4.scatter(np.arange(0.1, 3.501, 0.001), x1_arr, color='black', alpha = 0.3, s=1)
    ax4.scatter(np.arange(2.000, 3.501, 0.001), x2_arr, color='blue', alpha = 0.3, s=1)
    ax4.scatter(np.arange(2.000, 3.501, 0.001), x3_arr, color='red', alpha = 0.3, s=1)

    plt.grid(True)
    plt.legend(['x1', 'x2', 'x3'])
    plt.ylabel("x(alpha)")
    plt.xlabel("alpha")

    plt.show()