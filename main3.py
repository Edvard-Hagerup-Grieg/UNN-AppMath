from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


def example_1(t, x):
    return [-x[0]]

def example_2(t, x):
    return [x[0]]

def example_3(t, x):
    return [x[1], -x[0]]

def example_4(t, x):
    a = 0.2
    b = a
    r = 5.7

    return [-x[1] - x[2],
            x[0] + a * x[1],
            b + (x[0] - r) * x[2]]


def exact_1(x0, t):
    x = x0.copy()
    for j in range(len(x0)):
        x[j] = x[j] * np.exp(-t)
    return x

def exact_2(x0, t):
    x = x0.copy()
    for j in range(len(x0)):
        x[j] = x[j] * np.exp(t)
    return x

def exact_3(x0, t):
    x = x0.copy()
    for j in range(len(x0)):
        x[j] = np.sin(t)
    return x


def Euler_step(tn, xn, f, h):
    xn_ = xn.copy()
    for j in range(len(xn)):
        xn_[j] = xn[j] + h * f(tn, xn)[j]
    return xn_

def RK4_step(tn, xn, f, h):
    xn_ = xn.copy()

    k1 = f(tn, xn)
    k2 = f(tn + h / 2, [xn_[i] + k1[i] * h / 2 for i in range(len(xn))])
    k3 = f(tn + h / 2, [xn_[i] + k2[i] * h / 2 for i in range(len(xn))])
    k4 = f(tn + h, [xn_[i] + k3[i] * h for i in range(len(xn))])

    for j in range(len(xn)):
        xn_[j] = xn[j] + h*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])/6

    return xn_


def ode(f, method, t0, x0, n, h, exact=None):
    xn = x0.copy()
    tn = t0

    e = []
    x = [x0]
    t = [tn]
    if exact is not None: e = [exact(x0, t0)]
    for i in range(n):
        if       method == 'Euler': xn = Euler_step(tn, xn, f, h)
        elif     method == 'RK4':   xn = RK4_step(tn, xn, f, h)
        else:
            print("Invalid argument: ", method)
            break
        tn = tn + h

        x.append(xn)
        t.append(tn)
        if exact is not None: e.append(exact(x0, tn))

    return np.array(t), np.array(x), np.array(e)

def plot(t, x1, method1, x2=None, method2=None, exact=None, system=None):
    ax = plt.gca()

    legend = [method1]
    if method2 is not None: legend = legend + [method2]
    if exact is not None: legend = legend + ['exact']

    ax.plot(t, x1)
    if x2 is not None: ax.plot(t, x2)
    if exact is not None: ax.plot(t, exact, color='black', alpha=0.3)

    plt.grid(True)
    plt.title(system)
    plt.legend(legend)
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.show()

def plot_err(t, x1=None, exact1=None, x2=None, exact2=None, system=None):
    err1 = []
    err2 = []
    legend = []
    for i in range(t.shape[0]):
        if x1 is not None:
            err1.append(np.abs(x1[i] - exact1[i]))
            legend.append('Euler method error')
        if x2 is not None:
            err2.append(np.abs(x2[i] - exact2[i]))
            legend.append('RK4 method error')

    ax = plt.gca()
    if x1 is not None: ax.plot(t, err1)
    if x2 is not None: ax.plot(t, err2)

    plt.grid(True)
    plt.title(system)
    plt.legend(legend)
    plt.yscale('log')
    plt.ylabel("log(Euler_err), log(RK4_err)")
    plt.xlabel("t")
    plt.show()


if __name__ == "__main__":
    t0 = 0.0
    n = 1000
    h = 0.01

    # system 1
    x0 = [100]

    t1, x1, exact1 = ode(example_1, 'Euler', t0, x0, n, h, exact_1)
    _, x2, exact2 = ode(example_1, 'RK4', t0, x0, n, h, exact_1)

    plot(t1, x1, 'Euler', x2, 'RK4', exact1, 'system 1')
    plot_err(t1, x1, exact1, x2, exact2, 'system 1')


    # system 2
    x0 = [100]

    t1, x1, exact1 = ode(example_2, 'Euler', t0, x0, n, h, exact_2)
    _, x2, exact2 = ode(example_2, 'RK4', t0, x0, n, h, exact_2)

    plot(t1, x1, 'Euler', x2, 'RK4', exact1, 'system 2')
    plot_err(t1, x1, exact1, x2, exact2, 'system 2')


    # system 3
    x0 = [0.0, 1.0]
    n = 1000
    h = 0.1

    t1, x1, exact1 = ode(example_3, 'Euler', t0, x0, n, h, exact_3)
    _, x2, exact2 = ode(example_3, 'RK4', t0, x0, n, h, exact_3)

    plot(t1, x1[:,0], 'Euler', x2[:,0], 'RK4', exact1[:,0], 'system 3')
    plot_err(t1, x1[:,0], exact1[:,0], x2[:,0], exact2[:,0], 'system 3')


    # system 4
    x0 = [0.0, 1.0, 1.0]

    n = 10000
    h = 0.01

    _, x1E, _ = ode(example_4, 'Euler', t0, x0, 2*n-1, h/2)
    t2E, x2E, _ = ode(example_4, 'Euler', t0, x0, n-1, h)

    _, x1R, _ = ode(example_4, 'RK4', t0, x0, 2*n-1, h/2)
    t2R, x2R, _ = ode(example_4, 'RK4', t0, x0, n-1, h)

    plot(x1R[:,0], x1R[:,1], 'RK4', system='system 4')
    plot(x1R[:,0], x1R[:,2], 'RK4', system='system 4')
    plot_err(t=t2E, x1=x1E[[i*2 for i in range(x2E.shape[0])],0], exact1=x2E[:,0],
                    x2=x1R[[i*2 for i in range(x2R.shape[0])],0], exact2=x2R[:,0], system='system 4')

    ax = plt.axes(projection='3d')
    ax.plot3D(x1R[:,1], x1R[:,0], x1R[:,2])
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.show()