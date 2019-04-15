import matplotlib.pyplot as plt
import numpy as np
import math


def f(x, r):
    return r*x*(1 - x)

def df(x, r):
    return r - 2*r*x


if __name__ == "__main__":

    # LAMEREY DIAGRAM & SYSTEM EVALUTION
    RUN = True
    if RUN:
        x0 = 0.4
        r = 3.46

        xn = [x0]
        x = [x0]
        y = [0]
        for i in range(500):
            x1 = f(x0, r)

            x.append(x0)
            x.append(x1)

            y.append(x1)
            y.append(x1)

            xn.append(x1)

            x0 = x1

        plt.figure(figsize=(10,4))

        plt.subplot(1, 2, 1)
        plt.plot(range(100), xn[:100], color='black', linewidth=0.7)

        plt.title('SYSTEM EVALUTION')
        plt.xlabel('n')
        plt.ylabel('x(n)')

        plt.subplot(1,2,2)
        plt.plot(x,y,alpha=0.7,color='red', linewidth =0.7, linestyle='--', label='')
        plt.plot(np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01), linewidth =0.4, color = 'black',label='x(n+1)=x(n)')
        plt.plot(np.arange(0.0, 1.0, 0.01), [f(xn, r) for xn in np.arange(0.0, 1.0, 0.01)], linewidth =1, color = 'black', label='x(n+1)')

        plt.title('LAMEREY DIAGRAM')
        plt.xlabel('x(n)')
        plt.ylabel('x(n+1)')

        plt.show()

    # BIFURCATION DIAGRAM & LYAPUNOV EXPONENT
    RUN = True
    if RUN:
        x0 = 0.1

        X = []
        L = []
        for r in np.arange(0.8, 4.00, 0.001):
            x = x0
            ln = math.log(abs(df(x0, r)))
            xn = []
            for i in range(1000):
                x = f(x, r)
                xn.append(x)
                ln += math.log(abs(df(x, r)))
            X.append(xn[-200:])
            L.append(ln / 1000)

        X = np.array(X)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        for i in range(X.shape[1]):
            plt.scatter(np.arange(0.8, 4.00, 0.001), X[:,i], s=0.1, c='black')

        plt.title('BIFURCATION DIAGRAM')
        plt.xlabel('r')
        plt.ylabel('x*')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0.8, 4.00, 0.001), L, color='black')
        plt.plot(np.arange(0.8, 4.00, 0.001), [0]* 3200, color='red')

        plt.title('LYAPUNOV EXPONENT')
        plt.xlabel('r')
        plt.ylabel('L')

        plt.show()