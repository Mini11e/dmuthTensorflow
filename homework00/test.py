import numpy as np
import matplotlib.pyplot as plt

N = 20

def f(x):
    return 0.5*x + 0.25

def L(a, b, x, t):
    sum_loss = 0
    for x_point, t_point in zip(x, t):
        prediction = a*x_point + b
        sum_loss = (prediction - t_point)**2
        
    return sum_loss/N

def main():
    x = np.linspace(-1, 1, 20)
    y = f(x)
    print(y)
    noise = np.random.normal(loc = 0, scale = 0.25, size = 20)
    ynoise = y + noise

    #plt.scatter(x, y, label = "normal")
    #plt.scatter(x, ynoise, label = "noised")
    #plt.legend()
    #plt.show()

    a = np.linspace(-1, 1, 100)
    b = np.linspace(-1, 1, 100)
    A, B = np.meshgrid(a,b)

    print(L(a, b, x, t))

if __name__ == "__main__":
    main()
    