# import tensorflow\
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.0001
EPSILON = 0.001
LastCost = 0

X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 2]

w = 0
b = 0
    
def main():
    GD()
    print(f"{w}x+{b}")
    print(f"Cost: {Findcost()}")
    
    xpoints = np.array(X)
    ypoints = np.array(Y)

    plt.plot(xpoints, ypoints, 'o')
    x = np.linspace(-5,15,10)
    y = w*x+b
    plt.plot(x, y, '-r', label='y=2x+1')
    
    plt.plot(x, y)
    plt.show()

def model(x):
    return w*x + b

def Findcost():
    sum = 0
    M = len(X)
    for i in range(M):
        sum = sum +( model(X[i]) - Y[i] ) ** 2
    return (1/(2 * M)) * sum

def GD():
    M = len(X)
    global w, b
    LastCost = 0
    while True:
        cost = Findcost()
        if (cost - LastCost)**2 <= EPSILON**2:
            return
        LastCost = cost
        del_by_w = 0
        del_by_b = 0
        for i in range(M):
            del_by_w += (model(X[i]) - Y[i])*X[i]
            del_by_b += model(X[i]) - Y[i]
        del_by_b= del_by_b/ M
        del_by_w= del_by_w/ M
        w -= ALPHA * del_by_w
        b -= ALPHA * del_by_b
        print(f"cost: {Findcost()}")
        print(f"w: {w}")
        print(f"b: {b}")

if __name__ == "__main__":
    main()
