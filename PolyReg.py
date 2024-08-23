import matplotlib.pyplot as plt
import numpy as np
import random

ALPHA = 0.001  # Reduce learning rate further
EPSILON = 0.00001
complexity = 1 + 1  # Polynomial degree + 1


X = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
Y = np.array(sorted([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]), dtype=float)

# Feature Scaling
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

w = np.ones(complexity, dtype=float)
for i in range(complexity):
    # w[i] = random.randint(-300, 300)
    w[i] = 0
b = 0

def main():
    GD()
    costForEach()
    print(f"Final Model: y = {w}x + {b}")
    print(f"Final Cost: {Findcost()}")

    xpoints = X
    ypoints = Y

    plt.plot(xpoints, ypoints, 'o')

    x = np.linspace(0, 1000, 1000)
    x_scaled = (x - X_mean) / X_std  # Apply scaling to the input points for plotting
    print(f"X_Scaled: {X_scaled}")
    print(f"x_Scaled: {x_scaled}")

    y = np.zeros(1000, dtype=float)
    for i in range(complexity):
        y += w[i] * x_scaled ** i

    plt.plot(x, y, '-r', label=f'y={w}x + {b:.2f}')
    plt.legend()
    plt.show()
    
def costForEach():
    M = len(X)
    for i in range(M):
        print(f"{i+1}th point's cost: {(model(X_scaled[i]) - Y[i])**2}")
        print(f"Value of X: {X_scaled[i]}")
        print(f"Predicted_Y: {model(X_scaled[i])}")
        print(f"Real_Y: {Y[i]}")
        

def model(x):
    sum = b
    for i in range(complexity):
        sum += w[i] * x ** i
    return sum

def Findcost():
    sum = 0
    M = len(X_scaled)
    for i in range(M):
        sum += (model(X_scaled[i]) - Y[i]) ** 2
    return sum / (2 * M)

def GD():
    M = len(X_scaled)
    global w, b
    LastCost = Findcost() * 2 + 2 * EPSILON
    while True:
        cost = Findcost()
        if cost > LastCost:
            print("Error: Cost increased.")
            return
        if abs(cost - LastCost) <= EPSILON:
            break
        LastCost = cost
        
        del_by_w = np.zeros(complexity, dtype=float)            
        del_by_b = 0
        
        for i in range(M):
            error = model(X_scaled[i]) - Y[i]
            del_by_b += error
            for j in range(complexity):
                del_by_w[j] += error * X_scaled[i] ** j
        
        del_by_w /= M
        # del_by_b /= M
        
        w -= ALPHA * del_by_w
        # b -= ALPHA * del_by_b
        
        print(f"Cost: {Findcost()}")
        print(f"w: {w}")
        print(f"b: {b}")

if __name__ == "__main__":
    main()
