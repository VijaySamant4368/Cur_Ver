import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    degree = 10     #The degree of the curve. The highest '?' in (c1*x^?+...+cn*x^?) where c is some constant
    data_degree = 2
    slope = np.ones(data_degree + 1)
    for s in range(data_degree +1):
        slope[s] = random.randint(-10, 10)
    X, Y = collectData(observations = 10, degree = data_degree, randomness=1, slope=slope )
    X_scaled, max, min= scaleDown(X)
    W_ = gradientDescent(X_scaled, Y, degree, alpha= 10, epsilon=0.000001, var_grad=False)
    Y_Hat = f(W_, X_scaled, degree=degree)
    
    printData(X_scaled, Y, Y_Hat)
    printModel(X_scaled)
    
    plt.plot(X_scaled, Y, 'o')
    x = np.linspace(X_scaled.min(), X_scaled.max(), X_scaled.size * 10)
    y = f(W_, x, degree=degree)
    plt.plot(x, y, '-r', label=f'y={W_}x')
    plt.show()
    
    
def collectData(observations:int = 20, randomness:float = 0.1, degree:int = 1, slope:np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns two numpy arrays, the input and output.

    Args:
        observations (int, optional): Total number of input-output pairs. Defaults to 20.
        randomness (float, optional): The amount of randomness in the dataset. Defaults to 0.1.
        degree (int, optional): The degree of the curve, skeleton of the datapoints. Defaults to 1.
        slope (np.ndarray, optional): Array slope of each power of x. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Input array and corresponding output array
    """
    if type(slope) == 'NoneType':
        slope = np.ones(degree + 1)
    X = np.arange(observations)
    Y = curve(X, degree, slope)
    Y = randomize(Y, randomness)
    return X, Y

def printData(X, Y, Y_Hat):
    print(f"Input values: {X}")
    print(f"Predicted outputs: {Y_Hat}")
    print(f"Real outputs: {Y}")
    
def scaleDown(arr: np.ndarray)  -> tuple[np.ndarray, float, float]:
    """Scales data down of an array by Linear Scaling

    Args:
        arr (np.ndarray): The array to be scaled down

    Returns:
        tuple[np.ndarray, float, float]: The scaled down array, the max value of the array, the min value of the array
    """
    min = arr.min()
    max = arr.max()
    return (arr-min)/(max-min), max, min
    
def curve(X: np.ndarray, degree:int, slope:np.ndarray)  -> np.ndarray:
    """Returns numpy array of output points

    Args:
        X (np.ndarray): The input array
        degree (int): The degree of function
        slope (np.ndarray): The slope of variables

    Returns:
        np.ndarray: The output array
    """
    n = X.size
    Y = np.zeros(n)
    for i in range(n):
        x = X[i]
        y = 0
        for j in range(degree + 1):
            y += slope[j] * (x**j)
            
        Y[i] = y
    return Y

def randomize(arr: np.ndarray, randomness: float, sig_fig:int = 2)   -> np.ndarray:
    """Returns randomized array

    Args:
        arr (np.ndarray): Array to be randomized
        randomness (float): Degree of randomisation
        sig_fig (int, optional): Number of digits after decimal points. Defaults to 2.

    Returns:
        np.ndarray: Randomized array
    """
    n = arr.size
    random_range = (arr.sum())/(2*arr.size) * randomness
    for i in range(n):
        x = arr[i]
        upper = x + random_range
        lower = x - random_range
        arr[i] = round(random.uniform(lower, upper), sig_fig)
    return arr

def gradientDescent(X_: np.ndarray, Y_: np.ndarray, degree: int = 1, alpha:float = .5, epsilon:float = 0.001, var_grad: bool = False)    -> np.ndarray:
    """Performs Gradient Descent and returns the array of weights

    Args:
        X_ (np.ndarray): The input array
        Y_ (np.ndarray): The output array
        degree (int, optional): The degree of the curve to fit. Defaults to 1.
        alpha (float, optional): Learning Rate. Defaults to 0.1.
        epsilon (float, optional): The highest difference between cost function of successive itteration to declare as minima. Defaults to 0.000001.

    Returns:
        np.ndarray: Numpy array of weights
    """
    folder = "var_grad"
    N = degree +1
    M = X_.size
    W_ = np.ones(N)
    lastCost = (sqErrorCost(W_, X_, Y_, degree))
    
    iterations = 0
    save_gap = 1
    save_counts = 0
    
    while True:
        cost = (sqErrorCost(W_, X_, Y_, degree))
        if iterations%save_gap == 0 and lastCost >= cost:
            plotGraph(X_, W_, Y_, degree, filename=f"{folder}/iteration {iterations}.png", caption = f"iteration {iterations}; Cost {round(cost, 3)}", display=False)
            save_counts += 1
            if save_counts % 5 == 0:
                save_gap *=10
        if (iterations > 0):
            if (lastCost < cost):
                print(f"Cost exceeded from {lastCost} to {cost}")
                alpha /= 10
                plotGraph(X_, W_, Y_, degree, 
                          filename=f"{folder}/Iteration {iterations} Cost increased.png", 
                          caption=f"Cost exceeded, alpha was {alpha}\n Cost {round(cost, 3)}; Last {round(lastCost, 3)}") 
            if (0< lastCost - cost <= epsilon):
                print(f"Probably reached minima, cost decreased by only {lastCost - cost}")
                break
            if var_grad:
                alpha = 1/iterations**3
        del_W = np.zeros(N)
        for j in range(N):
            sum = 0
            for i in range(M):
                sum += ( f_i(W_, X_[i], degree) - Y_[i] ) * X_[i] ** j
            del_W[j] = sum/M
        W_ = W_ - alpha * del_W
        print(f"Weights: {W_}")
        print(f"Cost: {cost}")
        plt.legend('',frameon=False)
        lastCost = cost
        iterations += 1
    
    plotGraph(X_, W_, Y_, degree, filename=f"{folder}/Last Iteration {iterations}.png", caption = f"Last Iteration {iterations}; Cost {round(cost, 3)}", display=False)
    return W_
    
def sqErrorCost(W_:np.ndarray, X_:np.ndarray, Y_:np.ndarray, degree:int)    -> float:
    """Returns squared error for a given array of weights

    Args:
        W_ (np.ndarray): The array of weights
        X_ (np.ndarray): The array of input values
        Y_ (np.ndarray): The array of target values
        degree (int): The degree of the polynomial curve

    Returns:
        float: Cost function calculated
    """
    total_cost = 0
    M = X_.size
    N = degree + 1
    for i in range(M):
        error = 0
        for j in range(N):
            error +=( W_[j] * X_[i]**j)
        total_cost += (f_i(W_, X_[i], degree) - Y_[i])**2
    cost = total_cost/(2*M) 
    return cost

def f_i(W_: np.ndarray, x: float, degree:int)   -> float:
    """Returns the predicted value for a specific example

    Args:
        W_ (np.ndarray): The vector of weigths
        x (float): The input value of that particular example
        degree (int): The degree of the model

    Returns:
        float: The predicted value for x
    """
    sum = 0
    for j in range(degree + 1):
        sum += W_[j] * x**j
    return sum

def f(W_: np.ndarray, X_: np.ndarray, degree: int)  -> np.ndarray:
    """Returns the predicted values for all examples

    Args:
        W_ (np.ndarray): The array of weights
        X_ (np.ndarray): The array of input values
        degree (int): The degree of model

    Returns:
        np.ndarray: The array of predicted values
    """
    M = X_.size
    Y_Hat = np.zeros(M)
    for j in range(degree+1):
        Y_Hat +=  W_[j] * X_**j
    return Y_Hat

def printModel(W_: np.ndarray)  ->None:
    """Prints the model as string

    Args:
        W_ (np.ndarray): The array of weights
    """
    N = W_.size
    for i in range(N):
        print(f"{W_[i]}*x^{i}", end=" + ")
        
def plotGraph(X: np.ndarray, W_: np.ndarray, Y:np.ndarray, degree:int, sig_fig:int = 3, filename:str = False, display = True, renew = True, caption = ""):
    """Plots and saves/displays graph opptionally

    Args:
        X (np.ndarray): The values in the X-axis
        W_ (np.ndarray): The values of the powers of x
        Y (np.ndarray): The values on the Y axis
        degree (int): The degree of the polynomial to plot
        sig_fig (int, optional): Number of digits after decimal point, to be shown in the graph. Defaults to 3.
        filename (str, optional): The name of the graph to be saved. Defaults to False.
        display (bool, optional): Whether to display the graph on screen. Defaults to True.
        renew (bool, optional): Whether to clean the graph after plotting the curve. Defaults to True.
        caption (str, optional): Caption to be shown in the Graph. Defaults to "".
    """
    plt.plot(X, Y, 'o')
    x = np.linspace(X.min(), X.max(), X.size * 10)
    y = f(W_, x, degree=degree)
    label = ""
    for i in range(W_.size):
        if i == 0:
            label += f"{round(W_[i], sig_fig)}"
            continue
        if i == 1:
            label += f" {round(W_[i],sig_fig)}x"
            continue
        if i % 5 == 4:
            label += "\n"
        label += f" {round(W_[i],sig_fig)}x^{i}"
        
    label +=f"\n{caption}"
    
    plt.plot(x, y, '-r', label=label)
    plt.legend()
    if filename:
        plt.savefig(filename)
    if display:
        plt.show()
    if renew:
        plt.cla()

if __name__ == "__main__":
    main()