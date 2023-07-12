 # Import the libraries (as usual)
import numpy as np
import matplotlib.pyplot as plt

# Import the backtracking file from the folder utils to run the backtracking.
from utils.backtracking import backtracking


# The gradient_descent implementation.
def gradient_descent(f, grad_f, x0, kmax, tolf, tolx, alpha, bk):
    # Initialization
    k = 0
    
    x = np.zeros((kmax+1, x0.shape[0]))
    f_val = np.zeros(kmax+1)
    grads = np.zeros((kmax+1, x0.shape[0]))
    err = np.zeros(kmax+1)
    
    # Assign the values for the first iteration
    x[k, :] = x0
    f_val[k] = f(x[k, :])
    grads[k, :] = grad_f(x[k, :])
    err[k] = np.linalg.norm(grads[k, :], 2)
    
    # Choose step size
    if bk:
        alpha = backtracking(f, grad_f, x[k,:])
    else:
        alpha=alpha

    # Handle the condition for the first iteration
    x[-1, :] = 1
    
    # Start the iterations
    while (k < kmax) and (np.linalg.norm(grads[k, :]) > tolf*np.linalg.norm(grads[0, :])) and (np.linalg.norm(x[k, :]-x[k-1, :], 2) > tolx):
        # Update the value of x
        x[k+1, :] = x[k, :] - alpha * grads[k, :]
        
        # Update the step size alpha
        if bk:
            alpha = backtracking(f, grad_f, x[k,:])
        else:
            alpha=alpha

        # Update the values the the actual iteration
        k = k+1
        f_val[k] = f(x[k, :])
        grads[k, :] = grad_f(x[k, :])
        err[k] = np.linalg.norm(grads[k, :], 2)
    
    # Truncate the vectors that are (eventually) too long
    x = x[:k+1]
    f_val = f_val[:k+1]
    grads = grads[:k+1, :]
    err = err[:k+1]
    
    return x, k, f_val, grads, err














