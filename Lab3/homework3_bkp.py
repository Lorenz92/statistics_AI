# Import the libraries (as usual)
import numpy as np
import matplotlib.pyplot as plt

# Import the backtracking file from the folder utils to run the backtracking.
from utils import backtracking


# The gradient_descent implementation.
def gradient_descent(f, grad_f, x0, kmax, tolf, tolx):
    # Initialization
    k = ... #0
    
    x = ... #np.zeros((kmax, len(x0))
    f_val = ... #np.zeros(kmax)
    grads = ... #np.zeros((kmax, len(x0))
    err = ... #np.zeros(kmax)
    
    # Assign the values for the first iteration
    f_val[k] = ...
    grads[k, :] = ...
    err[k] = ...
    
    # Choose step size
    alpha = ...#.1
    
    # Handle the condition for the first iteration
    ...
    
    # Start the iterations
    while ...:
        # Update the value of x
        x[k+1, :] = ...
        
        # Update the step size alpha
        alpha = ... #alpha for the alpha constant case
        
        # Update the values the the actual iteration
        k = k+1
        f_val[k] = ...
        grads[k, :] = ...
        err[k] = ...
    
    # Truncate the vectors that are (eventually) too long
    f_val = ...
    grads = ...
    err = ...
    
    return x, k, f_val, grads, err

#%%
"""
Exercise 1:
"""
def f(x):
    ...


def grad_f(x):
    ...


n = ...

x0 = ..
kmax = ...
tolf = ...
tolx = ...

x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, kmax, tolf, tolx)














