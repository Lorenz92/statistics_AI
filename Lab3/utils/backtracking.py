import numpy as np

def backtracking(f, grad_f, x):
    """
    This function is a simple implementation of the backtracking algorithm for
    the GD (Gradient Descent) method.
    
    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    """
    alpha = 1
    c = 0.8
    tau = 0.25
    
    if isinstance(grad_f(x), np.ndarray):
        while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.linalg.norm(grad_f(x), 2) ** 2:
            alpha = tau * alpha
    else:
        while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.absolute(grad_f(x)):
            alpha = tau * alpha

    return alpha