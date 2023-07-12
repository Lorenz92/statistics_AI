import numpy as np
import matplotlib.pyplot as plt

from utils import backtracking


def gradient_descent(f, grad_f, x0, kmax, tolf, tolx):
    k = 0
    
    x = np.zeros((kmax+1, int(x0.shape[0])))
    f_val = np.zeros((kmax+1, ))
    grads = np.zeros((kmax+1, int(x0.shape[0])))
    err = np.zeros((kmax+1, ))
    
    f_val[k] = f(x[k, :])
    grads[k, :] = grad_f(x[k, :])
    err[k] = np.linalg.norm(grads[k], 2)
    
    # Choose step size
    alpha = 0.1 # backtracking.backtracking(f, grad_f, x[k, :])
    
    x[-1, :] = 1 # Required to run the first while iteration.
    while (k < kmax) and (err[k] > tolf * err[0]) and (np.linalg.norm(x[k, :] - x[k-1, :], 2) > tolx):
        x[k+1, :] = x[k, :] - alpha * grad_f(x[k, :])
        alpha = 0.1 # backtracking.backtracking(f, grad_f, x[k, :])
        
        k = k+1
        f_val[k] = f(x[k, :])
        grads[k, :] = grad_f(x[k, :])
        err[k] = np.linalg.norm(grads[k, :], 2)
    
    f_val = f_val[:k+1]
    grads = grads[:k+1, :]
    err = err[:k+1]
    
    return x, k, f_val, grads, err

#%%
"""
Exercise 1:
"""
def f(x):
    return 10*(x[0]-1)**2 + (x[1]-2)**2


def grad_f(x):
    return np.array([20*(x[0]-1), 2*(x[1]-2)])


n = 2

x0 = np.zeros((n, ))
kmax = 100
tolf = 1e-6
tolx = 1e-5

x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, kmax, tolf, tolx)

x0=np.array([0,0]) 
it=np.zeros((1,x0.shape[0]))

def callF(x):   #it takes as input the k-th iterate computed
    global it
    x = np.reshape(x,(1,x.shape[0]))
    it = np.append(it,x, axis=0)

x_ax = np.linspace(-5, 4,100)
y_ax = np.linspace(-6, 6, 100)
xv, yv = np.meshgrid(x_ax, y_ax)
z_ax = f([xv,yv])

contours = plt.contour(x_ax, y_ax, z_ax)
plt.plot(x[:, 0], x[:, 1], '-o')
plt.show()

#%%
"""
Exercise 2:
"""
n = 5
A = np.random.rand(n, n) #np.vander(np.ones((n, )))
b = np.random.rand(n)

def f(x):
    return 0.5 * np.linalg.norm(A@x - b, 2) ** 2

def grad_f(x):
    return A.T@(A@x - b)

x0 = np.zeros((n, ))
kmax = 100
tol = 1e-6

x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, kmax, tol)














