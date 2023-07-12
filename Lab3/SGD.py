# Import the libraries
import numpy as np

# Implement the SGD method
def SGD(f, grad_f, w0, data, batch_size, n_epochs, alpha):
    # Extract data
    x, y = data
    
    # Initialize
    w_val = np.zeros((n_epochs+1, w0.shape[0]))
    f_val = np.zeros(n_epochs+1)
    grads = np.zeros((n_epochs+1, w0.shape[0]))
    err = np.zeros(n_epochs+1)
    
    # Assign values for the first iteration
    w_val[0, :] = w0
    f_val[0] = f(w0, x, y)
    grads[0, :] = grad_f(w0, x, y)
    err[0] = np.linalg.norm(grads[0, :], 2)
    
    # Choose step size
    alpha = alpha
    
    w = w0

    # Copy the data
    x_copy = np.copy(x)
    y_copy = np.copy(y)

    # Compute the number of batch iteration for each epoch
    n_iter_per_epoch = int(x.shape[0] / batch_size)

    # For each epoch
    for epoch in range(1, n_epochs+1):
        
        # Inner iterations
        for k in range(n_iter_per_epoch):
            
            # Random indices that composes our mini-batch (look at np.random.choice)
            batch_idx = np.random.choice(x.shape[0], batch_size, replace=False)
            
            # Split
            mask = np.ones((x.shape[0], ), dtype=bool)
            mask[batch_idx] = False
            
            x_batch = x[~mask, :]
            y_batch = y[~mask]
            
            x = x[mask, :]
            y = y[mask]
            
            # Update weights
            w = w - alpha * grad_f(w, x_batch, y_batch)
            
        # Refill the data
        x = np.copy(x_copy)
        y = np.copy(y_copy)
        
        # Update the values of the vector after each epoch
        w_val[epoch] = w
        f_val[epoch] = f(w_val[epoch, :], x, y)
        grads[epoch, :] = grad_f(w_val[epoch, :], x, y)
        err[epoch] = np.linalg.norm(grads[epoch, :], 2)
    
    # Truncate the excess
    w_val  = w_val[:epoch, :]
    f_val = f_val[:epoch]
    grads = grads[:epoch, :]
    err = err[:epoch]
    
    return w_val, f_val, grads, err










