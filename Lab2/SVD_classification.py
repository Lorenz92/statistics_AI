import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt

import os

# Load the data
data = scipy.io.loadmat(...)

X = ...
I = ...

# Visualize an image
plt.imshow(..., cmap='gray')
plt.show()

# Extract the subdataset of X that contains digits 3 and 4
digits = [3, 4]

X = ...
I = ...

# Separate training and test
X_train = ...
X_test = ...

I_train = ...
I_test = ...

# Create the matrices X0, X1
X1 = ...
X2 = ...

# Compute the SVD decomposition of X0 and X1
U1, _, _ = ...
U2, _, _ = ...

# Take a new, unknown digit for the test set.
y = ...

# Compute the projections of y into the two spaces
y_1 = ...
y_2 = ...

# Compute the distances
d1 = ...
d2 = ...

# Assign to the predicted class
if d1 < d2:
    predicted_class = ...
else:
    predicted_class = ...
    
# Print out
print('Predicted: ', ...)
print('True: ', ...)
