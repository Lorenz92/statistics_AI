# This is a partial code, wrote just to help you to follow the exercise.
# Please complete every "..." in the following, and feel free to add new
# lines where needed.

import ...

data = ... # Load the data from SoundSourceData.mat

# Split the acquired data into F and X.
F = ...
X = ...

# Variables
m = ... 
n, p = ...

... # Set equal to zero, all negative elements of X

from utils import NMF

# NMF Algorithm
W, H = ...


# Visualize the Results (since you don't know how to use matplotlib, the 
# following part of the code is already complete).

plt.figure(figsize=(40, 10))
for i in range(m):
    plt.subplot(2, m, i+1)
    plt.plot(H[i, :])
    plt.grid()
for i in range(m):
    plt.subplot(2, m, m+i+1)
    plt.plot(F[i, :])
    plt.grid()
plt.show() 


























