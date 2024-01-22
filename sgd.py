# Below the code I used in my SGD blog on medium in [https://medium.com/munchy-bytes/navigating-the-learning-curve-gradient-descent-stochastic-gradient-descent-in-ml-e8ec03efa673]

'''
Here is the GD developped from scratch
'''


import numpy as np

# Define the dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
alpha = 0.01  # Learning rate
epochs = 1000  # Number of iterations
w = 0  # Model parameter

# Perform Gradient Descent
for epoch in range(epochs):
    y_pred = w * X
    gradient = (-2/len(X)) * sum(X * (y - y_pred))
    w = w - alpha * gradient

print("Optimal parameter is: w =", w)


'''
Here is the SGD algorithm applied to linear regression
'''


# Define the dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
alpha = 0.01  # Learning rate
epochs = 1000  # Number of iterations
w = 0  # Model parameter

# Perform Stochastic Gradient Descent
for epoch in range(epochs):
  # here is the main difference where teh gradient is computed on a single data point
    for i in range(len(X)):
        y_pred = w * X[i]
        gradient = -2 * X[i] * (y[i] - y_pred)
        w = w - alpha * gradient

print("Optimal parameter is: w =", w)
