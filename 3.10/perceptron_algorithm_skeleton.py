import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

# Return one or zero according to the input value
#
# Keyword arguments:
#   t: a number
def stepFunction(t):
    if t >= 0:
        return 1
    return 0

# Return the prediction as zero or one
#
#  Keyword arguments:
#   X: the data an array where each element is a array with the [x,y] coordinates of the point
#   W: the weights (as an array)
#   b:the bias
def prediction(X, W, b):
    # matmul perform a matrix product of two arrays.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
    return stepFunction((np.matmul(X,W)+b)[0])

# Update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
#
# Keyword arguments:
#   X: the data an array where each element is a array with the [x,y] coordinates of the point
#   y: the labels an array of 1 and 0 (same size as X array) 1 entry foreach point define in X
#   W: the weights (as an array)
#   b: the bias
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # TODO: Fill in the code below to implement the perceptron trick.

    return W, b

# Runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
#
# Keyword arguments:
#   X: the data an array where each element is a array with the [x,y] coordinates of the point,
#   y: the labels an array of 1 and 0 (same size as X array) 1 entry foreach point define in X
#   learn_rate: the learning rate
#   num_epochs: number of iteration
#
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
