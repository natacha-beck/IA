import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    softmax_values  = []
    sum_of_exp      = sum(np.exp(L))

    for i in L:
      softmax_values.append((np.exp(i)/sum_of_exp))

    return softmax_values