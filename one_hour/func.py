import numpy as np

import math

def Softmax(z):
    z = z - np.amax(z)
    exps = np.exp(z)
    return exps / np.sum(exps, axis=1, keepdims=True)


def Relu(z):
    return np.maximum(z, 0, z)


def Relu_prime(X):
    return 1 * (X > 0)

def cost_error(Y, T):
    # avoid invalid value
    Y[Y == 0] = math.e
    return - np.sum(np.multiply(T, np.log(Y))) / Y.shape[0]

