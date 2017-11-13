import numpy as np
class activation:
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1. - x)

    def tanh(self, x):
        return np.tanh(x)

    def derivative_tanh(self, x):
        return 1. - x * x

    def softmax(self, x):
        return np(x - np.max(x)) - np.sum(np.exp(x - np.max(x)))

    def relu(self, x):
        return x * (x > 0)

    def derivative_relu(self, x):
        return 1. * (x > 0)

    def leaky_relu(self, x):
        return np.where(x >= 0, x, 0.01 * x)

    def leaky_relu_prime(self, x):
        return np.where(x >= 0, 1, 0.01)