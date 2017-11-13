# coding=utf-8

import numpy as np

def onehot_vector(targets):
    y_enc = (np.arange(np.max(targets) + 1) == targets[:,None]).astype(float)
    return y_enc

def net_input(X, W, b):
    return (X.dot(W) + b)


def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def to_classbabel(z):
    return z.argmax(axis=1)

def cross_entropy(output, y_target):
    return -np.sum(np.log(output) * (y_target), axis = 1)

def cost(output, y_target):
    return np.mean(cross_entropy(output, y_target))

net_in = net_input(X, W, bias)
smax = softmax(net_in)

predicted_label = to_classbabel(smax)

J_cost = cost(smax, y_enc)

alphas = [0.001,0.01,0.1,1,10,100,1000]

for alpha in alphas:
    print('\nTraining with alpha')


