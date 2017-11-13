import numpy as np

def sigmoid(z, derivative = False):
    if derivative:
        return z * (1. - z)

    return 1. / (1+ np.exp(z))


x = np.array([[0,0,1],[0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0,1,1,0]]).T

hidden_layer = 4
eta = 0.05

l1_weights = np.random.random((x.shape[1],hidden_layer))
l1_biases = np.random.random((hidden_layer, 1)).T

l2_weights = np.random.random((hidden_layer, y.shape[0]))
l2_biases = np.random.random((y.shape[0], 1)).T

for iter in range(50):
    layer1 = sigmoid(x.dot(l1_weights) + l1_biases)

    output = sigmoid(np.dot(layer1, l2_weights) + l2_biases)

    error_signal = output - y

    layer2_delta = error_signal * sigmoid(output, derivative=True)

    layer1_error = layer2_delta.dot(l2_weights.T)
    layer1_delta = layer1_error * sigmoid(layer1, derivative=True)

    l2_weights -= eta * layer1.T.dot(layer2_delta)
    l2_biases -= eta * np.sum(error_signal, axis=0, keepdims=True)

    l1_weights -=  eta * x.T.dot(layer1_delta)
    l1_biases -= eta * np.sum(layer1_error, axis=0, keepdims=True)
    print("error:{}".format(np.mean(np.abs(error_signal))))

# import numpy as np
#
#
# def nonlin(x, deriv=False):
#     if (deriv == True):
#         return x * (1 - x)
#
#     return 1 / (1 + np.exp(-x))
#
#
# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])
#
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])
#
# np.random.seed(1)
#
# # randomly initialize our weights with mean 0
# syn0 = 2 * np.random.random((3, 4)) - 1
# syn1 = 2 * np.random.random((4, 1)) - 1
#
# for j in range(50):
#
#     # Feed forward through layers 0, 1, and 2
#     l0 = X
#     l1 = nonlin(np.dot(l0, syn0))
#     l2 = nonlin(np.dot(l1, syn1))
#
#     # how much did we miss the target value?
#     l2_error = y - l2
#
#     # if (j % 10000) == 0:


    # in what direction is the target value?
    # # were we really sure? if so, don't change too much.
    # l2_delta = l2_error * nonlin(l2, deriv=True)
    #
    # # how much did each l1 value contribute to the l2 error (according to the weights)?
    # l1_error = l2_delta.dot(syn1.T)
    #
    # # in what direction is the target l1?
    # # were we really sure? if so, don't change too much.
    # l1_delta = l1_error * nonlin(l1, deriv=True)
    #
    # syn1 += l1.T.dot(l2_delta)
    # syn0 += l0.T.dot(l1_delta)
    #
    # print("Error:" + str(np.mean(np.abs(l2_error))))
