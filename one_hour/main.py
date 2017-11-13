import numpy as np
import math
import matplotlib.pyplot as plt
import os.path
from func import *
from file import *
import random

np.set_printoptions(precision=16)

class layers(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

class softmax_layer(layers):
    def activation(self, net_input):
        return Softmax(net_input)

    def calc_gradient(self, output, targets):
        return (output - targets)

class relu_layer(layers):
    def activation(self, net_input):
        return Relu(net_input)

    def calc_gradient(self, output, output_grad):
        return np.multiply(Relu_prime(output), output_grad)

def init_weight_bias(list):
    """
    Initial weights in each layer using a Gaussian distribution with mean 0
        and variance 1 over the square root of the number of weights connecting to the same neuron.
    Initialize the biases in each layer using a Gaussian distribution with mean 0 and standard
        deviation 1.
    """
    bias = []
    weight = []
    for i in range(len(list) - 1):
        r = math.sqrt(2.0 / list[i])
        weight.append(np.random.uniform(size=(list[i], list[i + 1]), low=-r, high=r))
        # weight.append(np.random.random((list[i], list[i + 1]))/np.sqrt(list[i]))
        bias.append(np.zeros((1, list[i + 1])))
    return weight, bias

# create different function layers
def generate_layers(network, w, b):
    layers = []
    ## no need to change for the input layer
    num_layer = len(network) - 1
    for i in range(num_layer):
        if i < num_layer - 1:
            layers.append(relu_layer(w[i], b[i]))
        else:
            layers.append(softmax_layer(w[i], b[i]))
    return layers

# feedforward process
def feedforward(input, layers):
    outputs = [input]
    x = input
    for layer in layers:
        x = layer.activation(np.dot(np.array(x), layer.weight) + layer.bias)
        outputs.append(x)
    return outputs


def sgd(batch_size, eta, layers, num_epochs):
    """
    :param batch_size: num of batches to update
    :param eta: learning eta
    :param layers: list
    :param num_epochs: iterations times
    :return: train_acc, train_cost, test_acc, test_cost
    """
    test_cost, test_acc = [],[]
    train_cost, train_acc = [], []
    ## load training and test records from method in file_operation.py file
    x_train, x_test, y_train, y_test = train_test_files_q123()
    for epoch in range(num_epochs):
        assert len(x_train) == len(y_train)
        combined = list(zip(x_train, y_train))
        random.shuffle(combined)
        x_train[:], y_train[:] = zip(*combined)

        for i in range(0, x_train.shape[0], batch_size):
            training(layers, x_train[i:i + batch_size], y_train[i:i + batch_size], '', '', True, eta)
        tr_acc, tr_loss = accuracy_cost(x_train, layers, y_train)
        te_acc, te_loss = accuracy_cost(x_test, layers, y_test)
        print('iteration % d, the accuracy on train set is %.2f , and cost is %.2f' % (epoch + 1, tr_acc, tr_loss))
        # print('iteration %d, the accuracy on test set is %.2f , and cost is %.2f' % (j + 1, test_acc, te_loss))
        test_acc.append(te_acc)
        test_cost.append(te_loss)
        train_acc.append(tr_acc)
        train_cost.append(tr_loss)
    return train_acc, train_cost, test_acc, test_cost



# backpropagation process
def backpropagation(activations, targets, layers):
    output_grad = None
    dB = []
    dW = []
    for layer in reversed(layers):
        Y = activations.pop()
        ## last layer
        if output_grad is None:
            error_signal = layer.calc_gradient(Y, targets)
        else:
            error_signal = layer.calc_gradient(Y, output_grad)
        X = activations[-1]
        w_grad = np.float32(X.T.dot(error_signal) / error_signal.shape[0])
        if error_signal.shape[0] > 1:
            b_grad = np.float32(np.mean(error_signal, axis=0))
        else:
            b_grad = np.float32(error_signal)
        dW.append(w_grad)
        dB.append(b_grad)
        output_grad = error_signal.dot(layer.weight.T)
    return dW, dB

# one-time training
def training(layers, input, targets, wfile, bfile, isTraining, eta):
    # feedforward
    output = feedforward(input, layers)

    # backpropagation
    dW, dB = backpropagation(output, targets, layers)
    if isTraining:
        index = 0
        for layer in reversed(layers):
            layer.weight -= dW[index] * eta
            layer.bias -= dB[index] * eta
            index += 1
    else:
        write_to_file(bfile, dB)
        write_to_file(wfile, dW)

def accuracy_cost(input, layers, target):
    """
    combine accuracy and cost together
    :param input: training or test data
    :param layers: layer list
    :param target: training or test labels, alreadly converted to one-hot format
    :return: accuracy, loss
    """
    output = feedforward(input, layers)
    loss = cost_error(output[-1], target)
    preds_correct_boolean = output[-1].argmax(axis=1) == np.argmax(target, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / input.shape[0]
    return accuracy, loss



def q2_123(net, batch_size, eta, num_epochs):
    # initial weights and biases
    if net == 1:
        nn = '14-100-40-4'
        network = [14, 100,40,4]
        weights, biases = init_weight_bias(network)
        layers = generate_layers(network, weights, biases)
    elif net == 2:
        nn = '14-28*6-4'
        network = [14] + [28] * 6 + [4]
        weights, biases = init_weight_bias(network)
        layers = generate_layers(network, weights, biases)
    elif net == 3:
        nn = '14-14*28-4'
        network = [14] + [14] * 28 + [4]
        weights, biases = init_weight_bias(network)
        layers = generate_layers(network, weights, biases)
    else:
        print("Empty network")
        return

    train_acc, train_cost, test_acc, test_cost = sgd(batch_size, eta, layers, num_epochs)
    # draw figure of cost data
    draw_img = {'Test Cost': test_cost, 'Train Cost': train_cost, 'Train-Test Accuracy': [train_acc, test_acc]}
    file_name = [nn + ' ' + str(batch_size) + '-' + str(eta) + '-' + str(num_epochs) + '-' + key.lower() + '.jpeg' for
                 key in draw_img.keys()]
    for ele, (key, value) in zip(file_name, draw_img.items()):
        save_img(value, ele, key.split(' ')[0], key.split(' ')[1])


def q2_4():
    test_data = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    test_labels = np.array([[0, 0, 0, 1]])

    network1 = [14, 100, 40, 4]
    b1 = read_from_file('Question2_4/b/b-100-40-4.csv', False)
    w1 = read_from_file('Question2_4/b/w-100-40-4.csv', network=network1)
    layer1 = generate_layers(network1, w1, b1)

    network2 = [14] + [28] * 6 + [4]
    b2 = read_from_file('Question2_4/b/b-28*6-4.csv', False)
    w2 = read_from_file('Question2_4/b/w-28*6-4.csv', network=network2)
    layer2 = generate_layers(network2, w2, b2)

    network3 = [14] + [14] * 28 + [4]
    b3 = read_from_file('Question2_4/b/b-14*28-4.csv', False)
    w3 = read_from_file('Question2_4/b/w-14*28-4.csv', network=network3)
    layer3 = generate_layers(network3, w3, b3)

    # run one-time training
    training(layer1, test_data, test_labels, 'dw-100-40-4.csv', 'db-100-40-4.csv', False, 1)
    training(layer2, test_data, test_labels, 'dw-28-6-4.csv', 'db-28-6-4.csv', False, 1)
    training(layer3, test_data, test_labels, 'dw-14-28-4.csv', 'db-14-28-4.csv', False, 1)
    print('Result generated')

if __name__ == "__main__":
    # plot acc & cost
    # q2_123(1,40,0.01,50)
    ## generate output file
    q2_4()
