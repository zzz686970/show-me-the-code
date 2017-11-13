import random
import numpy as np
from functions import *
from file_operation import  *


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
        weight.append(np.random.random((list[i], list[i + 1]))/np.sqrt(list[i]))
        bias.append(np.zeros((1, list[i + 1])))
    return weight, bias

# create different function layers
def generate_layers(network, w, b):
    layers = []
    ## no need to change for the input layer
    num_layer = len(network) - 1
    for i in range(num_layer):
        if i < num_layer-1:
            layers.append(relu_layer(w[i],b[i]))
        else:
            layers.append(softmax_layer(w[i],b[i]))

    return layers

def feedforward(input, layers):
    """
    :param input: train_data
    :param layers: layer list
    :return: net_ou tput in each layer
    """
    outputs = [input]
    hidden_output = input
    for layer in layers:
        hidden_output = layer.activation(np.dot(np.array(hidden_output), layer.weight) + layer.bias)
        outputs.append(hidden_output)
    return outputs

def sgd(batch_size, eta, layers, num_epochs):
    """
    :param batch_size: num of batches to update
    :param eta: learning rate
    :param layers: list
    :param num_epochs: iterations times
    :return: train_acc, train_cost, test_acc, test_cost
    """
    test_cost, test_acc = [],[]
    train_cost, train_acc = [], []
    ## load training and test records from method in file_operation.py file
    train_data, test_data, train_labels, test_labels = train_test_files_q123()
    # total_size = train_data.shape[0]
    ## shuffle training data
    # apply mini-batch training
    # ater gradients update, test cost and accuracy
    for epoch in range(num_epochs):

        if len(train_data) == len(train_labels):
            combined = list(zip(train_data, train_labels))
            random.shuffle(combined)
            train_data[:],train_labels[:] = zip(*combined)

        for batch in range(0, train_data.shape[0], batch_size):
            training(layers, train_data[batch:batch + batch_size], train_labels[batch:batch + batch_size], '', '', True, eta)

        tr_acc, tr_lost = accuracy_cost(train_data, layers, train_labels)
        te_acc, te_lost = accuracy_cost(test_data, layers, test_labels)
        # print('iteration % d, the accuracy on train set is %.2f , and cost is %.2f' % (epoch + 1, score_train, cost_train))
        print('iteration % d, the accuracy on test set is %.2f , and cost is %.2f' % (epoch + 1, te_acc, te_lost))
        test_acc.append(te_acc)
        test_cost.append(te_lost)
        train_acc.append(tr_acc)
        train_cost.append(tr_lost)

    return train_acc, train_cost, test_acc, test_cost

def training(layers, input, targets, wfile, bfile, isTraining, eta):
    """
    :param layers: layer list
    :param input: training data or test data
    :param targets: training labels or test labels
    :param wfile: weight file for verification or final output, None in q123
    :param bfile: bias file for verification or final output, None in q123
    :param isTraining: export files
    :param eta: learning rate
    """
    # forward, return the stored output from all layers
    # decay_rate = 0.0001
    lmbda = 0.5
    # batch_size = len(input)
    output = feedforward(input, layers)

    # update weight and bias, but need to calculate delta_gradient first
    delta_weights, delta_biases = backprop(output, targets, layers)
    if isTraining:
        idx = 0
        for layer in reversed(layers):
            # layer.weight = [(1-eta * (lmbda/total_size))* w - (eta/input.shape[0]) * nw for w, nw in zip(layer.weight, weight_grad_sums)]
            # layer.bias = [b - (eta/input.shape[0]) * nb for b, nb in zip(layer.bias, bias_grad_sums)]
            layer.weight -= delta_weights[idx] * eta
            layer.bias -= delta_biases[idx] * eta
            idx += 1
    else:
        write_to_file(bfile, delta_biases)
        write_to_file(wfile, delta_weights)

def backprop(output, targets, layers):
    """
    gradient at a given layer is the matrix multiplication of
    the output grad from next layer and the transpose of the weights coming out of that layer
    :param output: output for each layer
    :param targets: labels
    :param layers: layer list
    :return: weights gridient and biases gradients
    """
    output_grad = None
    delta_biases = []
    delta_weights = []

    for layer in reversed(layers):
        Y = output.pop()
        ## last layer
        if output_grad is None:
            error_signal = layer.calc_gradient(Y, targets)
        else:
            error_signal = layer.calc_gradient(Y, output_grad)
        # each time output pop out an element, pick the last layer
        X = output[-1]
        ## calculate the average loss
        w_grad = np.float32(X.T.dot(error_signal) / error_signal.shape[0])
        if error_signal.shape[0] > 1:
            b_grad = np.float32(np.mean(error_signal, axis=0))
        else:
            b_grad = np.float32(error_signal)
        delta_weights.append(w_grad)
        delta_biases.append(b_grad)
        # weight_grad_sums = [nb + dnb for nb, dnb in zip(weight_grad_sums, delta_weights)]
        # bias_grad_sums = [nw + dnw for nw, dnw in zip(bias_grad_sums, delta_biases)]
        output_grad = np.dot(error_signal, layer.weight.T)
    return delta_weights, delta_biases

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


def q2_123(net, batch_size, rate, num_epochs):
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

    ## record cost and accuracy for plotting
    train_acc, train_cost, test_acc, test_cost = sgd(batch_size, rate, layers, num_epochs)

    # draw figure of cost data
    draw_img = {'Test Cost': test_cost, 'Train Cost': train_cost, 'Train-Test Accuracy': [train_acc, test_acc] }
    file_name = [nn + ' ' + str(batch_size) + '-' + str(rate) + '-' + str(num_epochs) + '-' + key.lower()+ '.jpeg' for key in draw_img.keys()]
    for ele, (key, value) in zip(file_name, draw_img.items()):
        save_img(value, ele, key.split(' ')[0], key.split(' ')[1])


def q2_4():
    test_data = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    test_labels = np.array([[0, 0, 0, 1]])

    network1 = [14, 100, 40, 4]
    b1 = read_from_file('Question2_4/c/b-100-40-4.csv', False)
    w1 = read_from_file('Question2_4/c/w-100-40-4.csv', network=network1)
    layer1 = generate_layers(network1, w1, b1)

    network2 = [14] + [28] * 6 + [4]
    b2 = read_from_file('Question2_4/c/b-28*6-4.csv', False)
    w2 = read_from_file('Question2_4/c/w-28*6-4.csv', network=network2)
    layer2 = generate_layers(network2, w2, b2)

    network3 = [14] + [14] * 28 + [4]
    b3 = read_from_file('Question2_4/c/b-14*28-4.csv', False)
    w3 = read_from_file('Question2_4/c/w-14*28-4.csv', network=network3)
    layer3 = generate_layers(network3, w3, b3)

    # run one-time training
    training(layer1, test_data, test_labels, 'dw-100-40-4.csv', 'db-100-40-4.csv', False, 1)
    training(layer2, test_data, test_labels, 'dw-28-6-4.csv', 'db-28-6-4.csv', False, 1)
    training(layer3, test_data, test_labels, 'dw-14-28-4.csv', 'db-14-28-4.csv', False, 1)
    print('Job finished!')

if __name__ == "__main__":
    ## generate plots
    q2_123(1, 50, 0.005, 100)
    # q2_123(1, 50, 0.01, 100)
    # q2_123(1, 50, 0.05, 100)
    # q2_123(1, 50, 0.1, 100)
    # q2_123(1, 50, 0.5, 100)
    # q2_123(2, 50, 0.005, 100)
    # q2_123(2, 50, 0.01, 100)
    # q2_123(2, 50, 0.05, 100)
    # q2_123(2, 50, 0.1, 100)
    # q2_123(2, 50, 0.5, 100)
    # q2_123(3, 50, 0.005, 100)
    # q2_123(3, 50, 0.01, 100)
    # q2_123(3, 50, 0.05, 100)
    # q2_123(3, 50, 0.1, 100)
    # q2_123(3, 50, 0.5, 100)
    # q2_123(2, 40, 0.010, 50)
    # q2_123(1, 512, 2, 10)

    # Output dw and db files
    # q2_4()
