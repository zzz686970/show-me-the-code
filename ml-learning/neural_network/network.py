from activation import *
import numpy as np
import random
import pickle

class CrossEntropyCost(object):
    """
    return the cost associated with an output a and desired output y
    if bothe a and y have 1.0 in the same slot, (1-y) * np.log(1-a) is converted to (0,0)
    """
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y) * np.log(1-a)))

    def delta(z, a, y):
        """
        :return: error delta from the ouput layer
        z is not used by the method, just for consistency with the delta method for other cost classes
        """
        return (a-y)

class Neural_Network(object):
    def __init__(self, layer_size, activation, activation_prime, loss_error):
        self.num_layers = len(layer_size)
        self.num_weights = self.num_layers - 1

        self.activation = activation
        self.activation_prime = activation_prime
        self.loss_error = loss_error

        self.weights = np.array([np.random.randn(layer_size[i+1],
                                                 layer_size[i]) * np.sqrt(2.0 / layer_size[i])
                                 for i in range(self.num_layers - 1)])
        self.biases = np.array([np.random.randn(s, 1) for s in layer_size[1:]])

        self.net_inputs = np.array([np.zeros((s, 1)) for s in layer_size])
        self.outputs = np.array([np.zeros((s, 1)) for s in layer_size])

        def write_to_file(self):
            with open('models/model.pkl', 'wb') as write_file:
                pickle.dump(self.weights, write_file)
                pickle.dump(self.biases, write_file)

        def read_from_file(self, read_file):
            with open(read_file, 'r') as f:
                self.weights = pickle.load(f)
                self.biases = pickle.load(f)

        def predict(self, image):
            self.forward_feedback(image)
            return np.argmax(self.outputs[-1])


        def forward_feedback(self, image):
            output = image
            self.net_inputs[0] = output
            self.outputs[0] = output
            for w in range(self.num_weights):
                net_input = np.dot(self.weights[w], output) + self.biases[w]
                output = self.activation(net_input)
                self.net_inputs[w+1] = net_input
                self.outputs[w+1] = output

        def calc_error(self, layer, targets=None, pre_errors=None):
            if targets is not None:
                return self.loss_error(self.net_inputs[-1], self.outputs[-1], targets)

            return(np.dor(np.transpose(self.weights[layer]), pre_errors) * self/activation_prime(self.net_inputs[layer]))

        def calc_weights_grad(self, weight_num, pre_errors):
            return np.dot(pre_errors, np.transpose(self.outputs[weight_num]))

        def calc_bias_grad(self, weight_num, pre_errors):
            return pre_errors

        def backprop(self, targets):
            weights_grads = np.array([np.zeros(w.shape) for w in self.weights])
            bias_grads = np.array([np.zeros(b.shape) for b in self.biases])

            errors = self.calc_errors(self.num_layers -1, targets = targets)
            weights_grads[-1] = self.calc_weight_grad(self.num_weights - 1)
            bias_grads[-1] = self.calc_bias_grad(self.num_weights - 1, errors)

            for layer in reversed(range(1, self.num_layers)):
                weights_grads[layer - 1] = self.calc_weight_grad(layer-1, errors)
                bias_grads[layer - 1] = self.calc_bias_grad(layer - 1, errors)

            return weights_grads, bias_grads

        def test(self, images, labels):
            num_correct = 0
            for image, label in zip(images, labels):
                prediction = self.predict(image)
                if prediction == label:
                    num_correct += 1
                return (num_correct * 1.) / len(images)

        def train(self, train_images, train_labels, test_images, test_labels,
                  num_epochs = 20, batch_size = 10, learning_rate = 0.5,
                  decay_rate = 0.0, momentum = 0.5):
            weight_vel = np.array([np.zeros(w.shape) for w in self.weights])
            bias_vel = np.array([np.zeros(b.shape) for b in self.biases])

            max_acc = 0.0

            for epoch in range(num_epochs):
                combined = list(zip(train_images, train_labels))
                random.shuffle(combined)
                train_images[:], train_labels[:] = zip(*combined)

                for batch in range(len(train_images)/batch_size):
                    start_idx = batch * batch_size
                    images = train_images[start_idx:start_idx+batch_size]
                    labels = train_labels[start_idx:start_idx+batch_size]
                    weight_grad_sums = np.array([np.zeros(w.shape) for w in self.weights])
                    bias_grad_sums = np.array([np.zeros(b.shapre) for b in self.biases])

                batch_size = len(images)
                weight_delta = learning_rate * (weight_grad_sums / batch_size + decay_rate * self.weights)
                bias_delta = learning_rate * (bias_grad_sums / batch_size)

                weight_vel = (momentum * weight_vel) - weight_delta
                bias_vel = (momentum * bias_vel) - bias_delta
                self.weights += weight_vel
                self.biases += bias_vel

            accuracy = self.test(test_images, test_labels)
            if accuracy > max_acc:
                max_ac= accuracy
                self.write_to_file()
            print('finished epoch {},\t Acciracy: {}'.format((epoch+1, accuracy)))
        print('Finished training')

