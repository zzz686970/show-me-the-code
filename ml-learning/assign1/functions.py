import numpy as np
import csv
# np.seterr(divide='ignore', invalid='ignore')


def Softmax(z):
    ## keep x numerically stable
    x = z - np.amax(z)
    return np.exp(x) / np.sum(np.exp(x),axis=1,keepdims=True)
#
#
def Relu(z):
    return np.maximum(z, 0.)
#
#
def Relu_prime(X):
    return 1 * (X > 0)


def cost_error(output, y_target):
    """calculate how similar the correct target class are to the actual target values,
    since our target value for every observation is one, the loss for every observation is simplified
    Hence we can sum the loss up to be the overall loss
    eg. output[0.2, 0.3, 0.5]  label [0, 0, 1]
    """
    epsilon = 1e-10
    class_label_indices = np.argmax(y_target, axis=1).astype(int)
    predicted_probability = output[np.arange(len(output)), class_label_indices]
    log_preds = np.log(predicted_probability )
    loss = -1.0 * np.sum(log_preds) / len(log_preds)

    return loss
