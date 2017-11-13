import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def train_test_files_q123():
    ## load the trainning file
    x_train_file = 'Question2_123/x_train.csv'
    y_train_file = 'Question2_123/y_train.csv'
    ## test samples for prediction
    x_test_file = 'Question2_123/x_test.csv'
    y_test_file = 'Question2_123/y_test.csv'

    training_data = np.loadtxt(x_train_file, delimiter=',')
    y_train = np.genfromtxt(y_train_file, delimiter=',', dtype=int)
    ## encode the class labels into one-hot encoding
    nb_classes = 4
    training_labels = np.eye(nb_classes, dtype=float)[y_train]

    test_data = np.loadtxt(x_test_file, delimiter=',')
    y_test = np.genfromtxt(y_test_file, delimiter=',', dtype=int)
    test_labels = np.eye(nb_classes, dtype=float)[y_test]

    return training_data, test_data, training_labels, test_labels

def read_from_file(file, isWeight=True, network=None):
    b1, w1 = [], []
    with open(file) as f:
        reader = csv.reader(f)
        if isWeight:
            end_point = len(network) - 1
            layer_num = 0
            idx = 0
            rows = np.empty((network[layer_num], network[layer_num + 1]))
            for row in reader:
                rows[idx] = np.array(row[1:])
                idx += 1
                if (idx == network[layer_num]):
                    w1.append(rows)
                    layer_num += 1
                    idx = 0
                    if (layer_num == end_point):
                        return w1
                    rows = np.empty((network[layer_num], network[layer_num + 1]))

        else:
            for row in reader:
                b1.append(np.array(row[1:], dtype=float))
        return b1


def write_to_file(file_name, data):
    path = 'e0146282/' + file_name
    with open(path, 'ab') as fp:
        for item in reversed(data):
            np.savetxt(fp, item, delimiter=',', fmt='%.17g')

def save_img(img, pic_name, str1, str2):
    path = 'e0146282'
    plt.figure(figsize=(12, 8))
    if isinstance(img, list) and len(img) == 2:
        plt.plot(img[0], 'b', label="Training")
        plt.plot(img[1], 'r', label='Test')
    else:
        plt.plot(img, label = str1)
    plt.xlabel('Iteration times')
    plt.ylabel(str1+str2)
    plt.title(str2 + ' on '+ str1 + ' Data')
    plt.legend()
    if os.path.isfile(path):
        os.remove(pic_name)
    plt.savefig(pic_name)