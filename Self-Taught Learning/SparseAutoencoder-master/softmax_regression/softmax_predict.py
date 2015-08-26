import numpy as np


def softmax_predict(softmax_model, data):
    # softmaxModel - model trained using softmaxTrain
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    # a single test set
    #
    # Your code should produce the prediction matrix
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).

    # Unroll the parameters from theta
    theta = softmax_model['opt_theta']  # this provides a numClasses x inputSize matrix

    #  Instructions: Compute pred using theta assuming that the labels start
    #                from 1.
    theta_x = np.dot(data, theta)

    pred = np.argmax(theta_x, axis=1)

    return pred

