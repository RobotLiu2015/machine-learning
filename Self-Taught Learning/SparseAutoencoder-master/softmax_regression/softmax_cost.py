import numpy as np


def softmax_cost_and_grad(theta, num_classes, input_size, decay_lambda, data, labels):
    # num_classes - the number of classes
    # input_size - the size N of the input vector
    # decay_lambda - weight decay parameter
    # data - the N x M input matrix, where each row data(i, :) corresponds to
    # a single test set
    # labels - an 1 x N matrix containing the labels corresponding for the input data
    #

    # Unroll the parameters from theta
    theta = theta.reshape((input_size, num_classes))

    num_cases = data.shape[0]

    ground_truth = np.zeros((num_cases, num_classes))
    ground_truth[xrange(num_cases), labels.tolist()] = 1

    thetagrad = np.empty_like(theta)

    # Instructions: Compute the cost and gradient for softmax regression.
    #               You need to compute thetagrad and cost.
    #               The groundTruth matrix might come in handy.
    theta_x = np.dot(data, theta)

    # subtract maximum value of theta_x to avoid overflow
    theta_x_max = np.amax(theta_x, axis=1, keepdims=True)
    theta_x -= theta_x_max
    exp_theta_x = np.exp(theta_x)

    # compute h(x)
    hypothesis = exp_theta_x / np.sum(exp_theta_x, axis=1, keepdims=True)

    # compute cost
    cost = -np.sum(ground_truth * np.log(hypothesis)) / num_cases + \
           decay_lambda / 2.0 * np.dot(theta.flatten(), theta.flatten())

    # compute gradient
    thetagrad = -np.dot(data.T, ground_truth - hypothesis) / num_cases + decay_lambda * theta

    # ------------------------------------------------------------------
    # Unroll the gradient matrices into a vector for minFunc
    grad = thetagrad.ravel()

    return cost, grad


def softmax_cost(theta, num_classes, input_size, decay_lambda, data, labels):
    return softmax_cost_and_grad(theta, num_classes, input_size, decay_lambda, data, labels)[0]