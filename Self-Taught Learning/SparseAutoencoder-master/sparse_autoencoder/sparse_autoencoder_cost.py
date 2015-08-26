import numpy as np

from math import sqrt
import gc


def initialize_parameters(visible_size, hidden_size):
    # Initialize parameters randomly based on layer sizes.
    r = sqrt(6) / sqrt(hidden_size + visible_size + 1)  # we'll choose weights uniformly from the interval [-r, r]
    w1 = np.random.rand(visible_size, hidden_size) * 2 * r - r
    w2 = np.random.rand(hidden_size, visible_size) * 2 * r - r

    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, visible_size))

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    theta = np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    return theta


# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sparse_autoencoder_cost_and_grad(theta, visible_size, hidden_size, decay_lambda, sparsity_param, beta, data):
    # visible_size: the number of input units (probably 64)
    # hidden_size: the number of hidden units (probably 25)
    # decay_lambda: weight decay parameter
    # sparsity_param: The desired average activation for the hidden units (denoted in the lecture
    # notes by the greek alphabet rho, which looks like a lower-case "p").
    # beta: weight of sparsity penalty term
    # data: Our 10000x64 matrix containing the training data.  So, data(i,:) is the i-th training example.

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    num_combinations = visible_size * hidden_size
    w1 = theta[0:num_combinations].reshape((visible_size, hidden_size))
    w2 = theta[num_combinations:2 * num_combinations].reshape((hidden_size, visible_size))
    b1 = theta[2 * num_combinations:2 * num_combinations + hidden_size]
    b2 = theta[2 * num_combinations + hidden_size:]

    #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    #
    # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
    # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
    # [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
    # of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    #
    # Stated differently, if we were using batch gradient descent to optimize the parameters,
    # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.

    # autoencoder, y = x
    y = data

    # feedforward pass
    a1 = data
    a2 = sigmoid(np.dot(a1, w1) + b1)
    a3 = sigmoid(np.dot(a2, w2) + b2)

    # compute all deltas
    # output layer
    delta3 = (a3 - y) * a3 * (1.0 - a3)
    # hidden layer
    one_over_m = 1.0 / np.float32(data.shape[0])
    sparsity_avg = one_over_m * np.sum(a2, axis=0)
    sparsity_term = -sparsity_param / sparsity_avg + (1.0 - sparsity_param) / (1.0 - sparsity_avg)
    delta2 = (np.dot(delta3, w2.T) + beta * sparsity_term) * a2 * (1.0 - a2)
    del sparsity_term
    gc.collect()

    # compute gradient
    w1grad = one_over_m * np.dot(a1.T, delta2) + decay_lambda * w1
    w2grad = one_over_m * np.dot(a2.T, delta3) + decay_lambda * w2
    b1grad = one_over_m * np.sum(delta2, axis=0)
    b2grad = one_over_m * np.sum(delta3, axis=0)

    # compute cost
    w1_ravel = w1.ravel()
    w2_ravel = w2.ravel()
    cost = np.dot((a3 - y).ravel(), (a3 - y).ravel()) * one_over_m / 2.0 + \
           decay_lambda * (np.dot(w1_ravel, w1_ravel) + np.dot(w2_ravel, w2_ravel)) / 2.0 + \
           beta * (np.sum(sparsity_param * np.log(sparsity_param / sparsity_avg) +
                          (1.0 - sparsity_param) * np.log((1.0 - sparsity_param) / (1.0 - sparsity_avg))))

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((w1grad.ravel(), w2grad.ravel(), b1grad.ravel(), b2grad.ravel()))

    return cost, grad


def sparse_autoencoder_cost(theta, visible_size, hidden_size, decay_lambda, sparsity_param, beta, data):
    return sparse_autoencoder_cost_and_grad(theta, visible_size, hidden_size,
                                            decay_lambda, sparsity_param, beta, data)[0]