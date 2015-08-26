import numpy as np

from sparse_autoencoder.sparse_autoencoder_cost import sigmoid


def feed_forward_autoencoder(theta, hidden_size, visible_size, data):
    # theta: trained weights from the autoencoder
    # visibleSize: the number of input units (probably 64)
    # hiddenSize: the number of hidden units (probably 25)
    # data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    num_combinations = visible_size * hidden_size
    w1 = theta[0:num_combinations].reshape((visible_size, hidden_size))
    b1 = theta[2 * num_combinations:2 * num_combinations + hidden_size]

    # Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
    activation = sigmoid(np.dot(data, w1) + b1)

    #-------------------------------------------------------------------
    return activation




