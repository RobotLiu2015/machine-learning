from scipy.optimize import minimize

from sample_images import sample_images
from display_network import display_network
from sparse_autoencoder_cost import initialize_parameters, sparse_autoencoder_cost_and_grad


def train():
    # STEP 0: Here we provide the relevant parameters values that will
    # allow your sparse autoencoder to get good filters; you do not need to
    # change the parameters below.

    patch_size = 8
    num_patches = 10000
    visible_size = patch_size ** 2  # number of input units
    hidden_size = 25  # number of hidden units
    sparsity_param = 0.01  # desired average activation of the hidden units.
    # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
    # in the lecture notes).
    decay_lambda = 0.0001  # weight decay parameter
    beta = 3  # weight of sparsity penalty term

    # STEP 1: Implement sampleIMAGES
    # After implementing sampleIMAGES, the display_network command should
    # display a random sample of 200 patches from the dataset

    patches = sample_images(patch_size, num_patches)
    #    list = [randint(0, patches.shape[0]-1) for i in xrange(64)]
    #    display_network(patches[list, :], 8)

    # Obtain random parameters theta
    #    theta = initialize_parameters(visible_size, hidden_size)

    # STEP 2: Implement sparseAutoencoderCost
    #
    #  You can implement all of the components (squared error cost, weight decay term,
    #  sparsity penalty) in the cost function at once, but it may be easier to do
    #  it step-by-step and run gradient checking (see STEP 3) after each step.  We
    #  suggest implementing the sparseAutoencoderCost function using the following steps:
    #
    #  (a) Implement forward propagation in your neural network, and implement the
    #      squared error term of the cost function.  Implement backpropagation to
    #      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking
    #      to verify that the calculations corresponding to the squared error cost
    #      term are correct.
    #
    #  (b) Add in the weight decay term (in both the cost function and the derivative
    #      calculations), then re-run Gradient Checking to verify correctness.
    #
    #  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
    #      verify correctness.
    #
    #  Feel free to change the training settings when debugging your
    #  code.  (For example, reducing the training set size or
    #  number of hidden units may make your code run faster; and setting beta
    #  and/or lambda to zero may be helpful for debugging.)  However, in your
    #  final submission of the visualized weights, please use parameters we
    #  gave in Step 0 above.
    #    cost, grad = sparse_autoencoder_cost_and_grad(theta, visible_size, hidden_size,
    #                                                  decay_lambda, sparsity_param, beta, patches)

    # STEP 3: Gradient Checking
    #
    # Hint: If you are debugging your code, performing gradient checking on smaller models
    # and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
    # units) may speed things up.

    # First, lets make sure your numerical gradient computation is correct for a
    # simple function.  After you have implemented compute_numerical_gradient,
    # run the following:
    #    check_numerical_gradient()

    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    #    func = lambda x: sparse_autoencoder_cost(x, visible_size, hidden_size,
    #                                             decay_lambda, sparsity_param, beta, patches)
    #    numgrad = compute_numerical_gradient(func, theta)

    # Use this to visually compare the gradients side by side
    #    print numgrad, grad

    # Compare numerically computed gradients with the ones obtained from backpropagation
    #    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    # Should be small. In our implementation, these values are usually less than 1e-9.
    #    print diff

    # STEP 4: After verifying that your implementation of
    # sparse_autoencoder_cost is correct, You can start training your sparse
    # autoencoder with minFunc (L-BFGS).

    # Randomly initialize the parameters
    # Use minimize interface, and set jac=True, so it can accept cost and grad together
    theta = initialize_parameters(visible_size, hidden_size)
    func_args = (visible_size, hidden_size, decay_lambda, sparsity_param, beta, patches)
    res = minimize(sparse_autoencoder_cost_and_grad, x0=theta, args=func_args, method='L-BFGS-B',
                   jac=True, options={'maxiter': 400, 'disp': True})

    # STEP 5: Visualization
    w1 = res.x[0: hidden_size * visible_size].reshape((visible_size, hidden_size))
    display_network(w1.T, 5, save_figure_path='../data/sparse_autoencoder.png')


if __name__ == "__main__":
    train()
