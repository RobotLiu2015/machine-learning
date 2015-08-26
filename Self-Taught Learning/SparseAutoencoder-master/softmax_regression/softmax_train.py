import numpy as np
from scipy.optimize import minimize

from softmax_cost import softmax_cost_and_grad


def softmax_train(input_size, num_classes, decay_lambda, input_data, labels, options):
    # softmax_train trains a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # inputSize: the size of an input vector x^(i)
    # numClasses: the number of classes
    # lambda: weight decay parameter
    # inputData: an N by M matrix containing the input data, such that
    # inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    if not 'maxiter' in options:
        options['maxiter'] = 400

    # initialize parameters
    theta = 0.005 * np.random.randn(input_size * num_classes, )

    # Use minFunc to minimize the function
    options['disp'] = True
    func_args = (num_classes, input_size, decay_lambda, input_data, labels)
    res = minimize(softmax_cost_and_grad, x0=theta, args=func_args, method='L-BFGS-B',
                   jac=True, options=options)
    opt_theta = res.x

    # Fold softmaxOptTheta into a nicer format
    softmax_model = {'opt_theta': opt_theta.reshape(input_size, num_classes),
                     'input_size': input_size,
                     'num_classes': num_classes}

    return softmax_model