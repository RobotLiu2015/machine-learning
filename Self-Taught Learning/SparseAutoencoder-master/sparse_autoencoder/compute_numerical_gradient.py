import numpy as np


def compute_numerical_gradient(func, theta):
    # theta: a vector of parameters
    # func: a function that outputs a real-number. Calling y = J(theta) will return the
    # function value at theta. 

    # Initialize numgrad (no need to initialize to zero, empty_like is a good fit here)
    numgrad = np.empty_like(theta)

    # Instructions: 
    # Implement numerical gradient checking, and return the result in numgrad.  
    # (See Section 2.3 of the lecture notes.)
    # You should write code so that numgrad(i) is (the numerical approximation to) the 
    # partial derivative of func with respect to the i-th input argument, evaluated at theta.
    # I.e., numgrad(i) should be the (approximately) the partial derivative of func with
    # respect to theta(i).
    #
    # Hint: You will probably want to compute the elements of numgrad one at a time.
    epsilon = 1e-4
    for i in xrange(theta.size):
        # temporarily save the value
        theta_i = theta[i]
        # temporarily increase the value
        theta[i] = theta_i + epsilon
        val_plus = func(theta)
        # temporarily decrease the value
        theta[i] = theta_i - epsilon
        val_minus = func(theta)
        # compute numerical gradient
        numgrad[i] = (val_plus - val_minus) / (epsilon * 2)
        # restore theta
        theta[i] = theta_i

    return numgrad
