import numpy as np

from compute_numerical_gradient import compute_numerical_gradient


def simple_quadratic_function(x):
    # this function accepts a 2D vector as input.
    # Its outputs are:
    # value: h(x1, x2) = x1^2 + 3*x1*x2
    #   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2
    # Note that when we pass simple_quadratic_function(x) to check_numerical_gradient, we're assuming
    # that compute_numerical_gradient will use only the first returned value of this function.
    value = x[0] ** 2 + 3 * x[0] * x[1]
    return value


def simple_quadratic_function_grad(x):
    grad = np.empty_like(x)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]
    return grad


def check_numerical_gradient():
    # This code can be used to check your numerical gradient implementation
    # in computeNumericalGradient.m
    # It analytically evaluates the gradient of a very simple function called
    # simpleQuadraticFunction (see below) and compares the result with your numerical
    # solution. Your numerical gradient implementation is incorrect if
    # your numerical solution deviates too much from the analytical solution.

    # Evaluate the function and gradient at x = [4, 10] (Here, x is a 2d vector.)
    x = np.array([4.0, 10.0])
    value, grad = simple_quadratic_function(x), simple_quadratic_function_grad(x)

    # Use your code to numerically compute the gradient of simple_quadratic_function at x.
    # (The notation "simple_quadratic_function" denotes a pointer to a function.)
    numgrad = compute_numerical_gradient(simple_quadratic_function, x)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print numgrad, grad
    print 'The above two columns you get should be very similar.\n' \
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n'

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print diff
    print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n'
