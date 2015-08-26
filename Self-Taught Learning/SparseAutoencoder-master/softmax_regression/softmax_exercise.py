# CS294A/CS294W Softmax Exercise 

from sklearn.datasets import fetch_mldata
import numpy as np

from softmax_cost import softmax_cost, softmax_cost_and_grad
from softmax_train import softmax_train
from softmax_predict import softmax_predict
from sparse_autoencoder.compute_numerical_gradient import compute_numerical_gradient


# Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function 
#  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#  (However, you may be required to do so in later exercises)

#======================================================================
# STEP 0: Initialise constants and parameters
#
# Here we define and initialise some constants which allow your code
# to be used more generally on any arbitrary input. 
# We also initialise some parameters used for tuning the model.

input_size = 28 * 28  # Size of input vector (MNIST images are 28x28)
num_classes = 10  # Number of classes (MNIST images fall into 10 classes)

decay_lambda = 1e-4  # Weight decay parameter

#======================================================================
# STEP 1: Load data
#
# In this section, we load the input and output data.
# For softmax regression on MNIST pixels, 
# the input data is the images, and 
# the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as 
# train-images.idx3-ubyte / train-labels.idx1-ubyte

mnist = fetch_mldata('MNIST original', data_home='../data/')

# scale the pixel values to the range [0,1]
images = np.float32(mnist.data) / 255.0
labels = mnist.target

input_data = images

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

DEBUG = False  # Set DEBUG to true when debugging.
if DEBUG:
    input_size = 8
    input_data = np.random.randn(100, 8)
    labels = np.random.randint(10, size=(100, ))

# Randomly initialise theta
theta = 0.005 * np.random.randn(input_size * num_classes, )

#======================================================================
# STEP 2: Implement softmaxCost
#
# Implement softmaxCost in softmaxCost.m.

cost, grad = softmax_cost_and_grad(theta, num_classes, input_size, decay_lambda, input_data, labels)

#======================================================================
# STEP 3: Gradient checking
#
# As with any learning algorithm, you should always check that your
# gradients are correct before learning the parameters.
# 

if DEBUG:
    func = lambda x: softmax_cost(x, num_classes, input_size, decay_lambda, input_data, labels)
    numgrad = compute_numerical_gradient(func, theta)

    # Use this to visually compare the gradients side by side
    print numgrad, grad

    # Compare numerically computed gradients with those computed analytically
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print diff
    # The difference should be small. 
    # In our implementation, these values are usually less than 1e-7.

    # When your gradients are correct, congratulations!

#======================================================================
# STEP 4: Learning parameters
#
# Once you have verified that your gradients are correct,
# you can start training your softmax regression code using softmaxTrain
# (which uses minFunc).

options = {'maxiter': 100}
softmax_model = softmax_train(input_size, num_classes, decay_lambda, input_data, labels, options)

# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

#======================================================================
# STEP 5: Testing
#
# You should now test your model against the test images.
# To do this, you will first need to write softmaxPredict
# (in softmaxPredict.m), which should return predictions
# given a softmax model and the input data.

# You will have to implement softmaxPredict in softmaxPredict.m
pred = softmax_predict(softmax_model, input_data)

acc = np.mean(labels == pred)
print 'Accuracy: {0:.3f}\n'.format(acc * 100)

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.200#
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)
