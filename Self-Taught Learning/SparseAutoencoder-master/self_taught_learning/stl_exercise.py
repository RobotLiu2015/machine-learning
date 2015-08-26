from random import shuffle

from sklearn.datasets import fetch_mldata
import numpy as np
from scipy.optimize import minimize

from sparse_autoencoder.display_network import display_network
from sparse_autoencoder.sparse_autoencoder_cost import initialize_parameters, sparse_autoencoder_cost_and_grad
from softmax_regression.softmax_train import softmax_train
from softmax_regression.softmax_predict import softmax_predict
from feed_forward_autoencoder import feed_forward_autoencoder


# # CS294A/CS294W Self-taught Learning Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
#  You will also need to have implemented sparseAutoencoderCost.m and 
#  softmaxCost.m from previous exercises.
#
## ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

input_size = 28 * 28
num_labels = 5
hidden_size = 200
sparsity_param = 0.1  # desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
#  in the lecture notes).
decay_lambda = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term
max_iter = 400

## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# Load MNIST database files
mnist = fetch_mldata('MNIST original', data_home='../data/')
images = np.float32(mnist.data) / 255.0
labels = mnist.target

# Set Unlabeled Set (All Images)

# Simulate a Labeled and Unlabeled set
labeled_set = np.where((labels >= 0) & (labels <= 5))[0]
unlabeled_set = np.where(labels >= 6)[0]

unlabeled_data = images[unlabeled_set]

# Output Some Statistics
print '# examples in unlabeled set: {0}'.format(unlabeled_data.shape[0])

## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images. 

trained_theta_file = '../data/opttheta.npy'
TRAIN = False
if TRAIN:
    #  Randomly initialize the parameters
    theta = initialize_parameters(input_size, hidden_size)

    #  Find opttheta by running the sparse autoencoder on
    #  unlabeledTrainingImages
    func_args = (input_size, hidden_size, decay_lambda, sparsity_param, beta, unlabeled_data)
    res = minimize(sparse_autoencoder_cost_and_grad, x0=theta, args=func_args, method='L-BFGS-B',
                   jac=True, options={'maxiter': max_iter, 'disp': True})
    opttheta = res.x
    np.save(trained_theta_file, opttheta)
else:
    opttheta = np.load(trained_theta_file)

## -----------------------------------------------------

# Visualize weights
w1 = opttheta[0: hidden_size * input_size].reshape((input_size, hidden_size))
display_network(w1.T, save_figure_path='../data/stl.png')

##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#  
#  You need to complete the code in feedForwardAutoencoder.m so that the 
#  following command will extract features from the data.

num_train = np.round(labeled_set.shape[0] / 2)
indices = [i for i in xrange(labeled_set.shape[0])]
shuffle(indices)
train_set = labeled_set[indices[0:num_train]]
test_set = labeled_set[indices[num_train:]]

train_data = images[train_set]
train_labels = labels[train_set]  # Shift Labels to the Range 1-5

test_data = images[test_set]
test_labels = labels[test_set]  # Shift Labels to the Range 1-5

print '# examples in supervised training set: {0}'.format(train_data.shape[0])
print '# examples in supervised testing set: {0}'.format(test_data.shape[0])

train_features = feed_forward_autoencoder(opttheta, hidden_size, input_size, train_data)

test_features = feed_forward_autoencoder(opttheta, hidden_size, input_size, test_data)

##======================================================================
## STEP 4: Train the softmax classifier
#  Use softmaxTrain.m from the previous exercise to train a multi-class
#  classifier. 

#  Use lambda = 1e-4 for the weight regularization for softmax

# You need to compute softmaxModel using softmaxTrain on trainFeatures and
# trainLabels

num_classes = 10
decay_lambda = 1e-4
options = {'maxiter': 100}
softmax_model = softmax_train(hidden_size, num_classes, decay_lambda, train_features, train_labels, options)

##======================================================================
## STEP 5: Testing 

# Compute Predictions on the test set (testFeatures) using softmaxPredict
# and softmaxModel
pred = softmax_predict(softmax_model, test_features)

## -----------------------------------------------------
# Classification Score
acc = np.mean(test_labels == pred)
print 'Test Accuracy: {0:.3f}\n'.format(100 * acc)

# (note that we shift the labels by 1, so that digit 0 now corresponds to
#  label 1)
#
# Accuracy is the proportion of correctly classified images
# The results for our implementation was:
#
# Accuracy: 98.3#
#
# 
