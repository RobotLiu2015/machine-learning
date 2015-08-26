import softmax
import sparseAutoencoder
import numpy
import scipy.io
import scipy.optimize
from random import shuffle

def sigmoid(x):
    return (1 / (1 + numpy.exp(-x)))
## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# Load MNIST database files
vis_patch_side  = 28
numLabels  = 5
hid_patch_side = 14
sparsityParam = 0.1
lamda = 0.0001
beta = 3
max_iterations = 100

mnistData = softmax.loadMNISTImages('train-images-idx3-ubyte')
mnistLabels = softmax.loadMNISTLabels('train-labels-idx1-ubyte')

labeledSet = numpy.where(mnistLabels<5)
unlabeledSet = numpy.where(mnistLabels>5)

unlabeledData = mnistData[:,unlabeledSet[0]]

visible_size = vis_patch_side * vis_patch_side  # number of input units
hidden_size  = hid_patch_side * hid_patch_side  # number of hidden units

## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images.

#调试后面时候可以屏蔽这一部分
Flag = True
if Flag:
    """ Initialize the Autoencoder with the above parameters """
    encoder = sparseAutoencoder.SparseAutoencoder(visible_size, hidden_size, sparsityParam, lamda, beta)

    """ Run the L-BFGS algorithm to get the optimal parameter values """
    opt_solution  = scipy.optimize.minimize(encoder.sparseAutoencoderCost, encoder.theta,
                                            args = (unlabeledData,), method = 'L-BFGS-B',
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    b1 = opt_theta[encoder.limit2 : encoder.limit3].reshape(hidden_size, 1)
    """ Visualize the obtained optimal W1 weights """
    #sparseAutoencoder.visualizeW1(opt_W1, vis_patch_side, hid_patch_side)
##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#
#  You need to complete the code in feedForwardAutoencoder.m so that the
#  following command will extract features from the data.
num_train = numpy.round(len(labeledSet[0]) / 2)
indices = labeledSet[0]

#shuffle(indices)
#这里并没有对所有的数据进行随机，只是0-4的数据一半用来训练，一半用来测试
train_set = indices[0:num_train]
test_set = indices[num_train:]

train_data = mnistData[:,train_set]
train_labels = mnistLabels[train_set]

test_data = mnistData[:,test_set]
test_labels = mnistLabels[test_set]

train_features = sigmoid(numpy.dot(opt_W1,train_data) + b1)
test_features =  sigmoid(numpy.dot(opt_W1,test_data) + b1)

##======================================================================
## STEP 4: Train the softmax classifier
#  Use softmaxTrain.m from the previous exercise to train a multi-class
#  classifier.

#  Use lambda = 1e-4 for the weight regularization for softmax

# You need to compute softmaxModel using softmaxTrain on trainFeatures and
# trainLabels
num_classes = 5
decay_lambda = 1e-4
maxiter = 100
#此时特征变为196个
input_size = 196
regressor = softmax.SoftmaxRegression(input_size, num_classes, decay_lambda)
""" Run the L-BFGS algorithm to get the optimal parameter values """

soft_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                            args = (train_features, train_labels,), method = 'L-BFGS-B',
                                            jac = True, options = {'maxiter': maxiter})
soft_theta     = soft_solution .x

""" Obtain predictions from the trained model """
predictions = regressor.softmaxPredict(soft_theta, test_features)

""" Print accuracy of the trained model """
correct = test_labels[:, 0] == predictions[:, 0]

print ("""Accuracy :""", numpy.mean(correct))