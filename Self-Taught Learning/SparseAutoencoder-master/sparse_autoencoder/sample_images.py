from random import randint

from scipy.io import loadmat
import numpy as np
# from matplotlib import pyplot as plt


def normalize_data(patches):
    """
    Squash data to [0.1, 0.9] since we use sigmoid as the activation
    function in the output layer
    """
    # Remove DC (mean of images).
    mean = np.mean(patches, axis=1)
    patches -= mean[:, np.newaxis]

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * np.std(patches, axis=1)
    pstd = pstd[:, np.newaxis]
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1

    return patches


def sample_images(patch_size, num_patches):
    """
    :return: 10000 patches for training
    """
    ## Get IMAGES.mat from http://ufldl.stanford.edu/wiki/resources/sparseae_exercise.zip
    images = loadmat('../data/IMAGES.mat')['IMAGES']  # load images from disk
    num_images = images.shape[2]

    # Initialize patches
    patches = np.empty([num_patches, patch_size * patch_size])

    for i in xrange(num_patches):
        # randomly pick an image
        image_idx = randint(0, num_images - 1)
        height, width = images[:, :, image_idx].shape

        # randomly pick a patch
        y = randint(0, height - patch_size)
        x = randint(0, width - patch_size)

        # copy data
        np.copyto(dst=patches[i, :], src=images[y:y + patch_size, x:x + patch_size, image_idx].reshape((1, 64)))

    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure
    # the range of pixel values is also bounded between [0,1]
    patches = normalize_data(patches)

    return patches
