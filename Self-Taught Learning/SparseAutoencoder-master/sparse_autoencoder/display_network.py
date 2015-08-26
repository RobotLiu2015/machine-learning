import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from math import sqrt, ceil, floor


def display_network(data, cols=-1, opt_normalize=True, opt_graycolor=True, save_figure_path=None):
    # This function visualizes filters in matrix A. Each row of A is a
    # filter. We will reshape each row into a square image and visualizes
    # on each cell of the visualization panel.
    # All other parameters are optional, usually you do not need to worry
    # about it.
    # opt_normalize: whether we need to normalize the filter so that all of
    # them can have similar contrast. Default value is true.
    # opt_graycolor: whether we use gray as the heat map. Default is true.
    # cols: how many rows are there in the display. Default value is the
    # squareroot of the number of rows in A.

    # rescale
    data -= np.mean(data)

    # compute rows, cols
    num, area = data.shape
    sz = int(sqrt(area))
    buf = 1
    if cols < 0:
        if floor(sqrt(num)) ** 2 != num:
            n = ceil(sqrt(num))
            while num % n != 0 and n < 1.2 * sqrt(num):
                n += 1
                m = ceil(num / n)
        else:
            n = sqrt(num)
            m = n
    else:
        n = cols
        m = ceil(num / n)
    n = int(n)
    m = int(m)

    array = -np.ones((buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        array *= 0.1

    k = 0
    for i in xrange(m):
        for j in xrange(n):
            if k >= num:
                continue
            if opt_normalize:
                clim = np.amax(np.absolute(data[k, :]))
            else:
                clim = np.amax(np.absolute(data))
            array[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
            buf + j * (sz + buf):buf + j * (sz + buf) + sz] = data[k, :].reshape([sz, sz]) / clim
            k += 1

    # simulate imagesc
    ax = plt.figure().gca()
    pix_width = 5
    h, w = array.shape
    exts = (0, pix_width * w, 0, pix_width * h)
    if opt_graycolor:
        ax.imshow(array, interpolation='nearest', extent=exts, cmap=cm.gray)
    else:
        ax.imshow(array, interpolation='nearest', extent=exts)

    plt.axis('off')

    if save_figure_path:
        plt.savefig(save_figure_path)

    plt.show()