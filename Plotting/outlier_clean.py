
import numpy as np
import matplotlib.pyplot as plt
import sys



def is_outlier(points, threshold=3.5):
    """
    This returns a boolean array with "True" if points are outliers and "False" otherwise
    These are the data points with a modified z-score greater than this:
    # value will be classified as outliers
    :param points:
    :param threshold:
    :return:
    """

    # transfrom the vector to rows, no column
    if len(points.shape) == 1:
        points = points[:, None]

    # compute median value
    # axis = 0 means along the column
    median = np.median(points, axis=0)

    # compute diff sums along the axis
    # axis = 1 means working along the row
    diff = np.sum((points - median) ** 2, axis=1)
    diff = np.sqrt(diff)

    # compute MAD
    med_abs_deviation = np.median(diff)

    # compute modified z-score
    modified_z_score = 0.6745 * diff / med_abs_deviation

    # return a mask for each outlier
    return modified_z_score > threshold


if __name__ == "__main__":
    # Random data
    x = np.random.random(100)

    # histogram buckets, bin_size
    buckets = 50

    # Add in a few outliers
    # np.r_ does row-wise merging
    # V = array([1, 2, 3, 4, 5, 6])
    # Y = array([7, 8, 9, 10, 11, 12])
    # np.r_[V[0:2], Y[0], V[3], Y[1:3], V[4:], Y[4:]]
    # array([1, 2, 7, 4, 8, 9, 5, 6, 11, 12])
    x = np.r_[x, -49, 95, 100, -100]

    # Keep valid data points
    # print(is_outlier(x))
    # "~" is logical NOT on boolean numpy arrays
    filtered = x[~is_outlier(x)]

    # plot histogram
    plt.figure()
    plt.subplot(211)
    plt.hist(x, buckets)
    plt.xlabel('Raw')


    plt.subplot(212)
    plt.hist(filtered, buckets)
    plt.xlabel("Cleaned")

    plt.show()

