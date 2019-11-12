import numpy as np


def euclidean_distance(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    return np.dot((x-y).T, (x-y))