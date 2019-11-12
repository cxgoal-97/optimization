import numpy as np


def vector_2norm(x):
    """

    :param x:
    :return:
    """
    return np.sqrt(np.sum(np.dot(x.T, x)))


def vector_1norm(x):
    """

    :param x:
    :return:
    """
    return np.max(np.abs(x))


def matrix_1norm(a):
    """
    max_{j} sum_{i} |a_{ij}|
    :param a:
    :return:
    """
    return np.max(np.sum(np.abs(a), axis=0))


def matrix_inftynorm(a):
    """
    max_{i} sum_{j} |a_{ij}|
    :param a:
    :return:
    """
    return np.max(np.sum(np.abs(a), axis=1))
