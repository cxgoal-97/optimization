import numpy as np


def LDL_equation(L, D, y):
    """
    return the answer to LDL^T x = y

    :param L:
    :param D:
    :param y:
    :return:
    """
    y = lt_equation(np.dot(L, D), y)
    x = ut_equation(L.T, y)
    return x


def ut_equation(U, y):
    """
    U is a upper triangular matrix
    we can use this function to get x which satisfies Dx = y
    :param U:
    :param y:
    :return:
    """
    x = np.zeros(y.shape)
    for i in range(U.shape[-1] - 1, -1, -1):
        for j in range(U.shape[-1] - 1, i, -1):
            y[i][0] = y[i][0] - x[j][0] * U[i][j]
        if U[i][i] == 0:
            raise ValueError("U[{}][{}] can not be zero".format(i, i))
        x[i][0] = y[i][0]/U[i][i]
    return x


def lt_equation(L, y):
    """
    L is a lower triangular matrix,
    we can use this function to get x which satisfies Lx = y

    x_i = (y_i - \sigma_{j=1}^{i-1} x_j * L[i][j]) / L[i][i]

    :param L:
    :param y:
    :return:
    """
    x = np.zeros(y.shape)
    for i in range(L.shape[0]):
        for j in range(i):
            #   print("y[{}], y[{}], x[{}] L[{}] is".format(y[i], y[j], L[j], x[j]))
            y[i][0] = y[i][0] - L[i][j] * x[j][0]
        if L[i][i] == 0:
            raise ValueError("L[{}][{}] can not be zero".format(i, i))
        x[i][0] = y[i][0] / L[i][i]
    return x
