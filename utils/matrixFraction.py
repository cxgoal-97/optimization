from decimal import Decimal
import numpy as np


def cholesky_fraction(g, fit_decimal=False):
    """

    :param g:
    :param fit_decimal:
    :return:
    """
    l = np.eye(g.shape[0])
    d = np.zeros(g.shape)
    if fit_decimal is True:
        l, d, g = l.astype(np.object), d.astype(np.object), g.astype(np.object)
        for i in range(l.shape[0]):
            for j in range(l.shape[1]):
                l[i][j], d[i][j], g[i][j] = Decimal(l[i][j]), Decimal(d[i][j]), Decimal(g[i][j])
    for j in range(g.shape[0]):
        d[j][j] = g[j][j] - np.sum([l[j][k] ** 2 * d[k][k] for k in range(j)])
        for i in range(j + 1, g.shape[0]):
            l[i][j] = 1 / d[j][j] * (g[i][j] - (np.sum([l[i][k] * l[j][k] * d[k][k] for k in range(j)])))
    return l, d


def modify_cholesky_fraction(g, fit_decimal=False, machine_err=1e-15):
    """

    :param g:
    :param fit_decimal
    :param machine_err:
    :return:
    """
    l = np.eye(g.shape[0]).astype(np.float)
    d = np.zeros(g.shape, np.float)
    if fit_decimal is True:
        l, d, g = l.astype(np.object), d.astype(np.object), g.astype(np.object)
        for i in range(l.shape[0]):
            for j in range(l.shape[0]):
                l[i][j], d[i][j], g[i][j] = Decimal(l[i][j]), Decimal(d[i][j]), Decimal(g[i][j])
                machine_err = Decimal(machine_err)
    epsilon = np.max(np.abs(g-np.diag(np.diag(g))))
    gamma = np.max(np.abs(np.diag(g)))
    delta = machine_err*np.max([epsilon+gamma, 1])
    condition_num = np.sqrt(g.shape[0]**2 - 1)
    condition_num = Decimal(condition_num) if fit_decimal is True else condition_num
    beta = np.sqrt(np.max([gamma, epsilon/condition_num, machine_err]))

    '''
    print("epsilon:{}\n"
          "gamma:{}\n"
          "delta:{}\n"
          "beta:{}, {}".format(epsilon, gamma, delta, beta, beta**1))
    '''

    for j in range(g.shape[0]):
        d[j][j] = np.max([delta, np.abs(g[j][j] - np.sum([g[j][r]*l[j][r] for r in range(0, j)]))])
        for i in range(j+1, g.shape[0]):
            g[i][j] = g[i][j]-np.sum([l[j][r]*g[i][r] for r in range(0, j)])
        if j == g.shape[0]-1:
            theta = 0
        else:
            theta = np.max([np.abs(g[i][j]) for i in range(j+1, g.shape[0])])
        d[j][j] = np.max([d[j][j], theta**2/beta**2])
        for i in range(j+1, g.shape[0]):
            l[i][j] = g[i][j]/d[j][j]
            print("i is {}, j is {}, g[i][j] is {}, d[i][j] is{}".format(i, j, l[i][j], d[j][j]))

    return l, d


def bunch_parlett_fraction(a, magic_num=2/3):
    """

    :param a:
    :param magic_num:
    :return:
    """
    A = a.copy()
    n, m, k = A.shape[0], 0, 0
    y, L, D = np.arange(n), np.zeros(A.shape, np.float), np.zeros(A.shape, np.float)
    while m < n:
        #   find a_tt
        tt = np.argmax(np.abs(np.diag(A))[m:])+m
        att = A[tt][tt]
        #   find a_ls
        if m == n-1:
            als = 0
        else:
            tmp = np.argmax(np.abs(A-np.diag(np.diag(A))), axis=1)
            l = np.argmax(np.array([np.abs(A[i+m][tmp[i+m]])] for i in range(n-m)))
            try:
                als = A[l+m][tmp[l]]
            except:
                print(A)
                print(m, l, tmp[l], tmp)
                raise ValueError("")
            l, s = np.max([l+m, tmp[l]]), np.min([l+m, tmp[l]])
        if att == 0 and als == 0:
            break
        if np.abs(att) > magic_num * np.abs(als):
            #   print("1 block")
            #   1x1 block
            #   translate block
            A[[m, tt], :] = A[[tt, m], :]
            A[:, [m, tt]] = A[:, [tt, m]]
            y[[tt, m]] = y[[m, tt]]
            #   compute
            dmm = A[m, m]
            lm = (A[:, m] / dmm).reshape(-1, 1)
            A = A-dmm * np.dot(lm, lm.T)
            D[m][m] = dmm
            L[:, m] = lm.reshape(-1)
            L[[m, tt], :k] = L[[tt, m], :k]
            m = m+1

        else:
            #   print("2 block")
            #   2x2 block
            #   translate block
            A[[m, s, m+1, l], :] = A[[s, m, l, m+1], :]
            A[:, [m, s, m+1, l]] = A[:, [s, m, l, m+1]]
            y[[s, m, l, m+1]] = y[[m, s, m+1, l]]
            D1 = A[m: m+2, m: m+2]
            L1 = np.dot(A[:, m:m+2], np.linalg.inv(D1))
            A = A - np.dot(np.dot(L1, D1), L1.T)
            D[m:m+2, m:m+2] = D1
            L[:, m:m+2] = L1
            m = m+2
        k = k+1
    P = np.eye(n)
    P = P[:, y]
    """
    print("L is {}\n"
          "D is {}\n"
          "P is {}\n".format(L, D, P))
    """
    return L, D, P