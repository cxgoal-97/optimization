import numpy as np
import math


class BasicFunction:
    """

    """
    def __init__(self, m, n):
        """

        :param m:
        :param n:
        """
        self.m, self.n = m, n

    def f(self, x):
        return np.sum([self._f_i(x, i) for i in range(self.m)])

    def g(self, x):
        return np.sum([self._df_i(x, i) for i in range(self.m)], axis=0)

    def gg(self, x):
        return np.sum([self._ddf_i(x, i) for i in range(self.m)], axis=0)

    def _r_i(self, x, i):
        pass

    def _dr_i(self, x, i):
        pass

    def _ddr_i(self, x, i):
        pass

    def _f_i(self, x, i):
        return math.pow(self._r_i(x, i), 2)

    def _df_i(self, x, i):
        return 2*self._r_i(x, i)*self._dr_i(x, i)

    def _ddf_i(self, x, i):
        return 2*np.dot(self._dr_i(x, i), self._dr_i(x, i).T) + 2*self._r_i(x, i) * self._ddr_i(x, i)


class BiggsEXP6(BasicFunction):
    """

    """
    def __init__(self, m, n=6):
        """

        :param m:
        :param n:
        """
        if n != 6:
            raise ValueError("variable dimension must be 6")
        if m < n:
            raise ValueError("function num must be larger than variable dimension")
        super().__init__(m, n)
        self.t = [0.1*(i+1) for i in range(m)]
        self.y = [math.exp(-self.t[i])-5*math.exp(-10*self.t[i])+3*math.exp(-4*self.t[i]) for i in range(m)]

    def _r_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        return x[2][0]*math.exp(-self.t[i]*x[0][0])-x[3][0]*math.exp(-self.t[i]*x[1][0])+x[5][0]*math.exp(-self.t[i]*x[4][0])-self.y[i]

    def _dr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        return np.array([-self.t[i]*x[2][0]*math.exp(-self.t[i]*x[0][0]),
                         self.t[i]*x[3][0]*math.exp(-self.t[i]*x[1][0]),
                         math.exp(-self.t[i]*x[0][0]),
                         -math.exp(-self.t[i]*x[1][0]),
                         -self.t[i]*x[5][0]*math.exp(-self.t[i]*x[4][0]),
                         math.exp((-self.t[i]*x[4][0]))
                         ]).reshape(-1, 1)

    def _ddr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        ti = self.t[i]
        res = np.zeros((self.n, self.n), dtype=np.float)
        res[0][2] = -ti*math.exp(-ti*x[0][0])
        res[2][0] = -ti*math.exp(-ti*x[0][0])
        res[1][3] = ti*math.exp(-ti*x[1][0])
        res[3][1] = ti*math.exp(-ti*x[1][0])
        res[4][5] = -ti*math.exp(-ti*x[4][0])
        res[5][4] = -ti*math.exp(-ti*x[4][0])
        res[0][0] = math.pow(ti, 2)*x[2][0]*math.exp(-ti*x[0][0])
        res[1][1] = -math.pow(ti, 2)*x[3][0]*math.exp(-ti*x[1][0])
        res[4][4] = math.pow(ti, 2)*x[5][0]*math.exp(-ti*x[4][0])
        return res


class ExtendedPowellSingular(BasicFunction):
    """

    """
    def __init__(self, m, n):
        """

        :param m:
        :param n:
        """
        if n % 4 != 0:
            raise ValueError("n ")
        if m != n:
            raise ValueError("m must be equal to n")
        super().__init__(m, n)

    def _r_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        if i % 4 == 0:
            res = x[i][0]+10*x[i+1][0]
        elif i % 4 == 1:
            res = np.sqrt(5)*(x[i+1][0]-x[i+2][0])
        elif i % 4 == 2:
            res = math.pow(x[i-1][0]-2*x[i][0], 2)
        else:
            res = np.sqrt(10)*math.pow(x[i-3][0]-x[i][0], 2)
        return res

    def _dr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = np.zeros((self.n, 1), dtype=np.float)
        if i % 4 == 0:
            res[[i, i+1], 0] = [1, 10]
        elif i % 4 == 1:
            res[[i+1, i+2], 0] = [np.sqrt(5), -np.sqrt(5)]
        elif i % 4 == 2:
            res[[i-1, i], 0] = [2*(x[i-1][0]-2*x[i][0]), -4*(x[i-1][0]-2*x[i][0])]
        else:
            res[[i-3, i], 0] = [2*np.sqrt(10)*(x[i-3][0]-x[i][0]), -2*np.sqrt(10)*(x[i-3][0]-x[i][0])]
        return res

    def _ddr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = np.zeros((self.n, self.n), dtype=np.float)
        if i % 4 == 2:
            res[i-1][i-1] = 2
            res[i-1][i] = -4
            res[i][i-1] = -4
            res[i][i] = 8
        elif i % 4 == 3:
            res[i-3][i-3] = 2*np.sqrt(10)
            res[i-3][i] = -2*np.sqrt(10)
            res[i][i-3] = -2*np.sqrt(10)
            res[i][i] = 2*np.sqrt(10)
        else:
            pass
        return res


class PowellBadlyScaled(BasicFunction):
    """

    """
    def __init__(self, m=2, n=2):
        """

        :param m:
        :param n:
        """
        if m != 2 or n != 2:
            raise ValueError("m must be 2 and n must be 2")
        super().__init__(m, n)

    def _r_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = 0
        if i == 0:
            res = 1e4 * x[0][0] * x[1][0] - 1
        elif i == 1:
            res = math.exp(-x[0][0]) + math.exp(-x[1][0]) - 1.0001
        return res

    def _dr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = np.zeros((2, 1), dtype=np.float)
        if i == 0:
            res[:, 0] = [1e4*x[1][0], 1e4*x[0][0]]
        elif i == 1:
            res[:, 0] = [-math.exp(-x[0][0]), -math.exp(-x[1][0])]
        return res

    def _ddr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = np.zeros((2, 2), dtype=np.float)
        if i == 0:
            res[0][1] = 1e4
            res[1][0] = 1e4
        elif i == 1:
            res[0][0] = math.exp(-x[0][0])
            res[1][1] = math.exp(-x[1][0])
        return res


class Rosenbrock(BasicFunction):
    """

    """
    def __init__(self, m=2, n=2):
        """

        :param m:
        :param n:
        """
        super().__init__(m, n)

    def _r_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        if i == 0:
            return 10*(x[1][0]-math.pow(x[0][0], 2))
        else:
            return 1-x[0][0]

    def _dr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        if i == 0:
            return np.array([-20*x[0][0], 10]).reshape(-1, 1)
        else:
            return np.array([-1, 0]).reshape((-1, 1))

    def _ddr_i(self, x, i):
        """

        :param x:
        :param i:
        :return:
        """
        res = np.zeros((self.n, self.n), dtype=np.float)
        if i == 0:
            res[0][0] = -20
        else:
            pass
        return res
