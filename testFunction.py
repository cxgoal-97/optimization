import numpy as np
import math
from utils.norm import vector_2norm


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
        gx = np.zeros(x.shape, np.float64)
        for i in range(self.m):
            gx = gx + self._df_i(x, i)
        return gx

    def gg(self, x):
        gxx = np.zeros((x.shape[0], x.shape[0]), np.float64)
        for i in range(self.m):
            gxx = gxx + self._ddf_i(x, i)
        return gxx

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

    def g(self, x):
        gx = np.zeros(x.shape, dtype='float64')
        for i in range(int(self.n / 4)):
            gx[4 * i] = 2 * x[4 * i][0] + 20 * x[4 * i + 1][0] + 40 * (x[4 * i][0] - x[4 * i + 3][0]) ** 3
            gx[4 * i + 1] = 20 * x[4 * i][0] + 200 * x[4 * i + 1][0] + 4 * (x[4 * i + 1][0] - 2 * x[4 * i + 2][0]) ** 3
            gx[4 * i + 2] = 10 * (x[4 * i + 2][0] - x[4 * i + 3][0]) - 8 * (x[4 * i + 1][0] - 2 * x[4 * i + 2][0]) ** 3
            gx[4 * i + 3] = -10 * (x[4 * i + 2][0] - x[4 * i + 3][0]) - 40 * (x[4 * i][0] - x[4 * i + 3][0]) ** 3
        return gx

    def gg(self, x):
        ggx = np.zeros([self.n, self.n], dtype= 'float64')
        for i in range(self.n):
            for j in range(self.n):
                if i % 4 == 0:
                    if j == i:
                        ggx[i, j] = 2 + 120 * (x[i][0] - x[i + 3][0]) ** 2
                    elif j == i + 1:
                        ggx[i, j] = 20
                    elif j == i + 3:
                        ggx[i, j] = -120 * (x[i][0] - x[j][0]) ** 2
                    else:
                        ggx[i, j] = 0
                elif i % 4 == 1:
                    if j == i - 1:
                        ggx[i, j] = 20
                    elif j == i:
                        ggx[i, j] = 200 + 12 * (x[i][0] - 2 * x[i + 1][0]) ** 2
                    elif j == i + 1:
                        ggx[i, j] = -24 * (x[i][0] - 2 * x[j][0]) ** 2
                    else:
                        ggx[i, j] = 0
                elif i % 4 == 2:
                    if j == i - 1:
                        ggx[i, j] = -24 * (x[j][0] - 2 * x[i][0]) ** 2
                    elif j == i:
                        ggx[i, j] = 10 + 48 * (x[i - 1][0] - 2 * x[i][0]) ** 2
                    elif j == i + 1:
                        ggx[i, j] = -10
                    else:
                        ggx[i, j] = 0
                else:
                    if j == i - 3:
                        ggx[i, j] = -120 * (x[j][0] - x[i][0]) ** 2
                    elif j == i - 1:
                        ggx[i, j] = -10
                    elif j == i :
                        ggx[i, j] = 120 * (x[i - 3][0] - x[i][0]) ** 2 + 10
                    else:
                        ggx[i, j] = 0
        return ggx

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


class PenaltyI(BasicFunction):
    def __init__(self, m, n):
        if m != n+1:
            raise ValueError("m must equal to n+1")
        super().__init__(m, n)
        self.eta = 1e-5

    def _r_i(self, x, i):
        if i != self.n:
            return np.sqrt(self.eta)*(x[i][0]-1)
        else:
            return vector_2norm(x)**2-0.25

    def _dr_i(self, x, i):
        if i != self.n:
            return self.eta*np.ones(x.shape)
        else:
            return 2*x

    def _ddr_i(self, x, i):
        if i != self.n:
            return np.zeros((x.shape[0], x.shape[0]))
        else:
            return 2*np.eye(x.shape[0])

    def g1(self, x):
        gx = np.zeros(x.shape, dtype='float64')
        h = np.sum(x**2)-0.25
        for i in range(self.n):
            gx[i] = 2 * self.eta * (x[i] - 1) + 2 * (2 * x[i]) * h
        return gx

    def g(self, x):
        # gf = np.zeros(x.shape, dtype= 'float64')
        x = x.flatten()
        gx = 2 * self.eta * (x - 1) + 4 * (x.T.dot(x) - 0.25) * x
        return gx.reshape(-1, 1)

    def gg1(self, x):
        # Gf = np.zeros([self.n, self.n], dtype= 'float64')
        h = x.T.dot(x)
        x = x.flatten()
        ggx = 2 * self.eta + 4 * (x.T.dot(x) - 0.25) + 8 * x.dot(x.T) + np.diag(-2 * self.eta - 4 * (h-0.25))
        return ggx

    def gg(self, x):
        gxx = np.zeros([self.n, self.n], dtype='float64')
        h = np.sum(x**2)-0.25
        for i in range(self.n):
            for j in range(self.n):
                if j == i:
                    gxx[i, j] = 2 * self.eta + 4 * h + 8 * (x[i][0] ** 2)
                else:
                    gxx[i, j] = 8 * x[i][0] * x[j][0]
        return gxx


class Trigonometric(BasicFunction):
    def __init__(self, m, n):
        if m != n:
            raise ValueError("m must equal to n")
        super().__init__(m, n)
        self.o = np.array([(i + 1) for i in range(n)])
        self.ii = np.arange(1, n + 1)

    def _r_i(self, x, i):
        return self.n-np.sum(np.cos(x))+(1+i)*(1-np.cos(x[i][0]))-np.sin(x[i][0])

    def _dr_i(self, x, i):
        tmp = np.zeros(x.shape)
        for j in range(x.shape[0]):
            if j != i:
                tmp[j][0] = np.sin(x[j][0])
            else:
                tmp[j][0] = (i+2)*np.sin(x[j][0])-np.cos(x[j][0])
        return tmp

    def _ddr_i(self, x, i):
        tmp = np.diag(np.cos(x).flatten())
        tmp[i][i] = tmp[i][i]+(i*np.sin(x[i][0])-np.cos(x[i][0]))
        return tmp

    def g(self, x):
        x = x.flatten()
        rhs = self.ii * (1 - np.cos(x)) + self.n - np.sin(x) - sum(np.cos(x))
        lhs = np.tile(2 * np.sin(x), (self.n, 1)).T
        lhs = lhs + np.diag(2 * self.ii * np.sin(x) - 2 * np.cos(x))
        return lhs.dot(rhs).reshape(-1, 1)

    def gg(self, x):
        x = x.flatten()
        lhs1 = np.tile(2 * np.sin(x), (self.n, 1)).T
        lhs1 = lhs1 + np.diag(2 * self.ii * np.sin(x) - 2 * np.cos(x))
        rhs1 = np.tile(np.sin(x), (self.n, 1)) + np.diag(self.ii * np.sin(x) - np.cos(x))

        lhs2 = np.tile(2 * np.cos(x), (self.n, 1)).T
        lhs2 = lhs2 + np.diag(2 * self.ii * np.cos(x) + 2 * np.sin(x))
        rhs2 = self.ii * (1 - np.cos(x)) + self.n - np.sin(x) - sum(np.cos(x))
        res = lhs2.dot(rhs2)
        return lhs1.dot(rhs1) + np.diag(res)

    def g1(self, x):
        gx = np.zeros(x.shape, dtype='float64')
        sum = 0
        for j in range(self.n):
            sum += np.cos(x[j])
        for i in range(self.n):
            for j in range(self.m):
                if i == j:
                    gr = (i + 2) * np.sin(x[i]) - np.cos(x[i])
                else:
                    gr = np.sin(x[i])
                r = self.n - sum + (j + 1) * (1 - np.cos(x[j])) - np.sin(x[j])
                gx[i] += 2 * r * gr
        return gx
    '''
    def gg(self, x):
        #print(time.time())
        ggx = np.zeros([self.n, self.n], dtype='float64')
        sum = np.sum(np.cos(x))
        #for j in range(self.n):
        #    sum += np.cos(x[j][0])

        for i in range(self.n):
            for j in range(self.m):
                if j == i:
                    sum1 = 0
                    b = np.cos(x[i][0])
                    for h in range(self.n):
                        if h == i:
                            r = 0
                        else:
                            r = self.n - sum + (h + 1) * (1 - np.cos(x[h][0])) - np.sin(x[h][0])
                        sum1 += 2 * b * r
                    a = np.sin(x[i])
                    ggx[i, j] = sum1 + 2 * ((i + 2) * a - b) ** 2 + 2 * (self.n - sum + (i + 1) * (1 - b) - a) * (
                                (i + 2) * b + a) + 2 * (self.n - 1) * a ** 2
                else:
                    a = np.sin(x[i][0])
                    b = np.sin(x[j][0])
                    ggx[i, j] = 2 * a * ((j + self.n) * b - np.cos(x[j][0])) + 2 * ((i + 2) * a - np.cos(x[i][0])) * b
        #print(time.time())
        return ggx
        
    '''

class ExtendedRosenbrock(BasicFunction):
    def __init__(self, m, n):
        if n % 2 != 0 or m != n:
            raise ValueError("m must equal to n and n must be odd")
        super().__init__(m, n)

    def _r_i(self, x, i):
        if i % 2 == 0:
            return 10*(x[i+1][0]-x[i][0]**2)
        else:
            return 1-x[i-1][0]

    def _dr_i(self, x, i):
        tmp = np.zeros(x.shape)
        if i % 2 == 0:
            tmp[i][0] = -20*x[i][0]
            tmp[i+1][0] = 10
        else:
            tmp[i-1][0] = -1
        return tmp

    def _ddr_i(self, x, i):
        tmp = np.zeros((x.shape[0], x.shape[0]))
        if i % 2 == 0:
            tmp[i][i] = -20
        return tmp

    def g(self, x):
        gx = np.zeros(x.shape)#, dtype= 'float64')
        for i in range(int(self.n / 2)):
            gx[2 * i] = -400 * x[2 * i] * (x[2 * i + 1] - x[2 * i] ** 2) - 2 + 2 * x[2 * i]
            gx[2 * i + 1] = 200 * (x[2 * i + 1] - x[2 * i] ** 2)
        return gx

    def gg(self, x):
        ggx = np.zeros([self.n, self.n], dtype='float64')
        for i in range(self.n):
            for j in range(self.n):
                if i % 2 == 0:
                    if i == j:
                        ggx[i, j] = -400 * x[i + 1][0] + 1200 * x[i][0] ** 2 + 2
                    elif j == i + 1:
                        ggx[i, j] = -400 * x[i][0]
                    else:
                        ggx[i, j] = 0
                else:
                    if i == j:
                        ggx[i, j] = 200
                    elif j == i - 1:
                        ggx[i, j] = -400 * x[j][0]
                    else:
                        ggx[i, j] = 0
        return ggx
