import numpy as np
import math
from linearSearch.linearSearch import LinearSearch


class NonmonotoneGLL(LinearSearch):
    """

    """
    def __init__(self, method="GLL", max_iter=100, **opt):
        """

        :param method:
        :param max_iter:
        :param opt:
        """
        super().__init__(method, max_iter, **opt)
        self.name = self.__class__.__name__
        self.old_value = []
        if "GLL_rho" not in self.opt or "GLL_alpha" not in self.opt\
                or "GLL_sigma" not in self.opt or "GLL_M" not in self.opt:
            raise NameError("GLL need rho, alpha, sigma, M")
        if self.opt["GLL_rho"] >= 1 or self.opt["GLL_rho"] <= 0:
            raise ValueError("rho must be in (0, 1)")
        else:
            self.rho = self.opt["GLL_rho"]

        if self.opt["GLL_alpha"] <= 0:
            raise ValueError("alpha must be positive")
        else:
            self.alpha = self.opt["GLL_alpha"]

        if self.opt["GLL_M"] <= 0:
            raise ValueError("M must be positive")
        else:
            self.M = self.opt["GLL_M"]

        if self.opt["GLL_sigma"] >= 1 or self.opt["GLL_sigma"] <= 0:
            raise ValueError("sigma must be in (0, 1)")
        else:
            self.sigma = self.opt["GLL_sigma"]

    def get_step_length(self, f, g, x, d):
        """
        :param f:
        :param g:
        :param x:
        :param d:
        :return:
        """
        if np.dot(g(x).T, d) > 0:
            print("np.dot(g(x).T, d) :{}".format(np.dot(g(x).T, d)))
            #   d = -d
            raise ValueError("g^T d must be negative.")
        residual = self.rho * np.dot(g(x).T, d)[0][0]
        self.old_value.append(f(x))
        k = len(self.old_value)-1
        '''
        for mk in range(min([k, self.M])+1):
            for t in range(1000):
                if f(x+alphak*d) <= np.max(self.old_value[k-mk:]) + alphak * residual:
                alphak = math.pow(self.sigma, t) * self.alpha
                    return np.max([alphak, 0.00001])
        '''
        mk = min([k, self.M])
        alphak = self.alpha
        for t in range(self.max_iter):
            self._global_iter_increment()
            try:
                if f(x + alphak * d) <= np.max(self.old_value[k - mk:]) + alphak * residual:
                    return alphak
            except:
                print("x is {}, alpha is {}, d is{}".format(x, alphak, d))
            alphak = alphak * self.sigma
        return alphak
        raise ValueError("can not find suitable alphak in GLL")