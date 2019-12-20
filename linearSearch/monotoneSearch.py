from linearSearch.linearSearch import LinearSearch
import numpy as np


class AccurateLinearSearch(LinearSearch):
    """
    This class is used to find a suitable search area.

    """
    def __init__(self, method="GoldenRate", find_area_method="GoAndBack", max_iter=100, **opt):
        """
        :Note
        This is an init our class object by selecting search method and area-search method.

        :param method: a string of accurate linear search method, including "GoldenRate",
        :param find_area_method: a string of area search method, including "GoAndBack"
        :param max_iter: a int num represents the maximum iter num we can tolerate
        :param opt: some other including parameter
        """
        if method not in ["GoldenRate"]:
            raise NameError("{} is not a suitable method".format(method))
        super().__init__(method, max_iter, **opt)
        self.find_area_method = find_area_method

    def get_step_length(self, f, g, x, d):
        """
        :Note
        get step length

        :param f: object function
        :param g: one-order function
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)

        :return alpha: step length. float value
        """
        a, b = self._get_search_area(f, x, d)
        return self._get_step_length(f, x, d, a, b)

    def _get_search_area(self, f, x, d):
        """
        :Note
        In accurate linear search method, we need get search area.
        "GoAndBack"

        :param f: object function
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)

        :return (a,b): search area. tuple of float
        """
        if self.find_area_method is "GoAndBack":
            return self._go_and_back(f, x, d)
        else:
            raise NameError("{} is not a legal way to find search area.".format(self.find_area_method))

    def _get_step_length(self, f, x, d, a, b):
        """
        :Note
        After search area be determined.
        "GoldenRate"

        :param f: object function
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param a: left bound
        :param b: right bound

        :return alpha: the
        """
        if self.method is "GoldenRate":
            return self._golden_rate(f, x, d, a, b)
        else:
            raise NameError("{} is a illegal method".format(self.method))

    def _golden_rate(self, f, x, d, a, b, threshold=1e-8):
        """

        :param f:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param a:
        :param b:
        :param threshold:

        :return:
        """
        if "GoldenRateThreshold" in self.opt:
            threshold = self.opt["GoldenRateThreshold"]
        while abs(b-a) > threshold:

            self._global_iter_increment()
            al = a + 0.382*(b-a)
            ar = a + 0.618*(b-a)
            if f(x+al*d) < f(x+ar*d):
                a, b = a, ar
            else:
                a, b = al, b
        return 0.5*(a + b)

    def _go_and_back(self, f, x, d, alpha=0, eta=1e-3, t=2):
        """
        :Note
        :param f: object function
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha: the basic alpha, float with default value 0
        :param eta: the alpha_add each interaction, float with default value 1e-3
        :param t: the scale times, float with default value 2

        :return: (a, b) the suitable search area, tuple of float
        """
        if 't' in self.opt:
            t = self.opt['GoAndBack_t']
        if 'eta' in self.opt:
            eta = self.opt['GoAndBack_eta']
        if 'alpha' in self.opt:
            alpha = self.opt['GoAndBack_alpha']
        a = -1
        for i in range(int(self.max_iter)):
            alpha_new = alpha + eta
            if alpha_new <= 0:
                alpha_new = 0
            elif f(x+alpha_new*d) >= f(x+alpha*d):
                pass
            else:
                eta = t*eta
                a = alpha
                alpha = alpha_new
                continue
            if i is 0:
                eta = -1 * eta
                alpha = alpha_new
            else:
                if a == -1:
                    raise ValueError("the parameter setting is wrong. so that the function failed.")
                else:
                    return min(a, alpha_new), max(a, alpha_new)
        return min(a, alpha_new), max(a, alpha_new)


class InaccurateLinearSearch(LinearSearch):
    """

    """
    def __init__(self, method="BackTracking", condition="GoldStein", max_iter=100, **opt):
        """

        :param method:
        :param condition:
        :param max_iter:
        :param opt:
        """
        super().__init__(method, max_iter, **opt)
        self.condition = condition
        self.name = method+condition
        if method is "BackTracking":
            self.theta = opt["BackTracking_theta"]

    def get_step_length(self, f, g, x, d, alpha=1):
        if np.dot(g(x).T, d) > 0:
            print("np.dot(g(x).T, d) :{}".format(np.dot(g(x).T, d)))
            raise ValueError("g^T d must be negative.")
        i = 0
        self._global_iter_increment()
        while i < self.max_iter and not self._check_condition(f, g, x, d, alpha):
            self._global_iter_increment()
            alpha = self._get_step_length(f, g, x, d, alpha)
            i = i + 1
        #   if i >= self.max_iter:
        #   raise ValueError("{} can't find suitable step length, please check the parameter.".format(self.method))
        #   print("alpha is {}".format(alpha))
        return alpha

    def _get_step_length(self, f, g, x, d, alpha):
        """

        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return:
        """
        if self.method == "Interp22":
            return self._interp22(f, g, x, d, alpha)
        elif self.method == "Interp33":
            return self._interp33(f, g, x, d, alpha)
        elif self.method == "BackTracking":
            return self._backtracking(alpha)
        else:
            raise NameError("{} is not a suitable method.".format(self.method))

    def _backtracking(self, alpha):
        return alpha * self.theta

    def _interp33(self, f, g, x, d, alpha):
        """
        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return:
        """
        alpha1 = self._interp22(f, g, x, d, alpha)
        return self._interp33_help(f, g, x, d, alpha, alpha1)

    def _interp33_help(self, f, g, x, d, alpha, alpha1):
        """

        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :param alpha1:
        :return:
        """
        c = np.dot(g(x).T, d)[0][0]
        left = np.array([alpha ** 2, -alpha1 ** 2, -alpha ** 3, alpha1 ** 3]).reshape(2, 2)
        right = np.array([f(x + alpha1 * d) - f(x) - alpha1 * (np.dot(g(x).T, d))[0][0],
                          f(x + alpha * d) - f(x) - alpha * (np.dot(g(x).T, d))[0][0]]).reshape(2, 1)
        # print("left is {}".format(left))
        # print("right is {}".format(right))
        # print("left*right is {}".format(np.dot(left, right)))
        # print(alpha1)
        # print(alpha)
        if alpha1 == 0:
            return alpha1
        tmp = 1 / (alpha1 ** 2 * alpha ** 2 * (alpha1 - alpha)) * np.dot(left, right)
        a, b = tmp[0][0], tmp[1][0]
        print("a is {}, b is {}, tmp is {}".format(a, b, tmp))
        return (-b + np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)

    def _interp22(self, f, g, x, d, alpha):
        """
        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return:
        """
        p = -1 * (np.dot(g(x).T, d))[0][0] * alpha ** 2
        q = 2 * (f(x + alpha * d) - f(x) - (np.dot(g(x).T, d))[0][0] * alpha)
        # print("f(x+alpha*d) is {}\n"
        #      "f(x) is {}\n"
        #      "np.dot()*alpha is{}".format(f(x+alpha*d), f(x), alpha*np.dot(g(x).T, d)))
        # print("q is :{}".format(q))
        if q == 0:
            raise ValueError("the num was divided by zero in interp22 method ")
        # print(p/q)
        return float(p / q)

    def _check_condition(self, f, g, x, d, alpha):
        """

        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return:
        """
        if self.condition == "GoldStein":
            g1, g2 = self._check_goldstein(f, g, x, d, alpha)
            return g1 and g2
        elif self.condition == "Wolfe":
            g1, g2 = self._check_wolfe(f, g, x, d, alpha)
            return g1 and g2
        elif self.condition == "WolfePower":
            g1, g2 = self._check_wolfe_power(f, g, x, d, alpha)
            return g1 and g2
        else:
            raise NameError("{} is illegal.".format(self.condition))

    def _check_wolfe_power(self, f, g, x, d, alpha):
        """
        :Note
        The WolfePower condition is:
        f(x + alpha * d) <= f(x) + rho * alpha * g(x)^T \cdot d
        |g(x + alpha * d)^T \cdot d| >= -sigma * g(x)^T \cdot d
        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return (g1, g2):
        """
        g1, g2 = self._check_wolfe(f, g, x, d, alpha)
        g1 = g1 and g2
        g2 = (np.dot(g(x + alpha * d).T, d) <= -1 * self.opt["Wolfe_sigma"] * np.dot(g(x).T, d))[0][0]
        return g1, g2

    def _check_wolfe(self, f, g, x, d, alpha):
        """
        :Note
        The Wolfe condition is:
        f(x + alpha * d) <= f(x) + rho * alpha * g(x)^T \cdot d
        g(x + alpha * d)^T \cdot d > sigma * g(x)^T \cdot d
        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return (g1, g2):
        """
        if "Wolfe_rho" not in self.opt:
            raise NameError("parameter rho is necessary in Wolfe.")
        if "Wolfe_sigma" not in self.opt:
            raise NameError("parameter sigma is necessary in Wolfe.")
        rho, sigma = self.opt["Wolfe_rho"], self.opt["Wolfe_sigma"]
        if 0 >= rho or rho >= sigma or sigma >= 1:
            raise ValueError("rho:{}, sigma{}, must satisfy 0<rho<sigma<1".format(rho, sigma))

        g1 = (f(x + alpha * d) <= f(x) + rho * alpha * np.dot(g(x).T, d))[0][0]
        g2 = (np.dot(g(x + alpha * d).T, d) >= sigma * np.dot(g(x).T, d))[0][0]
        return g1, g2

    def _check_goldstein(self, f, g, x, d, alpha):
        """
        :Note
        The GoldStein condition is:
        f(x+alpha*d) <= f(x) + rho*alpha*np.dot(g.T, d)
        f(x+alpha*d) >= f(x) + (1-rho)*alpha*np.dot(g.T, d)
        :param f:
        :param g:
        :param x: the start point, np.ndarray of shape (N, 1)
        :param d: the decent direction, np.ndarray of shape(N, 1)
        :param alpha:
        :return (g1, g2):
        """
        if "GoldStein_rho" not in self.opt:
            raise NameError("parameter rho is necessary in GoldStein")
        rho = self.opt["GoldStein_rho"]
        if rho >= 0.5 or rho <= 0:
            raise ValueError("the rho is {}, but it needs to be in (0, 1/2)".format(rho))

        g1 = (f(x + alpha * d) <= f(x) + rho * np.dot(g(x).T, d) * alpha)[0][0]
        g2 = (f(x + alpha * d) >= f(x) + (1 - rho) * np.dot(g(x).T, d) * alpha)[0][0]
        return g1, g2
