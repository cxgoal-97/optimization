import numpy as np
from optimizer.basicOptimizer import BasicOptimizer
from scipy.sparse.linalg import gmres
from utils.norm import vector_2norm
from scipy.optimize import fmin, fminbound


class InaccurateNewton(BasicOptimizer):
    def __init__(self, step_optimizer=None, max_error=1e-5, max_iter=3e3,
                 theta_min=0.1, theta_max=0.5, eta=0.5, t=1e-2, eta_method="1"):
        super().__init__(step_optimizer=step_optimizer, max_error=max_error, max_iter=max_iter)
        self.theta_min, self.theta_max = theta_min, theta_max
        self.eta, self.t, self.eta_method = eta, t, eta_method

    def get_theta(self, gx, ggx, gx1, ggx1, dk):
        theta_min = 0.1
        theta_max = 0.5
        a = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 1]])
        b = np.array([np.linalg.norm(gx) * np.linalg.norm(gx),
                      2 * dk.T.dot(ggx1).dot(gx1),
                      np.linalg.norm(gx1) * np.linalg.norm(gx1)])
        coff = np.linalg.solve(a, b)

        def f_quad(x):
            return coff[0] * x ** 2 + coff[1] * x + coff[2]

        min_global = fminbound(f_quad, theta_min, theta_max)
        return min_global

    def compute(self, f, g, gg, x0):
        self.iter_num = 0
        fx0 = f(x0)
        gx0 = g(x0)
        ggx0 = gg(x0)
        fx1 = fx0
        gx1 = gx0
        ggx1 = ggx0
        while True:
            print("{} 开始".format(self.iter_num))
            fx0 = fx1
            gx0 = gx1
            ggx0 = ggx1
            print("{} 开始球d ".format(self.iter_num))
            d = self._get_descent_direction(ggx0, gx0)
            '''
            if vector_2norm((np.dot(ggx0, d)+gx0).flatten()) > self.eta * vector_2norm(gx0.flatten()):
                print("{} f(x0) is {}".format(self.iter_num, fx0))
                raise ValueError("")
            '''
            # get suitable step length
            print("{} 开始更新 d".format(self.iter_num))
            eta = self.eta
            for i in range(2):
                x1 = x0 + d
                fx1, gx1, ggx1 = f(x1), g(x1), gg(x1)
                theta = self.get_theta(gx0, ggx0, gx1, ggx1, d)
                if vector_2norm(g(x0+d)) <= (1-self.t*(1-eta))*vector_2norm(gx0):
                    break
                d = theta * d
                print(i)
                eta = 1-theta*(1-eta)

            print("{} 开始更新参数 {}".format(self.iter_num, self.eta))
            self._update_eta(gx0, gx1, ggx0, d)
            print("iter_num is{}, f(x) is{} g(x) is {} x is {}".format(self.iter_num, fx1, vector_2norm(gx1), vector_2norm(x1)))
            if self._convergence(gx1, x1) is True or self._maximum_loop() is True:
                print("iter_num is{}, f(x) is{}".format(self.iter_num, fx1))
                break
            x0 = x1
            self._iter_increment()
        return x1

    def _update_eta(self, gx0, gx1, ggx0, d):
        if self.eta_method == "1":
            self.eta = vector_2norm(gx1-gx0-np.dot(ggx0, d))/vector_2norm(gx0)
        else:
            # delta \in (0, 1]
            # alpha \in (1, 2]
            self.eta = 0.5*(vector_2norm(gx1)/vector_2norm(gx0))**1.5

    def _get_descent_direction(self, ggx, gx):
        """

        :param ggx:
        :param gx:
        :return:
        """

        #d = gmres(ggx, -gx, maxiter=10)[0].reshape(-1, 1)
        d = gmres(ggx, (self.eta-1)*gx, atol=0, maxiter=5)[0].reshape(-1, 1)
        if np.dot(d.T, gx) > 0:
            d = -d
        return d

    def _convergence(self, gx, x):
        #   print("delta is{}, condition{}".format(np.abs(f(x0)-f(x1)), np.abs(f(x0)-f(x1)) < self.max_error))
        return True if vector_2norm(gx.flatten()) < self.max_error * np.max([1, vector_2norm(x.flatten())]) else False

