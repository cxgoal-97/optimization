import numpy as np


class BasicOptimizer:
    """
    general steps:
    while not (convergence or max_loop):
        1. get descent direction d
        2. get step length alpha
        3. update x <- x + alpha * d
    """
    def __init__(self, step_optimizer=None, max_error=1e-8, max_iter=1e5):
        """

        :param step_optimizer:
        :param max_error:
        :param max_iter:
        :param opt:
        """
        self.step_optimizer, self.max_error, self.max_iter = step_optimizer, max_error, max_iter
        self.iter_num = 0
        self.d_val = []
        self.f_val = []
        self.g_val = []
        self.alpha = []

    def compute(self, f, g, x0):
        """

        :param f:
        :param g:
        :param x0:
        :return:
        """
        while True:
            d = self._get_descent_direction(f, g, x0)
            alpha = 1 if self.step_optimizer is None else self.step_optimizer.get_step_length(f, g, x0, d)
            x1 = x0 + alpha * d
            if self.iter_num % 100 == 0:
                print("iter_num:{}, f is {}, x1 is {}".format(self.iter_num, f(x1), x1))
            self._iter_increment()
            if self._maximum_loop() is True or self._convergence(f, x0, x1) is True:
                print("convergence {}:".format((f(x0)-f(x1))))
                break
            x0 = x1
        return x1

    def _iter_increment(self):
        """

        :return:
        """
        self.iter_num = self.iter_num + 1

    def _maximum_loop(self):
        """

        :return:
        """
        return True if self.iter_num > self.max_iter else False

    def _convergence(self, f, x0, x1):
        """

        :param f:
        :param x0:
        :param x1:
        :return:
        """
        return True if np.abs(f(x0)-f(x1)) < self.max_error else False

    def _get_descent_direction(self, f, g, x0):
        pass
