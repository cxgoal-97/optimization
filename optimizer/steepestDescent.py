from optimizer.basicOptimizer import BasicOptimizer


class SteepestDescent(BasicOptimizer):
    """

    """

    def __init__(self,  step_optimizer=None, max_error=1e-6, max_iter=10000, **opt):
        """

        :param step_optimizer:
        :param max_error:
        :param max_iter:
        :param opt:
        """
        super().__init__(step_optimizer, max_error, max_iter, **opt)

    def _get_descent_direction(self, f, g, x0):
        """

        :param f:
        :param g:
        :param x0:
        :return:
        """
        return -g(x0)

