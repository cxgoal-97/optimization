class LinearSearch:
    """

    """
    def __init__(self,  method, max_iter=100, **opt):
        """

        :param method: the method of getting suitable step length
        :param max_iter: the maximum iter num we can tolerate
        :param opt: the suitable params
        """
        self.method = method
        self.max_iter = max_iter
        self._global_iter = 0
        self.opt = opt

    def get_step_length(self, f, g, x, d):
        raise NameError("virtual class shouldn't be used")

    def _check_condition(self, f, g, x, d, alpha):
        pass

    def _get_step_length(self, f, g, x, d, a, b):
        pass

    def _get_search_area(self, f, x, d):
        pass

    def _global_iter_increment(self):
        self._global_iter = self._global_iter+1

    def clear_iter(self):
        self._global_iter = 0

    def get_iter_num(self):
        return self._global_iter

