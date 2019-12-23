import numpy as np
from optimizer.basicOptimizer import BasicOptimizer
from utils.norm import vector_2norm
from utils.matrixEquation import ut_equation
from utils.matrixFraction import modify_cholesky_fraction
from utils.matrixFraction import bunch_parlett_fraction


class NewtonMethod(BasicOptimizer):
    """
    Note
    """
    def __init__(self, step_optimizer=None, max_error=1e-8, max_iter=1000):
        """

        :param step_optimizer:
        :param max_error:
        :param max_iter:
        :param opt:
        """
        super().__init__(step_optimizer, max_error, max_iter)

    def compute(self, f, g, gg, x0):
        """

        :param f:
        :param g:
        :param gg:
        :param x0:
        :return:
        """
        while True:
            d = self._get_descent_direction(f, g, gg, x0)
            alpha = 1 if self.step_optimizer is None else self.step_optimizer.get_step_length(f, g, x0, d)
            x1 = x0 + alpha * d
            self._iter_increment()
            '''
            self.f_val.append(f(x0))
            self.d_val.append(d)
            self.g_val.append(g(x0))
            self.alpha.append(alpha)
            print("the minimum eigenvalue is {}".format(np.min(np.linalg.eigvals(gg(x1)))))
            if self.iter_num % 100 == 0:
                print("iter_num:{}, f is {}, g is {}, x is {}".format(self.iter_num, f(x1), g(x1), x1))
            '''
            if self._maximum_loop() is True or self._convergence(f, x0, x1) is True:
                '''
                self.f_val.append(f(x1))
                self.d_val.append(d)
                self.g_val.append(g(x1))
                self.alpha.append(alpha)
                print("d is \n"
                      "{}\n"
                      "alpha is:{}".format(d, alpha))
                print("iter_num:{}\n"
                      "x is {}, f(x) is {},\n"
                      "delta is:{}".format(self.iter_num, x1, f(x1), np.abs(f(x0)-f(x1))))
                '''
                break
            x0 = x1
        return x1

    def _get_descent_direction(self, f, g, gg, x0):
        try:
            return -1*np.dot(np.linalg.inv(gg(x0)), g(x0))
        except BaseException:
            #print("x0 is {}, {}".format(x0, gg(x0)))
            raise ValueError("Hessian Matrix can not be singular.")


class GillMurrayNewton(BasicOptimizer):
    """
    GillMurrayNewton method
        1: init x0, epsilon>0, k=1
        2：use modified cholesky fraction get G_k
            G_k + E_k = L_k D_k L_k^T
        3: if ||g_k|| >= epsilon:
            L_k D_k L_k^T d = - g_k
            solve this equation to get d_k
            then go to step 5
        4:  compute direction of negative curvature
            if psi_t >= 0:  stop
            else:   dk = -dk if g_k^T d_k > 0 else dk
        5: get alpha_k, x_{k+1} = x_{k} + alpha_k * d_k
        6 if not stop, go to step 2
    """

    def __init__(self, step_optimizer=None, max_error=1e-18, max_iter=1000, fit_decimal=False):
        """

        :param step_optimizer:
        :param max_error:
        :param max_iter:
        :param fit_decimal:
        :param opt:
        """
        super().__init__(step_optimizer, max_error, max_iter)
        self.fit_decimal = fit_decimal

    def compute(self, f, g, gg, x0, epsilon=1e-8):
        while True:
            d = self._get_descent_direction(f, g, gg, x0, epsilon)
            alpha = 1 if self.step_optimizer is None else self.step_optimizer.get_step_length(f, g, x0, d)
            x1 = x0 + alpha*d
            '''
            #
            self.f_val.append(f(x0))
            self.d_val.append(d)
            self.g_val.append(g(x0))
            self.alpha.append(alpha)
            '''
            self._iter_increment()
            if self.iter_num % 100 == 0:
                pass
                '''
                print("iter_num:{}\n"
                      "f is \n{}\n"
                      "d is \n{}\n"
                      "x is \n{}\n"
                      "alpha is\n{}\n"
                      "f(x0)-f(x1) is {}".format(self.iter_num, f(x1), d, x1, alpha, f(x0) - f(x1)))
                '''
            if self._maximum_loop() is True or self._convergence(f, x0, x1) is True:
                '''
                self.f_val.append(f(x1))
                self.d_val.append(d)
                self.g_val.append(g(x1))
                self.alpha.append(alpha)
                print("iter_num:{}\n"
                      "f is \n{}\n"
                      "d is \n{}\n"
                      "x is \n{}\n"
                      "alpha is\n{}".format(self.iter_num, f(x1), d, x1, alpha))
                '''
                break
            x0 = x1
        return x1

    def _get_descent_direction(self, f, g, gg, x0, epsilon):
        L, D = modify_cholesky_fraction(gg(x0))
        """
        print("L is {}\n"
              "D is {}".format(L, D))
        print("gg(x0) is {}".format(gg(x0)))
        """
        # print("LDL^T is {}".format(np.dot(np.dot(L, D), L.T)-gg(x0)))
        if vector_2norm(g(x0)) > epsilon:
            #   d = LDL_equation(L, D, -g(x0))
            G = np.dot(np.dot(L, D), L.T)
            d = -np.dot(np.linalg.inv(G), g(x0))
        else:
            psi = np.diag(2*gg(x0)-np.dot(np.dot(L, D), L.T))
            index = np.argmin(psi)
            #   print("psi min is {}".format(np.min(psi)))
            if psi[index] > 0:
                return np.zeros(x0.shape)
            y = np.zeros(x0.shape)
            y[index][0] = 1
            d = ut_equation(L.T, y)
            dd = np.dot(np.linalg.inv(L.T), y)
            if np.dot(g(x0).T, dd) > 0:
                d = -dd
        return d


class FletcherFreemanMethod(BasicOptimizer):
    """

    """
    def __init__(self, step_optimizer=None, max_error=1e-18, max_iter=1000):
        super().__init__(step_optimizer, max_error, max_iter)
        self.d_tag = 0

    def compute(self, f, g, gg, x0):
        """

        :param f:
        :param g:
        :param gg:
        :param x0:
        :return:
        """
        while True:
            d = self._get_descent_direction(f, g, gg, x0)
            alpha = 1 if self.step_optimizer is None else self.step_optimizer.get_step_length(f, g, x0, d)
            x1 = x0 + alpha*d
            '''
            self.f_val.append(f(x0))
            self.d_val.append(d)
            self.g_val.append(g(x0))
            self.alpha.append(alpha)
            '''
            self._iter_increment()
            if self.iter_num % 100 == 0:
                pass
                '''
                print("iter_num:{}\n"
                      "f is \n{}\n"
                      "d is \n{}\n"
                      "x is \n{}\n"
                      "alpha is\n{}\n"
                      "f(x0)-f(x1) is {}".format(self.iter_num, f(x1), d, x1, alpha, f(x0)-f(x1)))
                '''
            if self._maximum_loop() is True or self._convergence(f, x0, x1) is True:
                '''
                self.f_val.append(f(x1))
                self.d_val.append(d)
                self.g_val.append(g(x1))
                self.alpha.append(alpha)
                print("iter_num:{}\n"
                      "f is \n{}\n"
                      "d is \n{}\n"
                      "x is \n{}\n"
                      "alpha is\n{}".format(self.iter_num, f(x1), d, x1, alpha))
                '''
                break
            x0 = x1
        return x1

    def _get_descent_direction(self, f, g, gg, x0):
        L, D, P = bunch_parlett_fraction(gg(x0))
        min_eigval = np.min(np.linalg.eigvals(D))
        """
        print("D is {}".format(D))
        print("L is {}\n"
              " P is {}".format(L, P))
        """
        #   all eigenvalue is positive
        if min_eigval > 1e-8:
            return -np.dot(np.linalg.inv(gg(x0)), g(x0))

        # has negative eigenvalue
        elif min_eigval < -1e-8:
            self.d_tag = 1
        if self.d_tag == 0:
            #   construct the a
            #
            a = np.zeros(x0.shape)
            m = 0
            while m != D.shape[0]:
                if m == D.shape[0] - 1:
                    a[m][0] = 1 if D[m][m] <= 0 else 0
                    break
                if np.abs(D[m][m + 1]) > 1e-15:
                    # 2x2 block
                    tmp = D[m:m + 2, m:m + 2]
                    eigval, eigvec = np.linalg.eig(tmp)
                    index = np.argmin(eigval)
                    if eigval[index] > 0:
                        raise ValueError("negative value")
                    try:
                        a[m:m + 2][0] = tmp[:, index] / (np.sqrt(np.sum(tmp[:, index] ** 2)))
                    except:
                        print(index)
                    m = m + 2
                else:
                    if D[m][m] <= 0:
                        a[m][0] = 1
                    m = m + 1

            a = ut_equation(L.T, a)
            d = np.dot(P, a)
            if np.dot(g(x0).T, d) > 0:
                d = -d
            self.d_tag = 1
        else:
            #   construct the positive D
            m = 0
            Dpp = np.zeros(D.shape)
            while m != D.shape[0]:
                if m == D.shape[0] - 1:
                    Dpp[m][m] = 1/D[m][m] if D[m][m] > 0 else 0
                    break
                if np.abs(D[m][m + 1]) > 1e-15:
                    # 2x2 block
                    tmp = D[m:m + 2, m:m + 2]
                    eigval, eigvec = np.linalg.eig(tmp)
                    index = np.argmax(eigval)
                    if eigval[index] <= 0:
                        raise ValueError("negative value")
                    try:
                        Dpp[m:m + 2][m:m+2] = 1/eigval[index] * np.dot(tmp[:, index], tmp[:, index].T)
                    except:
                        print(index)
                    m = m + 2
                else:
                    # 1x1 block
                    if D[m][m] > 0:
                        Dpp[m][m] = 1/D[m][m]
                    m = m + 1
            self.d_tag = 0

            lp = np.dot(P, np.linalg.inv(L).T)
            d = -np.dot(np.dot(lp, Dpp), lp.T)
            d = np.dot(d, g(x0))
        if vector_2norm(d) < 1e-8:
            # if d equals to zero
            # compute LDL d = 0,
            _, _, v = np.linalg.svd(np.dot(np.dot(L, D), L.T))
            print(np.dot(gg(x0), v[-1, :].reshape(-1, 1)))
            if np.dot(v[-1, :], g(x0)) > 0:
                return -v[-1, :].reshape(-1, 1)*10
            else:
                return v[-1, :].reshape(-1, 1)*10
        return d
