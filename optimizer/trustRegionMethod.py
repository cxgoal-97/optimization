import numpy as np
from optimizer.basicOptimizer import BasicOptimizer
from utils.norm import vector_2norm


class TrustRegion(BasicOptimizer):
    def __init__(self, subq_method="hebden", delta=0.1, max_error=1e-8, max_iter=1e3):
        """
        :Note init the

        :param subq_method: string, the method to be used for solving sub question
        :param delta: float, init radius
        :param max_error: float, maximum tolerance
        :param max_iter: int, maximum iter num
        """
        super().__init__(None, max_error=max_error, max_iter=max_iter)
        self.subq_method = subq_method
        self.delta = delta

    def compute(self, f, g, gg, x0):
        """
        :param f: object function
        :param g: one-order function
        :param gg: second-order function
        :param x0: the start point, np.ndarray of shape (N, 1)

        :return: x1: the convergence point, np.ndarray of shape (N, 1)
        """
        fx0, gx0, ggx0 = f(x0), g(x0), gg(x0)
        while True:
            d = self._get_descent_direction(fx0, gx0, ggx0)
            print(d)
            q_dk = self._q_dk(fx0, gx0, ggx0, d)
            x1 = x0 + d
            fx1 = f(x1)
            if fx0 == q_dk:
                return x1
            gamma = (fx0 - fx1) / (fx0 - q_dk)
            if gamma <= 0.25:
                self.delta = self.delta / 4
            elif gamma >= 0.75 and np.abs(vector_2norm(d) - self.delta) < 1e-2:
                self.delta = self.delta * 3
            else:
                pass

            if self._convergence(f, x0, x1) or self._maximum_loop():
                return x1

            if gamma > 0:
                x0, fx0 = x1, fx1
                gx0, ggx0 = g(x0), gg(x0)
            else:
                pass
            self._iter_increment()

    def _get_descent_direction(self, fx, gx, ggx):
        """
        :Note Get the descent direction

        :param fx: f(x), np.ndarray of shape (N, 1)
        :param gx: g(x), np.ndarray of shape (N, 1)
        :param ggx: gg(x), np.ndarray of shape (N, N)

        :return: d: the descent direction, np.ndarray of shape (N, 1)
        """
        if self.subq_method == "hebden":
            return self._hebden(fx, gx, ggx)
        elif self.subq_method == "cauthy":
            return self._cauthy(fx, gx, ggx)
        elif self.subq_method == "subspace":
            return self._subspace(fx, gx, ggx)
        else:
            raise NameError("{} not a suitable error.".format(self.subq_method))

    def _hebden(self, fx, gx, ggx):
        """
        :Note hebden method to solve sub question

        :param fx: f(x), np.ndarray of shape (N, 1)
        :param gx: g(x), np.ndarray of shape (N, 1)
        :param ggx: gg(x), np.ndarray of shape (N, N)

        :return: d: the descent direction, np.ndarray of shape (N, 1)
        """
        inv_ggx = np.linalg.inv(ggx)
        d = -np.dot(inv_ggx, gx)
        norm_d = vector_2norm(d)
        if norm_d <= self.delta:
            return d
        # find ||d(v)|| = delta_k
        v = 0
        #
        i = 0
        ori_norm_d = norm_d
        while np.abs(norm_d - self.delta) > 1e-2 * self.delta and norm_d / ori_norm_d > 5e-2 and i < 50:
            i = i + 1
            inv_ggx = np.linalg.inv((ggx + v * np.eye(ggx.shape[0])))
            d = -np.dot(inv_ggx, gx)
            norm_d = vector_2norm(d)
            partial_d = -1 * np.dot(inv_ggx, d)
            psi_v = norm_d - self.delta
            psi_d_v = np.dot(d.T, partial_d)
            v = v - (psi_v + self.delta) / self.delta * psi_v / psi_d_v
        return d

    def _cauthy(self, fx, gx, ggx):
        """
        :Note cauthy point method to solve sub question

        :param fx: f(x), np.ndarray of shape (N, 1)
        :param gx: g(x), np.ndarray of shape (N, 1)
        :param ggx: gg(x), np.ndarray of shape (N, N)

        :return: d: the descent direction, np.ndarray of shape (N, 1)
        """
        gxggxgx = np.dot(gx.T, np.dot(ggx, gx))
        eta = 1
        if gxggxgx > 0:
            eta = np.min([1, vector_2norm(gx) ** 3 / (self.delta * gxggxgx)])
        d = -1 * eta * self.delta / vector_2norm(gx) * gx
        return d

    def _subspace(self, fx, gx, ggx):
        """
        :Note two-dimension subspace method to solve sub question

        :param fx: f(x), np.ndarray of shape (N, 1)
        :param gx: g(x), np.ndarray of shape (N, 1)
        :param ggx: gg(x), np.ndarray of shape (N, N)

        :return: d: the descent direction, np.ndarray of shape (N, 1)
        """
        min_eigval = np.min(np.linalg.eigvals(ggx))
        if min_eigval < -1e-5:
            modify_ggx = ggx - 1.5*min_eigval*np.eye(ggx.shape[0])
            inv_modify_ggx = np.linalg.inv(modify_ggx)
            invmodifyggx_gx = np.dot(inv_modify_ggx, gx)
            a = vector_2norm(gx)**2
            b = np.dot(gx.T, invmodifyggx_gx)
            c = vector_2norm(invmodifyggx_gx)**2
            d = np.dot(gx.T, np.dot(ggx, gx))
            e = np.dot(np.dot(ggx, gx).T, invmodifyggx_gx)
            f = np.dot(np.dot(ggx, invmodifyggx_gx).T, invmodifyggx_gx)

            p = a*c - b ** 2
            q = e*b - a*f
            m = 4*b*e - 2*a*f - 2*c*d
            r = d*f - e ** 2
            n = a*e - b*d

            # 0=q_4 v^4 + q_3 v^3 + q_2 v^2 + q_1 v+ q_0
            q4 = 16*p**2 * self.delta**2
            q3 = 8*m*p*self.delta**2
            q2 = (8*p*r + m**2) * self.delta**2 - 4*a*p**2
            q1 = 2*m*r*self.delta**2 - 4*(a*p*q+b*n*p)
            q0 = self.delta**2*r**2 - (a*q**2 + 2*b*n*q + c*n**2)

            v = np.roots(np.array([q4, q3, q2, q1, q0]).flatten())
            v = np.sort(v)
            for i in v:
                if np.imag(i) == 0:
                    v = np.real(i)
                    break
            t = 4*p*v**2 + m*v + n
            d = 1/t * ((2*v*p+q)*gx + n*invmodifyggx_gx)

        elif min_eigval > 1e-5:
            a = vector_2norm(gx)**2
            inv_ggx = np.linalg.inv(ggx)
            b = np.dot(np.dot(inv_ggx, gx).T, gx)
            c = vector_2norm(np.dot(inv_ggx, gx))**2
            d = np.dot(np.dot(ggx, gx).T, gx)

            m = a*c - b*b
            n = a*b - c*d
            q = a*a - b*d

            # 0=q_4 v^4 + q_3 v^3 + q_2 v^2 + q_1 v+ q_0
            q4 = 16*m**2 * self.delta**2
            q3 = 16*m*n*self.delta**2
            q2 = 4*n**2 * self.delta ** 2 - 8*m*q*self.delta**2 - 4*a*m**2
            q1 = 4*b*q*m - 4*n*q*self.delta**2
            q0 = (self.delta**2 - c) * q**2

            v = np.roots(np.array([q4, q3, q2, q1, q0]).flatten())
            v = np.sort(v)
            for i in v:
                if np.imag(i) == 0:
                    v = np.real(i)
                    break
            t = 4*m*v**2 + 2*n*v - q
            d = (2*v*m*gx + q*np.dot(inv_ggx, gx)) / t
        else:
            return self._cauthy(fx, gx, ggx)

        return d

    def _q_dk(self, fx, gx, ggx, dx):
        """
        :Note return value of quadratic function: q(d)=fx + <gx, d> + \frac{1}{2} d^T ggx d

        :param fx: f(x), np.ndarray of shape (N, 1)
        :param gx: g(x), np.ndarray of shape (N, 1)
        :param ggx: gg(x), np.ndarray of shape (N, N)
        :param dx: the descent direction, np.ndarray of shape (N, 1)

        :return: value of quadratic function q, float
        """
        return fx + np.dot(gx.T, dx) + 0.5 * np.dot(dx.T, np.dot(ggx, dx))
