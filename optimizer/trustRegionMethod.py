import numpy as np
from optimizer.basicOptimizer import BasicOptimizer
from utils.norm import vector_2norm


class TrustRegion(BasicOptimizer):
    def __init__(self, subq_method="hebden", delta=0.1, max_error=1e-8, max_iter=1e3):
        super().__init__(None, max_error=max_error, max_iter=max_iter)
        self.subq_method = subq_method
        self.delta = delta

    def compute(self, f, g, gg, x0):
        fx0, gx0, ggx0 = f(x0), g(x0), gg(x0)
        while True:
            d, q_dk = self._get_descent_direction(fx0, gx0, ggx0)
            x1 = x0 + d
            fx1 = f(x1)
            gamma = (fx0-fx1)/(fx0-q_dk)
            if gamma <= 0.25:
                self.delta = self.delta / 4
            elif gamma >= 0.75 and vector_2norm(d) == self.delta:
                self.delta = self.delta * 3
            else:
                pass

            if self._convergence(f, x0, x1) or self._maximum_loop():
                print("sub is {}".format(f(x1)-f(x0)))
                print("iter num is {}".format(self.iter_num))
                return x1

            if gamma > 0:
                x0, fx0 = x1, fx1
                gx0, ggx0 = g(x0), gg(x0)
                self._iter_increment()
            else:
                pass
            self._iter_increment()

    def _get_descent_direction(self, fx, gx, ggx):
        if self.subq_method == "hebden":
            return self._hebden(fx, gx, ggx)
        elif self.subq_method == "cauthy":
            return self._cauthy(fx, gx, ggx)
        elif self.subq_method == "subspace":
            return self._subspace(fx, gx, ggx)
        q_dk = 0
        return fx, q_dk

    def _hebden(self, fx, gx, ggx):
        inv_ggx = np.linalg.inv(ggx)
        d = -np.dot(inv_ggx, gx)
        norm_d = vector_2norm(d)
        if norm_d <= self.delta:
            return d
        # find ||d(v)|| = delta_k
        # v0 = 0
        v = 0.1
        #
        i = 0
        while np.abs(norm_d-self.delta) > 1e-3*self.delta and i < 100:
            i = i+1
            inv_ggx = np.linalg.inv((ggx + v*np.eye(ggx.shape[0])))
            d = -np.dot(inv_ggx, gx)
            norm_d = vector_2norm(d)
            partial_d = -1 * np.dot(inv_ggx, d)
            psi_v = norm_d - self.delta
            psi_d_v = np.dot(d.T, partial_d)
            v = v - (psi_v + self.delta)/self.delta * psi_v/psi_d_v
        print(self._q_dk(fx, gx, ggx, d))
        return d, self._q_dk(fx, gx, ggx, d)

    def _cauthy(self, fx, gx, ggx):
        gxggxgx = np.dot(gx.T, np.dot(ggx, gx))
        eta = 1
        if gxggxgx > 0:
            eta = np.min([1, vector_2norm(gx)**3/(self.delta*gxggxgx)])
        dk = -1*eta*self.delta/vector_2norm(gx)*gx
        return dk, self._q_dk(fx, gx, ggx, dk)

    def _subspace(self, fx, gx, ggx):
        min_eigval = np.min(np.linalg.eigvals(ggx))
        if min_eigval < -1e-5:
            print("增加")
            ggx = ggx + 1.5 * min_eigval * np.eye(ggx.shape[0])
        elif min_eigval > 1e-5:
            #print("不变")
            pass
        else:
            #print("Cauthy")
            return self._cauthy(fx, gx, ggx)

        a = vector_2norm(gx)**2
        inv_ggx = np.linalg.inv(ggx)
        b = np.dot(np.dot(inv_ggx, gx).T, gx)
        c = vector_2norm(np.dot(inv_ggx, gx))**2
        d = np.dot(np.dot(ggx, gx).T, gx)

        m = a*c-b*b
        n = a*b-c*d
        q = a*a-b*d

        # 0=q_4 v^4 + q_3 v^3 + q_2 v^2 + q_1 v+ q_0
        q4 = 16*m*m*self.delta**2
        q3 = 16*m*n*self.delta**2
        q2 = 4*n*n*self.delta**2-8*m*q*self.delta**2-4*a*m**2
        q1 = 4*b*q*m-4*n*q*self.delta**2
        q0 = (self.delta**2-c)*q**2

        v = np.roots(np.array([q4, q3, q2, q1, q0]).flatten())
        v = np.sort(v)
        for i in v:
            if np.imag(i) == 0:
                v = np.real(i)
                break
        t = 4*m*v**2+2*n*v-q
        d = (2*v*m*gx+q*np.dot(inv_ggx, gx))/t
        return d, self._q_dk(fx, gx, ggx, d)

    def _q_dk(self, fx, gx, ggx, dx):
        return fx + np.dot(gx.T, dx) + 0.5*np.dot(dx.T, np.dot(ggx, dx))
