import numpy as np
from optimizer.basicOptimizer import BasicOptimizer
from utils.norm import vector_2norm
from utils.norm import matrix_inftynorm


class BFGS(BasicOptimizer):
    def __init__(self, m, step_optimizer=None, max_error=1e-3, max_iter=1e3, **opt):
        super().__init__(step_optimizer, max_error, max_iter)
        self.m = m

    def _convergence(self, gx, x):
        return True if vector_2norm(gx.flatten()) < self.max_error * np.max([1, vector_2norm(x.flatten())]) else False

    def compute(self, f, g, x0):
        self.iter_num = 0
        fx0, gx0 = f(x0), g(x0)

        while True:
            d = self._get_descent_direction(gx0)
            alpha = 1 if self.step_optimizer is None else self.step_optimizer.get_step_length(f, g, x0, d)
            x1 = x0 + alpha * d
            fx1, gx1 = f(x1), g(x1)
            #if self.iter_num % 10 == 0:
            print("iter_num is{}, f(x) is{}, alpha is {}, g(x) is {}, x is {}".format(self.iter_num, fx1, alpha, vector_2norm(gx1.flatten()), vector_2norm(x1.flatten())))
            if self._maximum_loop() is True or self._convergence(gx1, x1) is True:
                self.f_val.append(fx1)
                self.d_val.append(d)
                self.g_val.append(gx1)
                self.alpha.append(alpha)
                print("iter_num:{}\n"
                      "f(x) is {},\n"
                      "delta is:{}".format(self.iter_num, fx1, np.abs(fx0 - fx1)))
                break
            self._update_s_y(x1, x0, gx1, gx0)
            x0, fx0, gx0 = x1, fx1, gx1
            self._iter_increment()
        return x0


class LBFGSOriginal(BFGS):
    def __init__(self, m, step_optimizer, max_error, max_iter):
        super().__init__(m, step_optimizer, max_error, max_iter)
        # store s_k=x_{k+1}-x_k, y_k = g(x_{k+1})-g(x_k)
        self.s, self.y, self.rho = [], [], []

    def _update_s_y(self, x1, x0, gx1, gx0):
        if len(self.s) == self.m:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        self.s.append(x1-x0)
        self.y.append(gx1-gx0)
        self.rho.append(1/np.dot(self.s[-1].T, self.y[-1]))

    def _get_descent_direction(self, gx):
        # loop 1
        q = -gx
        alpha = []
        for i in range(len(self.s)-1, -1, -1):
            alpha.append(self.rho[i]*np.dot(self.s[i].T, q))
            q = q-alpha[-1]*self.y[i]

        # loop 2
        # we use Identy matrix for H0
        r = q
        for i in range(0, len(self.s), 1):
            beta = self.rho[i]*np.dot(self.y[i].T, r)
            r = r + self.s[i]*(alpha[len(self.s)-i-1]-beta)

        #

        if np.dot(gx.T, r) > 0:
            r = -r

        return r


class LBFGSCompressed(LBFGSOriginal):
    def __init__(self, m, step_optimizer, max_error, max_iter, **opt):
        super().__init__(m, step_optimizer, max_error, max_iter, **opt)
        self.s, self.y, self.sy = [], [], []

    def _get_descent_direction(self, gx):
        """

        :param f:
        :param g:
        :param x0:
        :return:
        """
        if len(self.s) < 1:
            return -gx
        gk = gx
        Sk, Yk = np.array(self.s).reshape(len(self.s), -1).T, np.array(self.y).reshape(len(self.s), -1).T
        Skgk = np.dot(Sk.T, gk)
        Ykgk = np.dot(Yk.T, gk)
        sk1 = self.s[-1]
        yk1 = self.y[-1]
        sk1yk1 = np.dot(sk1.T, yk1)[0][0]
        rk = np.zeros((len(self.s), len(self.s)), np.float64)
        for i in range(rk.shape[0]):
            for j in range(rk.shape[1]):
                if i <= j:
                    rk[i][j] = np.dot(self.s[i].T, self.y[j])
        dk = np.diag(np.array(self.sy).reshape(-1))
        ykyk = np.dot(Yk.T, Yk)
        '''
        if np.linalg.det(rk) == 0:
            print("rk 不可逆")
            return -gx
        '''
        gamma = sk1yk1/ykyk[-1][-1]
        rk_inv = np.linalg.inv(rk)
        p1 = np.dot(rk_inv.T, dk+gamma*ykyk)
        p1 = np.dot(p1, rk_inv)
        p1 = np.dot(p1, Skgk)
        p1 = p1 - gamma*np.dot(rk_inv.T, Ykgk)
        p2 = -np.dot(rk_inv, Skgk)
        p = np.vstack([p1, p2])
        q = np.hstack([Sk, gamma*Yk])
        d = gamma*gk + np.dot(q, p)
        
        """

        if len(self.s) == 0:
            return -gx

        sk = np.array(self.s, np.float64).T.reshape(-1, len(self.s))
        yk = np.array(self.y, np.float64).T.reshape(-1, len(self.y))
        dk = np.diag(np.array(self.sy, np.float64).reshape(-1))
        rk = np.zeros((len(self.s), len(self.s)), np.float64)
        for i in range(rk.shape[0]):
            for j in range(rk.shape[1]):
                if i <= j:
                    rk[i][j] = np.dot(self.s[i].T, self.y[j])

        eta = np.dot(self.y[-1].T, self.s[-1])/np.dot(self.y[-1].T, self.y[-1])[0][0]
        
        rk_inv = np.linalg.inv(rk)
        sk_rkinvt = np.dot(sk, rk_inv.T)
        yk_rkinv_skt = np.dot(yk, sk_rkinvt.T)
        H = eta*np.eye(gx.shape[0], dtype=np.float64)+np.dot(sk_rkinvt, np.dot(dk+eta*np.dot(yk.T, yk), sk_rkinvt.T)) - eta*(yk_rkinv_skt - yk_rkinv_skt.T)

        d = np.dot(H, -gx)
        """
        if np.dot(gx.T, d) > 0:
            d = -d
        #print("gamma is {}, gk is{}, dot is {}".format(gamma, gk, np.dot(q.T, p)))
        return d

    def _update_s_y(self, x1, x0, gx1, gx0):
        if len(self.s) == self.m:
            self.s.pop(0)
            self.y.pop(0)
            self.sy.pop(0)
        self.s.append(x1-x0)
        self.y.append(gx1-gx0)
        self.sy.append(np.dot(self.s[-1].T, self.y[-1])[0][0])
