import numpy as np
from optimizer.newtonMethod import NewtonMethod, GillMurrayNewton, FletcherFreemanMethod
from optimizer.steepestDescent import SteepestDescent
from linearSearch.monotoneSearch import AccurateLinearSearch, InaccurateLinearSearch
from linearSearch.nonmonotoneSearch import NonmonotoneGLL
from testFunction import ExtendedPowellSingular, Rosenbrock, BiggsEXP6, PowellBadlyScaled


def show_result(optimizer, xx):
    print(optimizer.__class__.__name__)
    print("-"*20)
    print("f(xx)\n {}\n"
          "g(xx)\n {}\n"
          "gg(xx)\n {}".format(optimizer.f(xx), optimizer.g(xx), optimizer.gg(xx)))
    print("-"*20)


if __name__ == "__main__":
    """
    RF: Rosenbrock Function
    PBS: Powell badly scaled Function
    EPS: Extended Powell singular Function
    BEXP: Biggs EXP6 Function

    """
    BEXP_m = 13
    EPS_m, EPS_n = 80, 80
    PBS, RF, EPS, BEXP = PowellBadlyScaled(), Rosenbrock(), ExtendedPowellSingular(m=EPS_m, n=EPS_n), BiggsEXP6(
        m=BEXP_m)
    PBS_x0, RF_x0, EPS_x0, BEXP_x0 = np.array([0, 1]).reshape((2, 1)), \
                                     np.array([-1.2, 1]).reshape((2, 1)), \
                                     np.array([3, -1, 0, 1] * (EPS_m // 4)).reshape((EPS_m, 1)), \
                                     np.array([1, 2, 1, 1, 1, 1]).reshape(6, 1)
    PBS_xx, RF_xx, EPS_xx, BEXP_xx = np.array([1.098e-5, 9.106]).reshape((2, 1)), \
                                     np.array([1, 1]).reshape((2, 1)), \
                                     np.array([0] * EPS_m).reshape((EPS_m, 1)), \
                                     np.array([1, 10, 1, 5, 4, 3]).reshape((-1, 1))



    gll_optimizer = NonmonotoneGLL(
        method="GLL",
        max_iter=1e3,
        GLL_rho=0.1,
        GLL_alpha=1,
        GLL_M=5,
        GLL_sigma=0.4
    )
    acc_optimizer = AccurateLinearSearch()
    gold_interp22_optimizer = InaccurateLinearSearch(
        method="Interp22",
        condition="GoldStein",
        GoldStein_rho=0.01)
    wolfe_interp22_optimizer = InaccurateLinearSearch(method="Interp22",
                                                      condition="Wolfe",
                                                      max_iter=10,
                                                      Wolfe_rho=0.1,
                                                      Wolfe_sigma=0.4,
                                                      )
    wolfeP_interp22_optimizer = InaccurateLinearSearch(method="Interp22",
                                                       condition="WolfePower",
                                                       Wolfe_rho=0.1,
                                                       Wolfe_sigma=0.4)
    wolfe_interp33_optimizer = InaccurateLinearSearch(method="Interp33",
                                                      condition="Wolfe",
                                                      Wolfe_rho=0.1,
                                                      Wolfe_sigma=0.4)
    wolfeP_interp33_optimizer = InaccurateLinearSearch(method="Interp33",
                                                       condition="WolfePower",
                                                       Wolfe_rho=0.1,
                                                       Wolfe_sigma=0.4)
    gold_interp33_optimizer = InaccurateLinearSearch(method="Interp33",
                                                     condition="GoldStein",
                                                     GoldStein_rho=0.1)
    gold_backtracking_optimizer = InaccurateLinearSearch(
        method="BackTracking",
        condition="GoldStein",
        BackTracking_theta=0.9,
        GoldStein_rho=0.1,
        max_iter=100
    )
    wolfeP_backtracking_optimizer = InaccurateLinearSearch(
        method="BackTracking",
        condition="WolfePower",
        Wolfe_rho=0.1,
        BackTracking_theta=0.9,
        max_iter=100,
        Wolfe_sigma=0.6
    )
    wolfe_backtracking_optimizer = InaccurateLinearSearch(
        method="BackTracking",
        condition="Wolfe",
        BackTracking_theta=0.9,
        max_iter=100,
        Wolfe_rho=0.1,
        Wolfe_sigma=0.6
    )
    #   method = NewtonMethod(step_optimizer=wolfe_interp22_optimizer, max_error=1e-16, max_iter=1e6)
    #   method = GillMurrayNewton(step_optimizer=gold_backtracking_optimizer, max_error=1e-18)
    #   method = NewtonMethod(step_optimizer=gll_optimizer, max_error=1e-16)

    #   method.compute(RF.f, RF.g, RF.gg, RF_x0)
    #   method.compute(EPS.f, EPS.g, EPS.gg, EPS_x0)
    #   method.compute(PBS.f, PBS.g, PBS.gg, PBS_x0)


