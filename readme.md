# Code flow

1. init a suitable step optimizer
2. init a suitable method optimizer
3. use the method_optimizer.compute to compute the minimum value of function

# Handbook:

### step optimizer

These step optimizers are divided into two categories according to monotonicity.
+ MonotoneSearch
    + AccurateLinearSearch
    + InaccurateLinearSearch
+ NonmonotoneSearch
    + GLL method

---



### MonotoneSearch

### AccurateLinearSearch

    acc_optimizer = AccurateLinearSearch(
        method="GoldenRate",
        find_area_method="GoAndBack",
        max_iter=1e3,
        **opt
        )

### InaccurateLinearSearch
    condition_method_optimizer = InaccurateLinearSearch(
        method="Interp22" or "Interp33"
        condition="GoldStein" or "Wolfe" or "WolfePower"
        max_iter=1e3,
        **opt
        )
    
    **Watch out**
    opt for different conditions need to be different 
    
    GoldStein condition
        opt = { "GoldStein_rho" : 0.1}
    Wolfe Condition 
        opt = { "Wolfe_rho" : 0.1,
                "Wolfe_sigma" : 0.4}
    WolfePower Condition
        opt = { "Wolfe_rho" : 0.1,
                "Wolfe_sigma" : 0.4}

## NonmonotoneSearch

    gll_optimizer = NonmonotoneGLL(
        method = "GLL",
        max_iter = 1e5,
        opt = { "GLL_rho" : ,
                "GLL_alpha": ,
                "GLL_M":,  
                "GLL_sigma"})