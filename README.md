# econ_growth
This project contains the Python code related to the economic growth model similar to the one considered in the paper https://link.springer.com/article/10.3103/S0278641917020042
but on infinite horizon.

The code allows to examine the n-dimensional economic model with the Cobb-Douglas production function on infinite horizon. The utility function has the integral form with discounting; the integrant is of the logarithm type. The model assumes that all depreciation rates are equal. Application of the Pontryagin maximum principle leads to a boundary-value problem with a special transversality condition. The presence of special regimes in the optimal solution complicates the boundary value problem of the maximum principle. Under certain assumptions on the right-hand sides of the differential equations, the studied problem allows a biological interpretation in the model of optimal growth of crops with an arbitrary number of vegetative organs.

HOW TO RUN THE CODE
1. Update the relevant parameters for the run in the corresponding .csv file. For example, ecmodParams2.csv contains parameters for the 2-dimensional economic model.
2. (Works only for n=3) Execute analytic_infty_n=3.py to obtain a numeric approximation for analytic infinite-horizon solution.
3. (Tested for n= 2, 3, 4, 8) Execute numeric_sol.py to obtain the numeric finite-horizon solution. 

The code requires Pyomo and IPOPT solver. We recommend Anaconda as a simple way to install both of them.

Keywords: multifactor economic model, Cobb-Douglas function, optimal control, maximum principle.
