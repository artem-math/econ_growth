# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *

# Create a Pyomo model
m = ConcreteModel()

# Define the time horizon and discretization step
duration = 2.0
tstep = 0.01
dicsr_points = np.arange(0.0, duration, tstep)
m.t = ContinuousSet(bounds=(0.0, duration), initialize = dicsr_points)

####PARAMETERS
paramDF = pd.read_csv('ecmodParams3c2.csv')
# paramDF = pd.read_csv('ecmodParams4.csv')
# paramDF = pd.read_csv('ecmodParams8.csv')

params = dict(zip(paramDF.key,paramDF.value))
#### Set model dimension
m.dim_n = RangeSet(params['dimension'])

#Here we initialize epsilon values
def eps_init(model, i):
	 return params['eps' + str(i)]
m.eps = Param(m.dim_n, rule= eps_init)

m.drate = Param(initialize = params['drate'])

### DERIVED PARAMETERS
def disc_fac(m, t):
	 return exp( - m.drate*t)
m.df = Param(m.t, rule= disc_fac)



m.U = Var(m.t, m.dim_n, domain=NonNegativeReals) #control vector U
#note that we assume that x01, x02, ... are sorted in increasing order
m.X = Var(m.t, m.dim_n, domain=NonNegativeReals) #state vector X
m.log_var = Var(m.t, domain=NonNegativeReals) #technical variable
m.F = Var(m.t, domain=NonNegativeReals) #production function
m.Util = Var(m.t, domain= Reals) #instantenious utility that we aggregate
m.dX = DerivativeVar(m.X, wrt=m.t) #vector of 1st derivatives of the state vector X


### EQUATIONS

# Simplex constraints for U
def _simplexU(model, t):
	return sum(model.U[t,j] for j in model.dim_n) <= 1.0
m.simplexU = Constraint(m.t, rule = _simplexU)

# Differential equation for X
def _diff_equ(model, t, j):
	return model.dX[t, j] == model.U[t, j] * model.F[t] / model.eps[j]
m.diff_equ = Constraint(m.t, m.dim_n,  rule = _diff_equ)

# Production function
def _F_equ(model, t):
	return model.F[t] == prod( (model.X[t, j]+0.000001)**model.eps[j] for j in model.dim_n) #can't evaluate pow'(0,0.1).
m.F_equ = Constraint(m.t,  rule = _F_equ)

# Technical equation to avoid numeric problems
def _numeric_problem1(m, t):
	uval = sum(m.U[t,j] for j in m.dim_n)
	return (1.000 - uval) * m.F[t] == m.log_var[t]
m.numeric_problem1 = Constraint(m.t,  rule = _numeric_problem1)

# Instantenious utility
def _utilityCalc(m, t):
	return m.Util[t] ==  m.df[t] * log(  m.log_var[t] + 0.000001) #can't evaluate log(0).
m.utilityCalc = Constraint(m.t, rule=_utilityCalc)

# # Integral utility
m.OBJ = Objective(expr = tstep * sum(m.Util[t] for t in m.t), sense = maximize)

####======= INITIAL CONDITIONS
for j in m.dim_n:
	m.diff_equ[m.t.first(), j].deactivate()

for j in m.dim_n:
	m.X[m.t.first(),j].fix(params['incond' + str(j)])

# for t in m.t:
# 	if t >tau1+tau2:
# 		for j in [1,2,3]:
# 			m.U[t,j].fix(epsilon[j]*(1-m.drate.value))


# Discretize the model using Backward Finite Difference method
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m,nfe=duration/tstep,scheme='BACKWARD')

# discretizer.apply_to(m,nfe=duration/tstep,scheme='FORWARD')

# discretizer = TransformationFactory('dae.collocation')
# discretizer.apply_to(m, wrt=m.t, nfe=duration/tstep, ncp=2, scheme='LAGRANGE-RADAU')

# # Collocation with piece-wise constant controls
# discretizer = TransformationFactory('dae.collocation')
# discretizer.apply_to(m, nfe=duration/tstep, ncp=5)
# m = discretizer.reduce_collocation_points(m, var=m.U, ncp=1, contset=m.t)



solver=SolverFactory('ipopt')
solver.options['halt_on_ampl_error'] = 'yes'
solver.options['acceptable_tol'] = 1e-8
solver.options['constr_viol_tol'] = 1e-13
solver.options['max_iter'] = 3000

results = solver.solve(m, tee=True)

print tstep*sum([m.Util[t].value for t in m.t])

# # Plot the integral utility over time
# plt.plot(np.cumsum([m.Util[t].value for t in m.t]))
# plt.show()



# Supplementary functions to work with Pyomo objects
def unpack_time_ndim(dic_in):
	out_dic = {}
	for j in m.dim_n:
		f_list = []
		for t in m.t:
			f_list.append(dic_in[t, j].value)
		out_dic[j] = f_list
	return out_dic
def unpack_time_1dim(list_in):
	out_list = []
	for t in m.t:
		out_list.append(list_in[t].value)
	return out_list

# Derive variables as lists from the Pyomo objects
U_dic = unpack_time_ndim(m.U)
X_dic = unpack_time_ndim(m.X)
# dX_dic = unpack_time_ndim(m.dX)
# F_list = unpack_time_1dim(m.F)

# Set the codename for the run
expername = ' dim= ' + str(len(m.dim_n)) + '; tfin= ' + str(m.t.last()) + '; discr= ' + str(len(m.t)) + '; dr= ' + str(m.drate.value) + '; eps= ' + str(m.eps.values()) + '; X= ' + str([params['incond' + str(j)] for j in m.dim_n])

# Create plot
fig, axes = plt.subplots(2, 2, dpi=200, figsize = (20,20))

# For U
plt.subplot(2, 2, 1)
for j in m.dim_n:
	 plt.plot(U_dic[j],  '.-', label = 'U' + str(j))
plt.title('U and X;' + expername)
plt.ylabel('U')
plt.legend()

# For X
plt.subplot(2, 2, 2)
plt.ylabel('X')
for j in m.dim_n:
	 plt.plot(X_dic[j],  '.-', label = 'X' + str(j))
plt.legend()

# plt.subplot(2, 2, 3)
# plt.ylabel('X2')
# plt.plot(X_dic[1], X_dic[2],  '.-', label = 'X1 vs X2')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.ylabel('F')
# plt.plot(F_list,  '.-', label = 'F')
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.ylabel('dX')
# for j in m.dim_n:
# 	 plt.plot(dX_dic[j],  '.-', label = 'dX' + str(j))
# plt.legend()


# Plot the first second only
plt.subplot(2, 2, 3)
plt.ylabel('U')
for j in m.dim_n:
	 plt.plot(U_dic[j][:int(1.0/tstep)],  '.-', label = 'U' + str(j) +'the first second')
plt.legend()
plt.subplot(2, 2, 4)
plt.ylabel('X')
for j in m.dim_n:
	 plt.plot(X_dic[j][:int(1.0/tstep)],  '.-', label = 'X' + str(j) +'the first second')
plt.legend()
# plt.show()
plt.savefig(expername+'.png')
plt.clf()
