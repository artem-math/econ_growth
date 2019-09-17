### The code below is ONLY valid for 3 points case

import pandas as pd
paramDF = pd.read_csv('ecmodParams3c2.csv')
params = dict(zip(paramDF.key,paramDF.value))

###===================================================
### Analytical part
###===================================================
x10 = params['incond1']
x20 = params['incond2']
x30 = params['incond3']
z0 = x10
ztau = x20
qtau1 =1./params['drate']/x20
qtau2 =1./params['drate']/x30
T = 1.5
nu = params['drate']


am = ConcreteModel()
# dicsr_points = range(duration)
duration = T #
tstep = 0.01
dicsr_points = np.arange(0.0, duration, tstep)
am.t = ContinuousSet(bounds=(0.0, duration), initialize = dicsr_points)

#sol1
def step1(params, iter = 100):
	epsilon = (0, params['eps1'], params['eps2'], params['eps3'], )
	m = ConcreteModel()
	tstep = (x30 - x20)/float(iter)
	dicsr_points = np.arange(x20, x30, tstep)

	m.z = ContinuousSet(bounds=(x20, x30), initialize = dicsr_points)
	# m.z.pprint()
	m.q = Var(m.z, domain = Reals, initialize = 7.0)
	m.dq = DerivativeVar(m.q, wrt=m.z, domain = NegativeReals)
	def _ode32(m, z):
		return m.dq[z] == - m.q[z]*(z**(epsilon[1]+epsilon[2]-1)*x30**epsilon[3]-nu)*(epsilon[1]+epsilon[2])/(z**(epsilon[1]+epsilon[2])*x30**epsilon[3]-1/m.q[z])
	m.ode32 = Constraint(m.z,  rule = _ode32)

	# m.ode32[m.z.first()].deactivate()
	m.q[m.z.last()].fix(qtau2)

	def obj_rule(m):
		return  m.q[m.z.last()]
	m.OBJ = Objective(rule=obj_rule, sense = maximize)


	discretizer = TransformationFactory('dae.finite_difference')
	discretizer.apply_to(m,nfe=len(dicsr_points),scheme='BACKWARD')
	solver=SolverFactory('ipopt')
	solver.options['halt_on_ampl_error'] = 'yes'
	solver.options['acceptable_tol'] = 1e-5
	solver.options['constr_viol_tol'] = 1e-10
	solver.options['max_iter'] = 300
	results = solver.solve(m,tee=True)
	# plt.plot(np.asarray([m.dq[z].value for z in m.z]))
	# plt.show()
	# print np.asarray(m.z)
	return m.q[m.z.first()].value, np.asarray([m.q[z].value for z in m.z])

#sol2
def step2(params, qtau, ztau, tt = 0.0, TT = T, iter = 100, full_out = False):
	epsilon = (0, params['eps1'], params['eps2'], params['eps3'], )
	m = ConcreteModel()
	tstep = (TT - tt)/float(iter)
	print tstep
	dicsr_points = np.arange(tt, TT, tstep)
	m.t = ContinuousSet(bounds=(tt, TT), initialize = dicsr_points)
	m.q = Var(m.t, domain = Reals, initialize = 0.1)
	m.dq = DerivativeVar(m.q, wrt=m.t, domain = Reals)
	m.z = Var(m.t, domain = NonNegativeReals, initialize = 0.1)
	m.dz = DerivativeVar(m.z, wrt=m.t, domain = Reals)
	def _ode12(m, t):
		return m.dz[t] ==  (m.z[t]**(epsilon[1]+epsilon[2])*x30**epsilon[3]-1/m.q[t])/(epsilon[1]+epsilon[2])
	m.ode12 = Constraint(m.t,  rule = _ode12)
	def _ode22(m, t):
		return m.dq[t] ==  -m.q[t]*(m.z[t]**(epsilon[1]+epsilon[2]-1)*x30**epsilon[3]-nu)
	m.ode22 = Constraint(m.t,  rule = _ode22)

	m.ode12[m.t.first()].deactivate()
	m.ode22[m.t.first()].deactivate()
	m.q[m.t.first()].fix(qtau)
	m.z[m.t.first()].fix(ztau)

	def obj_rule(m):
		return  m.q[m.t.first()]
	m.OBJ = Objective(rule=obj_rule, sense = maximize)


	discretizer = TransformationFactory('dae.finite_difference')
	discretizer.apply_to(m,nfe=len(dicsr_points),scheme='BACKWARD')
	solver=SolverFactory('ipopt')
	solver.options['acceptable_tol'] = 1e-5
	solver.options['constr_viol_tol'] = 1e-8
	solver.options['max_iter'] = 800
	results = solver.solve(m,tee=True)

	if full_out ==True:
		return np.asarray([m.z[t].value for t in m.t]), np.asarray([m.q[t].value for t in m.t]), np.asarray(m.t)
	else:
		return np.asarray([m.z[t].value for t in m.t]),  np.asarray(m.t)

#sol3
def step3(params, qtau):
	epsilon = (0, params['eps1'], params['eps2'], params['eps3'], )
	m = ConcreteModel()
	tstep = (x20 - x10)/100
	dicsr_points = np.arange(x10, x20, tstep)
	m.z = ContinuousSet(bounds=(x10, x20), initialize = dicsr_points)
	m.q = Var(m.z, domain = Reals, initialize = 1.0)
	m.dq = DerivativeVar(m.q, wrt=m.z, domain = NegativeReals)
	def _ode31(m, z):
		return m.dq[z] == -m.q[z]*(z**(epsilon[1]-1)*x20**(epsilon[2])-nu)/(1/epsilon[1]*(z**(epsilon[1])*x20**(epsilon[2])-1/m.q[z]))
	m.ode31 = Constraint(m.z,  rule = _ode31)


	m.ode31[m.z.first()].deactivate()
	m.q[m.z.last()].fix(qtau)

	def obj_rule(m):
		return  m.q[m.z.first()]
	m.OBJ = Objective(rule=obj_rule, sense = maximize)


	discretizer = TransformationFactory('dae.finite_difference')
	discretizer.apply_to(m,nfe=len(dicsr_points),scheme='BACKWARD')
	solver=SolverFactory('ipopt')
	solver.options['halt_on_ampl_error'] = 'yes'
	solver.options['acceptable_tol'] = 1e-5
	solver.options['constr_viol_tol'] = 1e-10
	solver.options['max_iter'] = 300
	results = solver.solve(m,tee=True)
	return m.q[m.z.first()].value

#sol4
def step4(params, q0, T, iter = 100):
	z0 = x10
	epsilon = (0, params['eps1'], params['eps2'], params['eps3'], )
	m = ConcreteModel()
	tstep = T/float(iter)
	dicsr_points = np.arange(0.0, T, tstep)
	m.t = ContinuousSet(bounds=(0, T), initialize = dicsr_points)
	m.q = Var(m.t, domain = Reals, initialize = 1.0)
	m.dq = DerivativeVar(m.q, wrt=m.t, domain = Reals)
	m.z = Var(m.t, domain = NonNegativeReals, initialize = 1.0)
	m.dz = DerivativeVar(m.z, wrt=m.t, domain = Reals)
	def _ode11(m, t):
		return m.dz[t] ==  1./epsilon[1]*(m.z[t]**(epsilon[1])*x20**(epsilon[2])-1/m.q[t])
	m.ode11 = Constraint(m.t,  rule = _ode11)
	def _ode21(m, t):
		return m.dq[t] ==  -m.q[t]*(m.z[t]**(epsilon[1]-1)*x20**(epsilon[2])-nu)
	m.ode21 = Constraint(m.t,  rule = _ode21)

	m.ode11[m.t.first()].deactivate()
	m.ode21[m.t.first()].deactivate()
	m.q[m.t.first()].fix(q0)
	m.z[m.t.first()].fix(z0)

	def obj_rule(m):
		return  m.q[m.t.last()]
	m.OBJ = Objective(rule=obj_rule, sense = maximize)


	discretizer = TransformationFactory('dae.finite_difference')
	discretizer.apply_to(m,nfe=len(dicsr_points),scheme='BACKWARD')
	solver=SolverFactory('ipopt')
	solver.options['halt_on_ampl_error'] = 'yes'
	solver.options['acceptable_tol'] = 1e-5
	solver.options['constr_viol_tol'] = 1e-10
	solver.options['max_iter'] = 800
	results = solver.solve(m,tee=True)

	return np.asarray([m.z[t].value for t in m.t]), np.asarray([m.q[t].value for t in m.t]), np.asarray(m.t)



qtau, list_q_step1 =  step1(params)
[list_z, list_t]  = step2(params, qtau, ztau, TT = T)
step2T = np.min(list_t[list_z> x30*(1.01)])
[list_z, list_t]  = step2(params, qtau, ztau, TT = step2T)
step2T = np.min(list_t[list_z> x30*(1.001)])
[list_z, list_t]  = step2(params, qtau, ztau, TT =step2T)
tau2 = np.min(list_t[list_z>= x30])

q0 = step3(params, qtau)

[list_z, list_q, list_t]  = step4(params, q0, T)
step4T = np.min(list_t[list_z> x20*(1.01)])
[list_z, list_q, list_t]  = step4(params, q0, step4T, iter = 200)
step4T = np.min(list_t[list_z> x20*(1.01)])
[list_z, list_q, list_t]  = step4(params, q0, step4T, iter = 200)
tau1 = np.min(list_t[list_z>= x20])


###===================================================
####Below we plot control U
epsilon = (0, params['eps1'], params['eps2'], params['eps3'], )
q = list_q[list_t<= tau1]
z = list_z[list_t<= tau1]
z4 = z
q4 = q
t01 = list_t[list_t<= tau1]

plotu1 = 1-1.0/(q*z**epsilon[1]*x20**epsilon[2]*x30**epsilon[3])

#sol5
[list_z, list_q, list_t] = step2(params, qtau, ztau, tt=tau1, TT=tau1+tau2, full_out = True)
q = list_q
z = list_z
z5 = z
q5 = q
t12 = list_t

plotu2 = epsilon[1]*(1.0-1.0/(q*z**(epsilon[1]+epsilon[2])*x30**epsilon[3]))/(epsilon[1]+epsilon[2])
plotu3 = epsilon[2]*(1.0-1.0/(q*z**(epsilon[1]+epsilon[2])*x30**epsilon[3]))/(epsilon[1]+epsilon[2])


t2T = np.linspace(tau1+tau2, T, num=50)
usng1 = epsilon[1]*(1-nu)*np.ones(len(t2T))
usng2 = epsilon[2]*(1-nu)*np.ones(len(t2T))
usng3 = epsilon[3]*(1-nu)*np.ones(len(t2T))
z1 = np.exp(t2T-tau1-tau2)*(x30-1.0/(qtau2*nu))+np.exp((1-nu)*(t2T-tau1-tau2))/(qtau2*nu)

#u1
# plt.plot(t01, plotu1)
# plt.plot(t12, plotu2)
# plt.plot(t2T, usng1)
# plt.show()
#
# #u2
# plt.plot(t01, np.zeros(len(t01)))
# plt.plot(t12, plotu3)
# plt.plot(t2T, usng2)
# plt.show()
#
# #u3
# plt.plot(t01, np.zeros(len(t01)))
# plt.plot(t12, np.zeros(len(t12)))
# plt.plot(t2T, usng3)
# plt.show()

##===================================================
####Below we plot X
# #X1
qtau2 =1./params['drate']/x30

plotT = np.hstack((t01, t12, t2T))
plotX1 = np.hstack((z4, z5, z1))
fig, axes = plt.subplots(1, 1, dpi=200, figsize = (10,10))

# plt.plot(t01, z4)
# plt.plot(t12, z5)
# plt.plot(t2T, z1)
plt.plot(plotT, plotX1, label = 'X1')
plt.ylabel('state')
plt.xlabel('time')
plt.title('X1, X2, and X3')


# #X2
# plt.plot(t01, np.ones(len(t01))* x20)
# plt.plot(t12, z5)
# plt.plot(t2T, z1)
plotX2 = np.hstack((np.ones(len(t01))* x20, z5, z1))
plt.plot(plotT, plotX2, label = 'X2')

# #X3
# plt.plot(t01, np.ones(len(t01))* x30)
# plt.plot(t12, np.ones(len(t12))* x30)
# plt.plot(t2T, z1)
plotX3 = np.hstack((np.ones(len(t01))* x30, np.ones(len(t12))* x30, z1))
plt.plot(plotT, plotX3, label = 'X3')
plt.legend()
plt.savefig('n=3.png')
plt.clf()


# ###===================================================
# ####Below we plot Utility
#
# plot113 = []
# for ti in t01:
# 	t01i = t01[t01<=ti]
# 	plot113.append( np.trapz(y = np.exp(-nu*t01i)*np.log(1./q4[t01<=ti]) , x = t01i))
# plot113 = np.asarray(plot113)
#
# plot112 = []
# for ti in t12:
# 	t12i = t12[t12<=ti]
# 	plot112.append(plot113[-1] + np.trapz(y = np.exp(-nu*t12i)*np.log(1/q5[t12<=ti]), x = t12i))
# plot112 = np.asarray(plot112)
#
# plot111 = []
# for ti in t2T:
# 	t2Ti = t2T[t2T<=ti]
# 	plot111.append(plot112[-1] + np.trapz(y = np.exp(-nu*t2Ti)*np.log(nu*z1[t2T<=ti]), x = t2Ti)) ## Mistake!? (1.0-nu)*z1[t2T<=ti] ==> nu*z1[t2T<=ti]
# plot111 = np.asarray(plot111)
#
# plt.plot(t01, plot113)
# plt.plot(t12, plot112)
# plt.plot(t2T, plot111)
# plt.show()
