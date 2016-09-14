import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

tau1 = 0.13
tau2 = 0.20
tau3 = 0.13 
tau4 = 0.13
tau5 = 0.13
tau6 = 0.13
tau7 = 0.13
tau8 = 0.13
tau9 = 0.13

z1 = 0 # training
z2 = 0 # observed behavior
z3 = 0 # support 
z4 = 0 # internal cues
z5 = 0 # barriers
z6 = 0 # issue awareness
z7 = 1.0 # environment
z8 = 0  # external cues
z33 = 1.0 # praise (CONTROLLABLE)
z77 = 1.0 # external vision (CONTROLLABLE)

# initial conditions for height
h1 = 0 # skills 
h2 = 0 # impact expectancy
h3 = 0 # confidence 
h4 = 0 # behavior
h5 = 0 # behavioral outcomes
h6 = 0 # cue to action
h7 = 0 # personal gain expectancy
h8 = 0 # mood
h9 = 0 # altruistic tendencies

# initial states at time = 0    
x_0 = np.vector[(z1, z2, z3, z4, z5, z6, z7, z8, z33, z77,\
	h1, h2, h3, h4, h5, h6, h7, h8, h9)]       

# inflow resistance
g11 = 0.1
g22 = 0.3
g33 = 0.3
g3 = 0.3
g333 = 0.3
g55 = 0.3
g66 = 0.3
g6 = 0.3
g666 = 0.3
g9 = 0.3
g99 = 0.3
g999 = 0.3

# resistances of our life-coaching controls
g1 = 0.5
g2 = 0.6

# outflow resistance
b14 = 0.1 ## check on this
b21 = 0.5  # b31+b21 <= 1 
b34 = 0.1
b42 = 1   # b42 <= 1
b43 = 1   # b43 <= 1
b54 = 0.1  # b54 <= 1
b25 = 0.1
b85 = 0.1  # b25 + b85 <= 1
b46 = .1  # b46 <= 1
b47 = .9
b87 = 0.1  # b47 + b87 <= 1
b38 = 0.1
b78 = 0.1  # b38 + b78 <= 1
b69 = 0.1   # b69 <= 1



A = np.matrix([\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[g11, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, b14, 0, 0, 0, 0, 0],\
		[0, g22, 0, 0, 0, 0, 0, 0, b21, -1, 0, 0, b25, 0, 0, 0, 0],\
		[0, g3, g33, 0, -g333, 0, 0, 0, b31, 0, -1, b34, 0, 0, 0, b38, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, b42, b43, -1, 0, b46, b47, 0, 0],\
		[0, 0, 0, 0, 0, 0, g55, 0, 0, 0, 0, b54, -1, 0, 0, 0, 0],\
		[0, 0, 0, g6, 0, g66, 0, z8, 0, 0, 0, 0, 0, -1, 0, 0, b69],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, b78, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b85, 0, b87, -1, 0],\
		[g9, g999, g99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])

B = ([\
	[5,0,0],\
	[0,0,0],\
	[0,g1,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,g2],\
	[0,0,0],\
	[0,0,0],\
	])

n = 17 # number of states 
m = 3 # number of controllable inputs
T = 52 # number of weeks being considered 

x = Variable(n.T+1) # matrix of non-controllable inputs and states (tank heights)
u = Variable(m, T)  # matrix of life-coaching inputs
   
#/ tau ####### ********


states = [] # explanation****
# the following for loop will define the problem and then minimize the cost function using cvxpy 
for t in range(T):
    J = sum_squares(x[:,t+1]) + sum_squares(u[:,t]) # this is our cost function, working with states
    												   # and controllable life coaching inputs
    constr = [x[:,t+1] == A*x[:,t] + B*u[:,t], norm(u[:,t], 'inf') <= 1] # equation (9) in the paper 
    																	 # and infinity norm of u <= 1
    states.append(Problem(Minimize(J), constr))

prob = sum(states) 
prob.constraints += [x[11,T] == 2, x[:,0] == x_0] # the final number of volunteering hours/week = 2,
												  # define initial conditions x_0
prob.solve()



# f, ((x0, x1, x2), (x3, x4, x5), (x6, x7, x8)) = plt.subplots(3, 3)

# x0.plot(t,y[:,0])
# x0.set_title('Skills')
# x1.plot(t,y[:,1])
# x1.set_title('Impact Expectancy')
# x2.plot(t,y[:,2])
# x2.set_title('Confidence')
# x3.plot(t,y[:,3])
# x3.set_title('Behavior')
# x4.plot(t,y[:,4])
# x4.set_title('Behavioral Outcomes')
# x5.plot(t,y[:,5])
# x5.set_title('Cue to Action')
# x6.plot(t,y[:,6])
# x6.set_title('Personal Gain Expectancy')
# x7.plot(t,y[:,7])
# x7.set_title('Mood')
# x8.plot(t,y[:,8])
# x8.set_title('Altruistic Tendencies')

# f.set_size_inches(18.5, 10.5)

# plt.suptitle('Main title')
# plt.show()

