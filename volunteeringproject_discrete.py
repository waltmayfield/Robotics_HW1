# The purpose of this code is to utilize model predictive control as it applies to human behavior. 
# Specifically we investigate how the controllable inputs of a life coach- internal cues as a result of 
# goal setting, praise, and external vision- drive an individual to reach the goal of volunteering 
# for two hours per week. 

# import CVXcanon
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import *

# time constants
tau1 = 0.13
tau2 = 0.13
tau3 = 0.13 
tau4 = 0.13
tau5 = 0.13
tau6 = 0.13
tau7 = 0.13
tau8 = 0.13
tau9 = 0.13

# non-controllable inputs and their initial states at time = 0
z1 = 0.1 # training
z2 = 0.1 # observed behavior
z3 = 0.1 # support 
z4 = 0.1 # internal cues
z5 = 0.1 # barriers
z6 = 0.1 # issue awareness
z7 = 0.1 # environment
z8 = 0.1  # external cues
z33 = 0.1 # praise (CONTROLLABLE)
z77 = 0.1 # external vision (CONTROLLABLE)

# initial conditions for tank heights
h1 = 1 # skills 
h2 = 0 # impact expectancy
h3 = 0 # confidence 
h4 = 0 # behavior
h5 = 1 # behavioral outcomes
h6 = 1 # cue to action
h7 = 1 # personal gain expectancy
h8 = 1 # mood
h9 = 1 # altruistic tendencies

# array of initial states at time = 0    
x_0 = np.array([z1, z2, z3, z4, z5, z6, z7, z8, h1, h2, h3, h4, h5, h6, h7, h8, h9])      

# inflow resistance
g11 = 0.3
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
g0 = 5/100
g1 = 0.5/100
g2 = 0.6/100

# outflow resistance
b14 = 0.1 ## b14 <= 1
b21 = 0.5  # b31+b21 <= 1 
b31 = 0.1 
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

# A matrix, with coefficients of states  (tank heights and input factors)
A = np.matrix([\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[g11, 0, 0, 0, 0, 0, 0, 0, -.1, 0, 0, b14, 0, 0, 0, 0, 0],\
		[0, g22, 0, 0, 0, 0, 0, 0, b21, -.1, 0, 0, b25, 0, 0, 0, 0],\
		[0, g3, g33, 0, -g333, 0, 0, 0, b31, 0, -.1, b34, 0, 0, 0, b38, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, b42, b43, -.1, 0, b46, b47, 0, 0],\
		[0, 0, 0, 0, 0, 0, g55, 0, 0, 0, 0, b54, -.1, 0, 0, 0, 0],\
		[0, 0, 0, g6, 0, g66, 0, z8, 0, 0, 0, 0, 0, -.1, 0, 0, b69],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -.1, b78, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b85, 0, b87, -.1, 0],\
		[g9, g999, g99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -.1]])

# B matrix with coefficients of life-coach, controllable inputs
# notice that the "transfer function" impacting internal cues is a constant 5
B = np.matrix([\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[g0,0,0],\
	[0,0,0],\
	[0,g1,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,0],\
	[0,0,g2],\
	[0,0,0],\
	[0,0,0],\
	])

n = 17 # number of states (tank heights and input factors)
m = 3 # number of controllable inputs
T = 54 # number of weeks being considered 

x = Variable(n,T+1) # matrix of states (tank heights and input factors)
# u = Variable(m, T)  # matrix of life-coaching inputs
u = Int(m, T)


goal = np.ones((n, 1)) # goal tank heights for all tanks except behavior are equal to one
goal[11] = 2 # behavior goal is 2 hours of volunteering per week


# Note: time constants are considered to be ten


states = [] # States keeps track of all the tank heights during each of the time steps. 
			# Cvxpy uses this information to associate x and u variables with tank height via the cost function J
# the following for loop will define the problem and then minimize the cost function using cvxpy 
for t in range(T):
    J = sum_squares(x[:,t]-goal) +2e1*sum_squares(x[11,t]-goal[11]) + 1e-4*sum_squares(u[:,t]) # this is our cost function, working with states
    										      # and controllable life coaching inputs
    										      # notice that we have biased our cost function towards achieving volunteer goal
    constr = [x[:,t+1] == A*x[:,t] + B*u[:,t]] # equation (9) in the paper 
    										   
    states.append(Problem(Minimize(J), constr))

prob = sum(states)

# prob.constraints += [u >= 0] # u's are non-negative
prob.constraints += [x[:,0] == x_0] # define initial conditions x_0


prob.solve()

u=u/100 #in order to use intergers we had to have the optimal u values vairy between 0 and 100 instead of between 0 and 1.
#then we divide by 100 to bring u back to between 0 and 1 so it is normalized
#u now a multiple of 1/10

# plot results 
f, ((x0), (x1), (x2), (x3), (x4)) = plt.subplots(5, 1)
#f, ((x0), (x1), (x2), (x3), (x4), (x5), (x6), (x7), (x8), (x9), (x10), (x11)) = plt.subplots(12, 1)
x0.plot(u[0,0:50].value.A.flatten())
x0.set_title('Internal Cues Driven By Goal')
x1.plot(u[1,0:50].value.A.flatten())
x1.set_title('Praise')
x2.plot(u[2,0:50].value.A.flatten())
x2.set_title('External Vision')
x3.plot(x[11,0:50].value.A.flatten())
x3.set_title('Behavior')

# x5.plot(x[9,:].value.A.flatten())
# x5.set_title('Impact Expectancy')
# x6.plot(x[10,:].value.A.flatten())
# x6.set_title('Confidence')
# x7.plot(x[12,:].value.A.flatten())
# x7.set_title('Behavioral Outcomes')
# x8.plot(x[13,:].value.A.flatten())
# x8.set_title('Cue to Action')
# x9.plot(x[14,:].value.A.flatten())
# x9.set_title('Personal Gain Expectancy')
# x10.plot(x[15,:].value.A.flatten())
# x10.set_title('Mood')
# x11.plot(x[16,:].value.A.flatten())
# x11.set_title('Altruistic Tendencies')

plt.suptitle('Main title')
plt.show()

