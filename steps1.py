import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def stepper(t,Ton,Toff):
    return np.piecewise(t, [t < Ton,  t >= Ton], [0, 1]) * np.piecewise(t, [t <= Toff,  t > Toff], [1, 0])
   

def steps(n,t): # inputs are initial heights of tanks and time period being evaluated 
# time constants
    tau1 = 1.0
    tau2 = 1.0
    tau3 = 1.0 
    tau4 = 2.0
    tau5 = 1.0
    tau6 = 3.0

# inputs


    z1 = 0
    z2 = 3.0
    z3 = 3.0
    z4 = 0 
    z5 = 10.0
    z6 = 0
    z7 = 0
    z8 = 5.0*(stepper(t,2,7)+stepper(t,8,12))
    
    g11 = 3.0
    g22 = 1.0
    g32 = 2.0
    g33 = 1.0
    g35 = 1.0
    g36 = 1.0
    g57 = 2.0
    g64 = 15
    g68 = 15

    b14 = 0.23
    b21 = 0.3
    b25 = 0.3
    b31 = 0.5
    b34 = 0.2
    b42 = 0.3
    b43 = 0.8
    b46 = 0.44
    b45 = 0.1
    b54 = 0.3

# inflows
    qin1 = g11*z1 + b14*n[3]
    qin2 = g22*z2+b21*n[0]+b25*n[4]
    qin3 = g32*z2+g33*z3-g35*z5+g36*z6+b31*n[0]+b34*n[3]
    qin4 = b42*n[1]+b43*n[2]+b46*n[5]+b45*n[4]
    qin5 = g57*z7+b54*n[3]
    qin6 = g64*z4+g68*z8

  
    qout1 = n[0] # skills
    qout2 = n[1] # impact expectancy
    qout3 = n[2] # confidence
    qout4 = n[3] # behavior
    qout5 = n[4] # behavioral outcomes
    qout6 = n[5] # cue to action


 # change in height of tank   
    dndt1 = (qin1   - qout1) / tau1
    dndt2 = (qin2   - qout2) / tau2
    dndt3 = (qin3   - qout3) / tau3
    dndt4 = (qin4   - qout4) / tau4
    dndt5 = (qin5   - qout5) / tau5
    dndt6 = (qin6   - qout6) / tau6

 
    dndt = [dndt1,dndt2,dndt3, dndt4, dndt5, dndt6] 
    return dndt # deterivative of y at t0

 # integrate the equations
t = np.linspace(0,20) # times to report solution
#h0 = [5,5,5,5,5,5]
h0 = [0,0,0,0,0,0]

            # initial conditions for height
y = odeint(steps,h0,t) # integrate

f, ((x0, x1, x2), (x3, x4, x5)) = plt.subplots(2, 3)
x0.plot(t,y[:,0])
x0.set_title('Skills')
x1.plot(t,y[:,1])
x1.set_title('Impact Expectancy')
x2.plot(t,y[:,2])
x2.set_title('Confidence')
x3.plot(t,y[:,3])
x3.set_title('Behavior')
x4.plot(t,y[:,4])
x4.set_title('Behavioral Outcomes')
x5.plot(t,y[:,5])
x5.set_title('Cue to Action')

f.set_size_inches(18.5, 10.5)

plt.suptitle('Main title')
plt.show()