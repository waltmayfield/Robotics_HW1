import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def stepper(t,Ton,Toff):
    return np.piecewise(t, [t < Ton,  t >= Ton], [0, 1]) * np.piecewise(t, [t <= Toff,  t > Toff], [1, 0])
    

def volunteer(n,t):
    tau1 = 0.13
    tau2 = 0.20
    tau3 = 0.13 
    tau4 = 0.13
    tau5 = 0.13
    tau6 = 0.13
    tau7 = 0.13
    tau8 = 0.13
    tau9 = 0.13

# inputs    
    
#    z1 = 1*stepper(t,5,15)
#    z2 = 3.0
#    z3 = 4.0
#    z4 = 1.0
#    z5 = 2.3
#    z6 = 3.0
#    z7 = 4.0
#    z8 = 3.0
    
    z1 = 0
    z2 = 0
    z3 = 0
    z4 = 0
    z5 = 0
    z6 = 0
    z7 = 1.0*stepper(t,5,10)
    z8 = 0    
    
#    print(repr(t))
#    print(repr(z1))

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

# outflow resistance
    b31 = 0.1
    b21 = 0.5  # b31+b21 <= 1 
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

# inflows
    qin1 = g11*z1 + b31*n[3]
    qin2 = b21*n[0] + b25*n[4] + g22*z2
    qin3 = g33*z3 + g3*z2 + b31*n[0] + b38*n[7] - g333*z5
    qin4 = b46*n[5] + b43*n[2] + b42*n[1] + b47*n[6]
    qin5 = b54*n[3] + g55*z7
    qin6 = g66*z6 + g6*z4 + g666*z8 + b69*n[8]
    qin7 = b78*n[7]
    qin8 = b87*n[6] + b85*n[5]
    qin9 = g99*z3 + g999*z2 + g9*z1
  
    qout1 = n[0]
    qout2 = n[1]
    qout3 = n[2]
    qout4 = n[3]    
    qout5 = n[4]
    qout6 = n[5]
    qout7 = n[6]
    qout8 = n[7]
    qout9 = n[8]    

 # change in height of tank   
    dndt1 = (qin1   - qout1) / tau1
    dndt2 = (qin2   - qout2) / tau2
    dndt3 = (qin3   - qout3) / tau3
    dndt4 = (qin4   - qout4) / tau4
    dndt5 = (qin5   - qout5) / tau5
    dndt6 = (qin6   - qout6) / tau6
    dndt7 = (qin7   - qout7) / tau7
    dndt8 = (qin8   - qout8) / tau8
    dndt9 = (qin9   - qout9) / tau9
 
    dndt = [dndt1,dndt2,dndt3, dndt4, dndt5, dndt6, dndt7, dndt8, dndt9] 
    return dndt # deterivative of y at t0

 # integrate the equations
t = np.linspace(0,20) # times to report solution
#h0 = [5,5,5,5,5,5,5,5,5]            # initial conditions for height
h0 = [0,0,0,0,0,0,0,0,0]            # initial conditions for height
#h0 = np.arrange(1.,9.)           # initial conditions for height


str(volunteer(h0,1))

y = odeint(volunteer,h0,t) # integrate

f, ((x0, x1, x2), (x3, x4, x5), (x6, x7, x8)) = plt.subplots(3, 3)

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
x6.plot(t,y[:,6])
x6.set_title('Personal Gain Expectancy')
x7.plot(t,y[:,7])
x7.set_title('Mood')
x8.plot(t,y[:,8])
x8.set_title('Altruistic Tendencies')

f.set_size_inches(18.5, 10.5)

plt.suptitle('Main title')
plt.show()