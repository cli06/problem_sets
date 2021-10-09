"""
PHYS-512 PS-3
Q1

chuyang Li
260744689

Sincer there are four function calls per step in rk4_step, there are 12 function 
calls at each step in rk4_stepd. The modified stepper rk4_stepd is more accurate by 
roughly 400 times, as illustrated by the example in this problem.
"""
import numpy as np
from matplotlib import pyplot as plt

def dydx(x,y):
    #dy/dx = y/1+x^2 --> c0*exp(arctan(x))
    return y/(1+x**2)

def rk4_step(fun, x, y, h):
    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h, y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return dy

def rk4_stepd(fun, x, y, h):
    y1 = rk4_step(fun, x, y, h)
    y2a = rk4_step(fun, x, y, h/2)
    y2b = rk4_step(fun, x+h/2, y+y2a, h/2)
    y2 = y2a + y2b
    #Since there are four function calls per step in rk4_step,
    #there are 12 function calls per step in rk4_stepd
    #y1 = y_true + err
    #y2 = y_true + err/2^4
    #y_true = (16y2-y1)/15
    return (16*y2-y1)/15

npts = 200
x = np.linspace(-20, 20, npts)
y1 = np.zeros(npts) #using rk4_step
y2 = np.zeros(npts) #using rk4_stepd
y1[0] = 1 #y(-20)=1
y2[0] = 1 #y(-20)=1
y_real = 4.576058010298909*np.exp(np.arctan(x)) #c0 = 4.576058010298909 for y(-20)=1
for i in range(npts-1):
    h=x[i+1]-x[i]
    y1[i+1] = y1[i]+rk4_step(dydx,x[i],y1[i],h)
    y2[i+1] = y2[i]+rk4_stepd(dydx,x[i],y2[i],h)

print("The errors of rk4_step and rk4_stepd are {} and {}, respectively ".format(np.std(y_real-y1), 
                                                                                 np.std(y_real-y2)))
plt.ion()
plt.plot(x, y1-y_real)
plt.plot(x, y2-y_real)
plt.legend(["rk4_step", "rk4_stepd"])
plt.title('RK4 error, {} points'.format(npts))
plt.show()
# plt.title('RK4 error, ' + repr(npt)+ ' points')
# plt.savefig('rk4_err.png')


    




