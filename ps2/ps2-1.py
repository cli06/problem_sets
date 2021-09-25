"""
PHYS-512 PS-2
Q1

chuyang Li
260744689
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def simpsons(fun, a, b):
    x=np.linspace(a, b, 5)
    y = fun(x)
    dx=(b-a)/(len(x)-1)
    area=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    return area

def trap(fun, a, b):
    x = np.linspace(a,b,20) # N+1 points make N subintervals
    y = fun(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/19
    T = (dx/2) * np.sum(y_right + y_left)
    return T

global R, z
R = 120 #radius of a hydrogen atom -> 120pm

def func(u):
    numerator = z-R*u
    denominator = ((R**2)+(z**2)-2*R*z*u)**(3/2)
    return numerator/denominator

# result_s = simpsons(func, -1, 1)
# result_q = integrate.quad(func, -1, 1)[0]

results_s = np.empty(200)
results_q = np.empty(200)
results_t = np.empty(200)
for i in range(200):
    z = i
    results_s[i] = simpsons(func, -1, 1)
    results_q[i] = integrate.quad(func, -1, 1)[0]
    results_t[i] = trap(func, -1, 1)

plt.plot(results_s)
# plt.plot(results_s2)
plt.xlabel('distance from center')
plt.ylabel('electric field')
ax=plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.show()



