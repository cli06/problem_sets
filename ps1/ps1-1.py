"""
PHYS-512 PS-1
Q1 b)

chuyang Li
260744689
"""
import numpy as np

x = 3
eps = 10**-16

#f(x)=exp(x)
dx1 = eps**(1/3) #dx = epsilon^1/3
f1 = np.exp(x+dx1) #f(x+dx)
f2 = np.exp(x-dx1) #f(x-dx)
f3 = np.exp(x+2*dx1) #f(x+2dx)
f4 = np.exp(x-2*dx1) #f(x-2dx)
num_deriv1 = (8*f1-8*f2-f3+f4)/(12*dx1) #The numerical solution to d/dx exp(x)
ana_deriv1 = np.exp(x) #The analytical solution: d/dx exp(x) = exp(x)

#f(x)=exp(0.01x)
dx2 = eps**(1/3)*np.power(10, 4/3) #dx=(epsilon^1/3)*(10^4/3)
f5 = np.exp(0.01*(x+dx2)) #f(x+dx)
f6 = np.exp(0.01*(x-dx2)) #f(x-dx)
f7 = np.exp(0.01*(x+2*dx2)) #f(x+2dx)
f8 = np.exp(0.01*(x-2*dx2)) #f(x-2dx)
num_deriv2 = (8*f5-8*f6-f7+f8)/(12*dx2) #The numerical solution to d/dx exp(0.01x)
ana_deriv2 = 0.01*np.exp(0.01*x) #the analytical solution: d/dx exp(0.01x) = 0.01*exp(0.01x)

print('The total error of f(x)=exp(x) is', num_deriv1-ana_deriv1,
      '\nThe total error of f(x)=exp(0.01x) is', num_deriv2-ana_deriv2)



