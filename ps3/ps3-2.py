"""
PHYS-512 PS-3
Q2

chuyang Li
260744689
"""
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

half_life = [1.41*(10**17), 2082240, 24120, 7.74*(10**12), 2.38*(10**12),
             5.05*(10**10), 330350.4, 186, 1608, 1194, 0.1643, 7.03*(10**8),
             1.58*(10**8), 11955686.4] #in seconds

def fun(x,y,half_life=half_life):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    for i in range(len(dydx)-2):
        dydx[i+1] = y[i]/half_life[i]-y[i+1]/half_life[i+1]
    dydx[len(dydx)-1] = y[len(dydx)-2]/half_life[len(dydx)-2]
    return dydx

y0 = np.zeros(len(half_life)+1)
y0[0] = 1
x0=0
x1=8*(10**17)
time = np.geomspace(1e-10, 8e17, num=1000)
ans=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau',t_eval=time)
length = ans.t.size
pb206_u238 = np.zeros(length)
th230_u234 = np.zeros(length)
for i in range(length):
    pb206_u238[i] = ans.y[-1,i]/ans.y[0,i] #Pb206/U238
    th230_u234[i] = ans.y[4,i]/ans.y[3,i] #Th230/U234
# print(ans.y[0,-1])
plt.ion()
plt.plot(ans.t, pb206_u238)
plt.xlim([3*(10**17),8*(10**17)])
plt.xlabel("time [s]")
plt.ylabel("Pb206/U238")
plt.title('Ratio of Pb206 to U238 over time')
plt.show()
plt.clf()
plt.plot(ans.t, th230_u234)
plt.xlim([10**11,10**14])
plt.xlabel("time [s]")
plt.ylabel("Th230/U234")
plt.title('Ratio of Th230 to U234 over time')
plt.show()

