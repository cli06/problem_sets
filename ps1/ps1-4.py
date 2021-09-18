"""
PHYS-512 PS-1
Q4

chuyang Li
260744689
"""
import numpy as np
from scipy import interpolate

#Helper functions for rational function interpolation
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

def rat_fit2(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y) #Replace linag.inv with linalg.pinv
    p=pars[:n]
    q=pars[n:]
    return p,q

#==========Part 1 -> f(x) = cos(x)==========
x_min = -np.pi/2
x_max = np.pi/2
x_npts = 11 #Number of points for x
xx_npts = 501 #Number of points for xx
x1 = np.linspace(x_min, x_max, x_npts)
y1 = np.cos(x1)
xx1 = np.linspace(x1[0], x1[-1], xx_npts) 
y_true1 = np.cos(xx1)

#polynomial
p1 = np.polyfit(x1, y1, 3)
yy_p1 = [np.polyval(p1, xx1[i]) for i in range(len(xx1))]
print('cos(x): Error in polynomial interpolation: ', np.std(y_true1-yy_p1))

#Cublic spline
spln1=interpolate.splrep(x1,y1)
yy_s1=interpolate.splev(xx1,spln1)
print('cos(x): Error in cubic spline interpolation: ', np.std(y_true1-yy_s1))

#Raional function
n1=5
m1=7
p1,q1=rat_fit(x1,y1,n1,m1)
yy_r1=rat_eval(p1,q1,xx1)
print('cos(x): Error in rational function interpolation: ', np.std(y_true1-yy_r1))

#==========Part 2 -> Lorentzian==========
x2 = np.linspace(x_min, x_max, x_npts)
y2 = np.empty(len(x2))
for i in range(len(x2)):
    y2[i] = 1/(1+x2[i]**2)
xx2 = np.linspace(x2[0], x2[-1], xx_npts) 
y_true2 = np.empty(len(xx2))
for i in range(len(xx2)):
    y_true2[i] = 1/(1+xx2[i]**2)

#polynomial
p2 = np.polyfit(x2, y2, 3)
yy_p2 = [np.polyval(p2, xx2[i]) for i in range(len(xx2))]
print('Lorentzian: Error in polynomial interpolation: ', np.std(y_true2-yy_p2))

#Cublic spline
spln2=interpolate.splrep(x2,y2)
yy_s2=interpolate.splev(xx2,spln2)
print('Lorentzian: Error in cubic spline interpolation: ', np.std(y_true2-yy_s2))

#Raional function
n2=5
m2=7
p2,q2=rat_fit(x2,y2,n2,m2)
yy_r2=rat_eval(p2,q2,xx2)
p2_mod, q2_mod=rat_fit2(x2,y2,n2,m2)
yy_r2mod=rat_eval(p2_mod,q2_mod,xx2)
print('Lorentzian: Error in rational function interpolation: ', np.std(y_true2-yy_r2),
      '\nLorentzian(modified): Error in rational function interpolation: ', np.std(y_true2-yy_r2mod),)

