"""
PHYS-512 PS-2
Q2

chuyang Li
260744689
"""
import numpy as np

# def lorentz(x):
#     return 1.0/(1.0+x**2)

def integrate_adaptive(fun, a, b, tol, extra=None):
    # print('integrating between ', a, b)
    x=np.linspace(a, b, 5)
    if extra is None:
        y = fun(x)
    else:
        y = np.empty(5)
        y[0] = extra[0]
        y[1] = fun(x[1])
        y[2] = extra[1]
        y[3] = fun(x[3])
        y[4] = extra[2]
    dx=(b-a)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(a+b)/2
        y_left = np.empty(3)
        y_right = np.empty(3)
        for i in range(4):
            if i <= 2:
                y_left[i] = y[i]
            if i >= 2:
                y_right[i-2] = y[i]
        left=integrate_adaptive(fun,a,xmid,tol/2,y_left)
        right=integrate_adaptive(fun,xmid,b,tol/2,y_right)
        return left+right

# x0=-100
# x1=100
# if False:
#     ans=integrate_adaptive(np.exp,x0,x1,1e-7, None)
#     print(ans-(np.exp(x1)-np.exp(x0)))
# else:
#     ans=integrate_adaptive(lorentz,x0,x1,1e-7, None)
#     print(ans-(np.arctan(x1)-np.arctan(x0)))




