"""
PHYS-512 PS-1
Q2

chuyang Li
260744689
"""
def ndiff(fun, x, full=False):
    eps = 10**-16 #Machine precision
    dx = eps**(1/3)*x #dx~eps^(1/3)*xc, with xc=x
    num_deriv = (fun(x+dx)-fun(x-dx))/(2*dx)
    err = (eps**(2/3))*(fun(x)**(2/3)) #The estimated fractional error, ignoring f' and f'''
    if full == True:
        return num_deriv, dx, err
    else:
        return num_deriv
