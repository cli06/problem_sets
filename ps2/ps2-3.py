"""
PHYS-512 PS-2
Q3

chuyang Li
260744689
"""
import numpy as np

#========== part a ==========
def log2_cheb(x):
    x_true = np.linspace(0.5, 1.0, 500)
    y_true = np.log2(x_true)
    x_use = np.linspace(-1.0, 1.0, 500)
    cheb_c = np.polynomial.chebyshev.chebfit(x_use, y_true, 10)
    max_err = 0
    for i in range(cheb_c.size-1, 0, -1):
        if max_err <= 10**-6:
            max_err = max_err + cheb_c[i]
        else:
            truncated_c = cheb_c[:i]
            break
    x_scaled = ((x-0.5)/0.5)*2 +(-1)
    y_cheb = np.polynomial.chebyshev.chebval(x_scaled, truncated_c)
    return y_cheb


# x_true = np.linspace(0.5, 1.0, 500)
# y_true = np.log2(x_true)
# x_use = np.linspace(-1.0, 1.0, 500)
# cheb_c = np.polynomial.chebyshev.chebfit(x_use, y_true, 10)
# max_err = 0
# for i in range(cheb_c.size-1, 0, -1):
#     if max_err <= 10**-6:
#         max_err = max_err + cheb_c[i]
#     else:
#         truncated_c = cheb_c[:i]
#         break
# y_cheb = np.polynomial.chebyshev.chebval(x_use, truncated_c)
# print('rms error for chebyshev is ',np.sqrt(np.mean((y_cheb-y_true)**2)),
#       ' with max error ',np.max(np.abs(y_cheb-y_true)))

# def x_scaled1(x):
#     return ((x-0.5)/0.5)*2 +(-1)

# def log2_cheb(x):
#     x_scaled = x_scaled1(x)
#     y_pred = np.polynomial.chebyshev.chebval(x_scaled, truncated_c)
#     return y_pred

#========== part b ==========
def mylog2(x):
    m, e = np.frexp(x)
    ln_x = (log2_cheb(m)+e)/np.log2(np.e)
    return ln_x

# print(np.log2(0.7), log2_cheb(0.7))
# print(np.log(0.6), mylog2(0.6))
