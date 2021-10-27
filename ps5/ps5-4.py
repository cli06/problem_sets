"""
PHYS-512 PS5
Q4
Chuyang Li
260744689
"""
import numpy as np
from matplotlib import pyplot as plt

def conv_safe(f, g):
    length = len(f) if len(f)>len(g) else len(g) 
    arr1 = np.zeros(2*length) #expand the array length to 2x the longer array and pad 0's
    arr2 = np.zeros(2*length) #expand the array length to 2x the longer array and pad 0's
    for i in range(len(f)):
        arr1[i] = f[i]
    for j in range(len(g)):
        arr2[j] = g[j]
    arr1 /= arr1.sum()
    arr2 /= arr2.sum()
    h = np.fft.irfft(np.fft.rfft(arr1)*np.fft.rfft(arr2), len(arr1))
    return h

x1 = np.linspace(-15,15,3001)
x2 = np.linspace(-5, 15, 2001)
f = np.exp(-0.5*(x1-0)**2/1**2)
g = np.exp(-0.5*(x2-2)**2/0.35**2)
h = conv_safe(f,g)
f = f/f.sum()
g = g/g.sum()

plt.clf()
plt.plot(f, label="function f")
plt.plot(g, label="function g")
plt.plot(h, label="function h = f*g")
plt.legend(loc="upper right")
plt.show()
