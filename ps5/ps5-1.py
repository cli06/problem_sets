"""
PHYS-512 PS5
Q1
Chuyang Li
260744689
"""
import numpy as np
from matplotlib import pyplot as plt

def conv_shift(array, shift):
    f = array
    g = 0*f
    g[int(round(shift))] = 1
    f = f/f.sum()
    g = g/g.sum()
    h = np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(f))
    return h

x=np.linspace(-15,15,3001)
f = np.exp(-0.5*(x-0)**2/1**2)
f = f/f.sum()
array = f
shift = len(array)/2
shifted_array = conv_shift(array, shift)

plt.clf()
plt.plot(x, array, "-b", label="original array")
plt.plot(x, shifted_array, "-r", label="shifted array")
plt.legend(loc="upper left")
plt.show()


        
