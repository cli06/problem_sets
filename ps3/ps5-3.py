"""
PHYS-512 PS5
Q3
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

def correlation(arr1, arr2):
    f = arr1
    g = arr2
    f = f/f.sum()
    g = g/g.sum()
    return np.fft.irfft(np.fft.rfft(f)*np.conj(np.fft.rfft(g)),len(f))

x = np.linspace(-15,15,3001)
f = np.exp(-0.5*(x-0)**2/1**2)
f = f/f.sum()
array = f

plt.clf()
plt.plot(x, array, "-b", label="original array")
shift = len(array)/100
shifted_array = conv_shift(array,shift)
plt.plot(x, shifted_array, "-r", label="shifted array")
corr = correlation(shifted_array, shifted_array)
plt.plot(x, corr, "-g", label="correlation, shift = %.3f" %(int(round(shift))/len(x)))
plt.legend(loc="upper left")
plt.show()
