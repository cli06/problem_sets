"""
PHYS-512 PS5
Q2
Chuyang Li
260744689
"""
import numpy as np
from matplotlib import pyplot as plt

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
corr = correlation(f, f)

plt.clf()
plt.plot(x, array, "-b", label="Gaussian")
plt.plot(x, corr, "-r", label="Correlation")
plt.legend(loc="upper left")
plt.show()



