"""
PHYS-512 PS5
Q5
Chuyang Li
260744689
"""
import numpy as np
from matplotlib import pyplot as plt

#part c
def analytical_dft(k, k0, N):
    #Fourier transform of sine: https://mathworld.wolfram.com/FourierTransformSine.html
    myft = np.zeros(len(k), dtype="complex")
    e1 = (1-np.exp(-2*np.pi*1j*(k+k0)))/(1-np.exp(-2*np.pi*1j*(k+k0)/(N+1)))
    e2 = (1-np.exp(-2*np.pi*1j*(k-k0)))/(1-np.exp(-2*np.pi*1j*(k-k0)/(N+1)))
    myft = abs((1/2j) * (e1-e2))
    # for index in range(len(k)):
    #     e1 = np.cos(-2*np.pi*(k[index]+k0)*index)+1j*np.sin(2*np.pi*(k[index]+k0)*index)
    #     e2 = np.cos(-2*np.pi*(k[index]-k0)*index)+1j*np.sin(2*np.pi*(k[index]-k0)*index)
    #     myft[index] = abs((e1-e2)/2j)
    return myft

N = 50
k0 = 5.5
k = np.linspace(-N/2, N/2, N*5+1)
x = k.copy()
y = np.sin(2*np.pi*k0*x/len(x))
sin_myft = analytical_dft(k, k0, N)
sin_fft = np.fft.fft(y)
plt.clf()
plt.plot(k, sin_myft, label="analytical")
plt.plot(x, abs(sin_fft), label="FFT")
plt.legend(loc="upper left")
# plt.plot(x,y)
plt.show()

print(np.std(abs(sin_fft-sin_myft)))