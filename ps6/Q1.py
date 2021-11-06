"""
PHYS-512 PS6
Q1

a):
    All strain data points are assumed to represent noise, as the amount of signal is nearly negligible in the data.
    Therefore, the each set of strains were convereted to power spectrums using Welch's method (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#r34b375daf612-1).
    The Tukey functions was used as the window function because it has a wide flat area around the center (https://community.sw.siemens.com/s/article/window-types-hanning-flattop-uniform-tukey-and-exponential).
    The average of the power spectrums of each detector can be considered the noise model of it, since the weak signals were removed by averaging the 
    values of PSD of each data set.
    The noise power spectrums were smoothed by calculating their moving averages.
    
Chuyang Li
260744689
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import json
import h5py
import os

location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))+'\\' #path of the current directory with data files in it

#---------------------------PART A---------------------------
#------------------------------------------------------------
#functions from simple_read_ligo.py
def read_template(filename): 
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename): 
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

#Hanford
fname1 = 'H-H1_LOSC_4_V1-1167559920-32.hdf5'
fname2 = 'H-H1_LOSC_4_V2-1126259446-32.hdf5'
fname3 = 'H-H1_LOSC_4_V2-1128678884-32.hdf5'
fname4 = 'H-H1_LOSC_4_V2-1135136334-32.hdf5'
strain_h1,dt1,utc1 = read_file(location+fname1)
strain_h2,dt2,utc2 = read_file(location+fname2)
strain_h3,dt3,utc3 = read_file(location+fname3)
strain_h4,dt4,utc4 = read_file(location+fname4)
fs = 1/dt1
h_frequencies, h_pxx1 = signal.welch(x=strain_h1, fs=fs, window=('tukey', 0.05), nperseg=2048)
h_frequencies, h_pxx2 = signal.welch(x=strain_h2, fs=fs, window=('tukey', 0.05), nperseg=2048)
h_frequencies, h_pxx3 = signal.welch(x=strain_h3, fs=fs, window=('tukey', 0.05), nperseg=2048)
h_frequencies, h_pxx4 = signal.welch(x=strain_h4, fs=fs, window=('tukey', 0.05), nperseg=2048)
h_pxx = (h_pxx1+h_pxx2+h_pxx3+h_pxx4)/4 #average of 4 psd's
h_pxx_smooth = np.convolve(h_pxx, np.ones(4), mode='same')/4 #smooth by getting the moving average
# h_pxx_intp = interp1d(h_frequencies, h_pxx_smooth)
plt.clf()
plt.loglog(h_frequencies, h_pxx, "-b", label='noise psd')
plt.loglog(h_frequencies, h_pxx_smooth, "-r", label='smooth noise psd')
plt.legend(loc='upper right')
plt.title('Hanford')
plt.show()

#Livingston
fname5 = 'L-L1_LOSC_4_V1-1167559920-32.hdf5'
fname6 = 'L-L1_LOSC_4_V2-1126259446-32.hdf5'
fname7 = 'L-L1_LOSC_4_V2-1128678884-32.hdf5'
fname8 = 'L-L1_LOSC_4_V2-1135136334-32.hdf5'
strain_l1,dt1,utc1 = read_file(location+fname5)
strain_l2,dt2,utc2 = read_file(location+fname6)
strain_l3,dt3,utc3 = read_file(location+fname7)
strain_l4,dt4,utc4 = read_file(location+fname8)
fs = 1/dt1
l_frequencies, l_pxx1 = signal.welch(x=strain_l1, fs=fs, window=('tukey', 0.05), nperseg=2048)
l_frequencies, l_pxx2 = signal.welch(x=strain_l2, fs=fs, window=('tukey', 0.05), nperseg=2048)
l_frequencies, l_pxx3 = signal.welch(x=strain_l3, fs=fs, window=('tukey', 0.05), nperseg=2048)
l_frequencies, l_pxx4 = signal.welch(x=strain_l4, fs=fs, window=('tukey', 0.05), nperseg=2048)
l_pxx = (l_pxx1+l_pxx2+l_pxx3+l_pxx4)/4 #average of 4 psd's
l_pxx_smooth = np.convolve(l_pxx, np.ones(4), mode='same')/4 #smooth by getting the moving average
plt.clf()
plt.loglog(l_frequencies, l_pxx, "-b", label='noise psd')
plt.loglog(l_frequencies, l_pxx_smooth, "-r", label='smooth noise psd')
plt.legend(loc='upper right')
plt.title('Livingston')
plt.show()

