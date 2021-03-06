"""
PHYS-512 PS-1
Q3

chuyang Li
260744689

Error estimation: The interpolation is inaccurate, as the interpolation error 
displayed in my test below, which compares the results generated by the routine
with lakeshore.txt as input and the values from the text file, has a standard deviation of 133.29.
"""
import numpy as np

def lakeshore(V, data):
    n_entries = len(data) #number of entries
    x = np.empty(n_entries)
    y = np.empty(n_entries)
    for i in range(n_entries):
        x[i] = data[n_entries-i-1][1] #x: voltage, reversed the values
        y[i] = data[n_entries-1-i][0] #y: temperature, reversed the values
    dx = (x[-1]-x[0])/n_entries
    xx = np.linspace(x[2], x[-3], n_entries)
    for i in range(n_entries):
        ind=(xx[i]-x[0])/dx
        ind=int(np.floor(ind))
        x_use=x[ind-1:ind+3]
        y_use=y[ind-1:ind+3]
        p=np.polyfit(x_use,y_use,3) #cubic interpolation
    #If V is a number
    if np.isscalar(V):
        return np.polyval(p, V)
    #If V is an array
    yy = np.empty(len(V))
    for i in range(len(V)):
        yy[i] = np.polyval(p, V[i])
    return yy



#test
# dat=np.loadtxt("lakeshore.txt")
# n_entries = len(dat) #number of entries
# x = np.empty(n_entries)
# y = np.empty(n_entries)
# for i in range(n_entries):
#     x[i] = dat[n_entries-i-1][1] #x: voltage
#     y[i] = dat[n_entries-i-1][0] #y: temperature
# yy = lakeshore(x, dat)
# print(np.std(yy-y))