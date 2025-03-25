#!/usr/bin/python
#%matplotlib inline
#
# Import needed libraries
#
import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})
#
# Read in output of ddPSI
#
#data=np.genfromtxt('log_quench.lammps', usecols=[0,1,2,3,4,5,6], skip_header=4, skip_footer=28)
data=np.genfromtxt('/Users/leonardodelgaudio/Documents/14_Master/Msc_SS25/03_Multi/Multiscale_Msc_SS25/W6/md_lammps/lammps/exercise4/log_quench_1nanosec.lammps', usecols=[0,1,2,3,4,5,6], skip_header=4, skip_footer=28)
time=[x[1] for x in data]
temperature=[x[2] for x in data]
energy=[(x[3]/500.0) for x in data]
pressure=[x[5] for x in data]
volume=[(x[6]/500.0) for x in data]
#
# Use subplots to output all data as figures
#
fig, panel = plt.subplots(2,2)
#
# Plot temperature versus time
#
panel[0,0].set(xlabel='time', ylabel='temperature')
panel[0,0].scatter(time,temperature,s=5.0)
#
# Plot energy versus time
#
panel[0,1].set(xlabel='temperature', ylabel='energy per atom')
panel[0,1].scatter(temperature,energy,s=5.0)
#
# Plot pressure versus time
#
panel[1,0].set(xlabel='temperature', ylabel='pressure')
panel[1,0].scatter(temperature,pressure,s=5.0)
#
# Plot volume versus time
#
panel[1,1].set(xlabel='temperature', ylabel='volume per atom')
panel[1,1].scatter(temperature,volume,s=5.0)

fig.set_figheight(20)
fig.set_figwidth(15)
fig.tight_layout()
plt.show()

