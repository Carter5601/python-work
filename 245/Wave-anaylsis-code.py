# Wessley, Carter, Ian

##  LoadWaveFile.py
##  Dallin S. Durfee 2016
##  
##  This file was written for Physics 245
##  to allow students to load wave files
##  as numpy arrays so that they can do
##  data analysis, etc.
##
##  The function "loadwave(fname,navg=1)" 
##  loads the wave file with filename fname.
##  The optional parameter navg will average
##  the data into bins of size navg.
##  A tuple is returned.  The first element
##  is a float value indicating the spacing
##  in time between data points.  The second
##  element is a numpy array of all of the
##  data.
##
##  The function "binavg(data,navg)" is used
##  by "loadwave" to do averaging.

import wave, struct
import numpy as np


def binavg(data,navg):
	N = data.shape[0]
	N = int(N/navg)*navg
	d = data[:N]
	return(d.reshape(-1,navg).mean(axis=1))
	

def loadwave(fname,navg=1):
	wavefile = wave.open(fname,'r')
	(nchannels, sampwidth, framerate, nframes, comptype, compname) = wavefile.getparams ()
	if(nchannels > 1):
		print("Only mono data please!")
		
	if(nchannels < 1):
		print("No data!")
		
	dt = 1.0/framerate
	N = nframes*nchannels
	frames = wavefile.readframes(N)
	data=np.array(struct.unpack_from ("%dh" % N, frames))
	#print(fname+" data shape = %s" % data.shape)
	data = binavg(data,navg)
	#print(fname+" data shape = %s" % data.shape)
	return(dt*navg,data)

import matplotlib.pyplot as plt
def fploter(filename):
    dt, wave1 = loadwave(filename)
    t= np.arange(0,len(wave1)*dt,dt)
    plt.figure
    plt.plot(t, wave1)
    fdata=np.fft.fft(wave1)
    wdata=np.fft.fftfreq(len(wave1), dt)
    pdata= (np.abs(fdata))**2
    plt.figure()
    plt.plot(wdata, pdata)
    plt.title(" Power specturm of of data")
    plt.xlabel("frequency")
    plt.ylabel("Power")
    return(wdata, pdata)

fploter("newdata.wav")