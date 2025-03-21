# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:37:22 2023

@author: carte
"""

# build our A matrix
A = np.zeros((N+2,N+2)) # load a matrix A with zeros

for i in range(1,N+1): # build our matrix
    A[i,i-1] = -(r[i]**2)/(2*h**2)
    A[i,i] = (rho*c*r[i]**2)/(k*tau) - (2*r[i])/h + (r[i]**2)/(h**2)
    A[i,i+1] = -(r[i]**2)/(2*h**2)

A[0,0] = -1/h
A[0,1] = 1/h
A[-1,-1] = 0.5
A[-1,-2] = 0.5

# build our b matrix
b = np.zeros((N+2,N+2)) # load b with zeros

for i in range(1,N+1): # build our matrix
    b[i,i-1] = (r[i]**2)/(2*h**2)
    b[i,i] =  (rho*c*r[i]**2)/(k*tau) - (2*r[i])/h - (r[i]**2)/(h**2)
    b[i,i+1] = (r[i]**2)/(2*h**2)
    
#%% 

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True)

A = np.zeros((N+2,N+2)) # load a matrix A with zeros

for i in range(1,N+1): # build our matrix
    A[i,i-1] = 1/(h**2)
    A[i,i] = -2/(h**2) + np.sin(x[i])
    A[i,i+1] = 1/(h**2)

A[0,0] = 0.5;
A[0,1] = 0.5;
#A[-1,-3] = 3/(2*h);
#A[-1,-2] = -2/h;
#A[-1,-1] = 1/(2*h);
A[-1,-2] = -1/h;
A[-1,-1] = 1/h;

b = np.zeros(N+2) # load b with zeros

for i in range(1,N+1): # right hand side
    b[i] = np.cos(x[i])
    
b[0] = 0;
b[N+1] = b[N] - 5

ynexact = la.solve(A,b) # solve for a better y

plt.figure(1)
plt.plot(x,ynexact,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - 2.4b - x') 

#%%

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True)
y = np.cosh(x) + np.sin(x)*np.exp(x);

rmserr = 1;
B = np.zeros(N+2)
ynew = np.zeros_like(y)
while rmserr > (1*10**(-5)): # set a while loop to break once we reach the right accuracy
    ynew = la.solve(y,b)
    err = y@ynew-y
    
    rmserr = np.sqrt(np.mean(err[1:-2]**2))
    
