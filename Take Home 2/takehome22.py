# Carter Colton

#%% Take Home 2 Problem 2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

c=900;
k=238;
N=20 # the number of grid points
a0=0
a=0.08
tau=.01
rho=2.7*10**3;
e=0.05;
sig=5.67*10**(-8)
finalt=25.68;

r,h = np.linspace(a0,a,N+2,retstep = True) # make the grid

T = 900*np.ones((N+2,N+2))

A = np.zeros((N+2,N+2)) # load a matrix A with zeros

for i in range(1,N+1): # build our matrix
    A[i,i-1] = -(r[i]**2)/(2*h**2)
    A[i,i] = (rho*c*r[i]**2)/(k*tau) - (2*r[i])/h + (r[i]**2)/(h**2)
    A[i,i+1] = -(r[i]**2)/(2*h**2)
    
    #A[i,i-1] = (1/(r[i]*h)) - (1/(2*h**2))
    #A[i,i] = (rho*c)/(tau) + 1/(h**2)
    #A[i,i+1] = -(1/(r[i]*h)) - (1/(2*h**2))

A[0,0] = -1/h
A[0,1] = 1/h
A[-1,-1] = 0.5
A[-1,-2] = 0.5

b = np.zeros((N+2,N+2)) # load b with zeros

for i in range(1,N+1): # build our matrix
    b[i,i-1] = (r[i]**2)/(2*h**2)
    b[i,i] =  (rho*c*r[i]**2)/(k*tau) - (2*r[i])/h - (r[i]**2)/(h**2)
    b[i,i+1] = (r[i]**2)/(2*h**2)
    
    #b[i,i-1] = -1/(r[i]*h) + (1/(2*h**2))
    #b[i,i] =  (rho*c)/(tau) - 1/(h**2)
    #b[i,i+1] =  1/(r[i]*h) + (1/(2*h**2))

p=0
t = 0
tmax = 50
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    p=p+1
    t = t + tau
    # matrix multiply to get the right-hand side
    R = b@T
    # set r as appropriate for the boundary conditions
    R[0] = 900
    R[-1] = -(sig*e*T[N+1])/k
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    #T[0]=-T[1]
    #T[N+1] = -T[N]
    
    
    
    T = la.solve(A,R)
    
    if (324<T[0,0]<326):
        finalt=t
        print(t)
        break
    
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if p % 25 == 0:
        plt.clf() # clear the figure window
        plt.plot(r,T,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},finalt={:1.3f} tau=0.01'.format(t,finalt))
        #plt.ylim([-1,1])
        plt.xlim([0,0.08])
        plt.show()
        #plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw