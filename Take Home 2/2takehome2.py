# Carter Colton

#%% Take Home 2 Problem 2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

# define all constants
c=900;
k=238;
N=20 # the number of grid points
a0=0
a=0.08
tau=.01
rho=2.7*10**3;
e=0.05;
finalt=49.55;

# make our grid and initialize T
r,h = np.linspace(a0,a,N+2,retstep = True) # make the grid
T = 900*np.ones((N+2,N+2))

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

# define while lop constants and while loop
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
    R[-1] = 300
    # Solve AT = R. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    
    T = la.solve(A,R)
    
    if (324<T[0,0]<326):
        finalt=t
        print(t)
        break
    
    # make plots every 25 time steps
    if p % 25 == 0:
        plt.clf() # clear the figure window
        plt.plot(r,T,'b-')
        plt.xlabel('r')
        plt.ylabel('T')
        plt.title('time={:1.3f},finalt={:1.3f} tau=0.01'.format(t,finalt))
        #plt.ylim([-1,1])
        plt.xlim([0,0.08])
        plt.show()
        #plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw