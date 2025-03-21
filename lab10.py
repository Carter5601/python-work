# Carter Colton

#%% Problem 1

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

p = 1 + np.exp(-200*(x/L - 1/2)**2)

vy0 = np.ones(N+2)

p[0] = 1
p[N+1] = 1

c = 2;
tau = 0.2*h/c;

pnew = np.zeros_like(p)
j = 0
t = 0
tmax = 5
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    
    pnew[1:-1] = p[1:-1] - tau/(2*h) * (p[2:]*vy0[2:] - p[:-2]*vy0[:-2]);
    p = np.copy(pnew);
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-5,5])
        plt.xlim([0,10])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Problem 1 Second BC

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

p = 1 + np.exp(-200*(x/L - 1/2)**2)

vy0 = np.ones(N+2)

p[0] = 1
p[N+1] = 2*p[N]-p[N-1]

c = 2;
tau = 0.2*h/c;

pnew = np.zeros_like(p)
j = 0
t = 0
tmax = 5
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    
    pnew[1:-1] = p[1:-1] - tau/(2*h) * (p[2:]*vy0[2:] - p[:-2]*vy0[:-2]);
    p = np.copy(pnew);
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-5,5])
        plt.xlim([0,10])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Problem 2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

p = 1 + np.exp(-200*(x/L - 1/2)**2)

v0 = 5;

tau = 0.001;

pnew = np.zeros_like(p)
j = 0
t = 0
tmax = 5
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    
    pnew[1:-1] = p[1:-1] - ((v0*tau)/(2*h)) * (p[2:] - p[:-2]) + ((v0**2 * tau**2)/(2*h**2)) * (p[2:] - 2*p[1:-1] + p[:-2]);
    p = np.copy(pnew);
    p[0] = 2 - p[1]
    p[N+1] = 2*p[N]-p[N-1]
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 100 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-5,5])
        plt.xlim([0,10])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Problem 2 Function

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

def laxwendrof(tau,timeiterator):
    L=10;
    N=400 # the number of grid points
    a=0
    b=L
    x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

    p = 1 + np.exp(-200*(x/L - 1/2)**2)

    v0 = 5;


    pnew = np.zeros_like(p)
    j = 0
    t = 0
    tmax = 5
    plt.figure(1) # Open the figure window
    # the loop that steps the solution along
    while t < tmax:
        j = j+1
        t = t + tau
        
        
        pnew[1:-1] = p[1:-1] - ((v0*tau)/(2*h)) * (p[2:] - p[:-2]) + ((v0**2 * tau**2)/(2*h**2)) * (p[2:] - 2*p[1:-1] + p[:-2]);
        p = np.copy(pnew);
        p[0] = 2 - p[1]
        p[N+1] = 2*p[N]-p[N-1]
       # Use leapfrog and the boundary conditions to load
        # ynew with y at the next time step using y and yold
        # update yold and y for next timestep
        # remember to use np.copy
        # make plots every 50 time steps
        if j % timeiterator == 0:
            plt.clf() # clear the figure window
            plt.plot(x,p,'b-')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('time={:1.3f} tau={:1.6f}'.format(t,tau))
            plt.ylim([-5,5])
            plt.xlim([0,10])
            #plt.draw() # Draw the plot
            plt.pause(0.1) # Give the computer time to draw
    return

#%% Problem 2 tau = 0.001
laxwendrof(0.001,100)

#%% Problem 2 tau = 0.0001
laxwendrof(0.0001,1000)

#%% Problem 2 tau = 0.0001
laxwendrof(0.00001,10000)

#%% Problem 2 tau = 0.0025
laxwendrof(0.0025,50)

#%% Problem 2 tau = 0.005
laxwendrof(0.005,50)

#%% Problem 2 tau = 0.01
laxwendrof(0.01,50)


#%% Problem 3 

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
tau=0.001
vy0 = 1;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

C1 = (-tau/(8*h))*(vy0 + vy0)
C2 = (-tau/(8*h))*(vy0 + vy0)

p = 1 + np.exp(-200*(x/L - 1/2)**2)

p[0] = 1;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = C2
    A[c,c] = 1
    A[c,c+1] = -C1
    c = c + 1
    d = d + 1

A[0,0] = 0.5
A[0,1] = 0.5
A[-1,-1] = 1
A[-1,-2] = -2
A[-1,-3] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]= -C2
    b[j1,j1] =  1
    b[j1,j1+1] = C1
    
    j1=j1+1
    k=k+1

q=0
t = 0
tmax = 10
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
    t = t + tau
    # matrix multiply to get the right-hand side
    r = b@p
    # set r as appropriate for the boundary conditions
    r[0] = 1
    r[-1] = 0
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    p = la.solve(A,r)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 100 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('8.3 time={:1.3f}, tau={:1.3f}'.format(t,tau))
        plt.ylim([-4,4])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Problem 3 c

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
tau=0.01
vy0 = 1.2 - x/L;
C1 = (-tau/(8*h))*(vy0 + vy0)
C2 = (-tau/(8*h))*(vy0 + vy0)

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

p = 1 + np.exp(-200*(x/L - 1/2)**2)

p[0] = 1;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = (-tau/(8*h))*(vy0[c-1] + vy0[c-1])
    A[c,c] = 1
    A[c,c+1] = -(-tau/(8*h))*(vy0[c+1] + vy0[c+1])
    c = c + 1
    d = d + 1

A[0,0] = 0.5
A[0,1] = 0.5
A[-1,-1] = 1
A[-1,-2] = -2
A[-1,-3] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]= -(-tau/(8*h))*(vy0[k-1] + vy0[k-1])
    b[j1,j1] =  1
    b[j1,j1+1] = (-tau/(8*h))*(vy0[k+1] + vy0[k+1])
    
    j1=j1+1
    k=k+1

q=0
t = 0
tmax = 10
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
    t = t + tau
    # matrix multiply to get the right-hand side
    r = b@p
    # set r as appropriate for the boundary conditions
    r[0] = 1
    r[-1] = 0
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    p = la.solve(A,r)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 20 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('8.3 time={:1.3f}, tau={:1.3f}'.format(t,tau))
        plt.ylim([0,7])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Problem 3d

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
tau=0.01
vy0 = 1;
C1 = (-tau/(8*h))*(vy0 + vy0)
C2 = (-tau/(8*h))*(vy0 + vy0)

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

p = np.zeros(N+1)

for i in range(1,N+1):
    if 0 <= x[i] <= L/2:
        p[i] = 1
    else:
        p[i] = 0;    

A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = C2
    A[c,c] = 1
    A[c,c+1] = -C1
    c = c + 1
    d = d + 1

A[0,0] = 0.5
A[0,1] = 0.5
A[-1,-1] = 1
A[-1,-2] = -2
A[-1,-3] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]= -C2
    b[j1,j1] =  1
    b[j1,j1+1] = C1
    
    j1=j1+1
    k=k+1
        
q=0
t = 0
tmax = 10
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
    t = t + tau
    # matrix multiply to get the right-hand side
    r = b@p
    # set r as appropriate for the boundary conditions
    r[0] = 1
    r[-1] = 0
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    p = la.solve(A,r)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 20 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('8.3 time={:1.3f}, tau={:1.3f}'.format(t,tau))
        plt.ylim([-4,4])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
      
#%% Problem 10.3e

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

p = np.zeros(N+2)

for i in range(1,N+2):
    if 0 <= x[i] <= L/2:
        p[i] = 1
    else:
        p[i] = 0;

v0 = 5;


tau = 0.001;

pnew = np.zeros_like(p)
j = 0
t = 0
tmax = 5
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    
    pnew[1:-1] = p[1:-1] - ((v0*tau)/(2*h)) * (p[2:] - p[:-2]) + ((v0**2 * tau**2)/(2*h**2)) * (p[2:] - 2*p[1:-1] + p[:-2]);
    p = np.copy(pnew);
    p[0] = 2 - p[1]
    p[N+1] = 2*p[N]-p[N-1]
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 100 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,p,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-5,5])
        plt.xlim([0,10])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw