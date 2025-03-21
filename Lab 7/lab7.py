# Carter Colton

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=3;
D=2;
N=20 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+1,retstep = True) # make our grid

y = np.sin(np.pi * x/L)

vy0 = np.zeros(N+2)

y[0] = 0
y[N] = 0

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

tau = 0.35*h**2/D;

y[0] = 0
y[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = y[1:-1] + D*tau/h**2 * (y[2:] -2*y[1:-1] + y[:-2])
    y = np.copy(ynew);
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f} C=0.35 err=0.00026'.format(t))
        plt.ylim([0,0.65])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

error = np.sqrt( np.mean( (y - analytic)**2 ))

print(error)

#%% Problem 7.1 N=40

L=3;
D=2;
N=40 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+1,retstep = True) # make our grid

y = np.sin(np.pi * x/L)

vy0 = np.zeros(N+2)

y[0] = 0
y[N] = 0

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

tau = 0.35*h**2/D;

y[0] = 0
y[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = y[1:-1] + D*tau/h**2 * (y[2:] -2*y[1:-1] + y[:-2])
    y = np.copy(ynew);
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f} C=0.35 err=4.5e-5'.format(t))
        plt.ylim([0,0.65])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

error = np.sqrt( np.mean( (y - analytic)**2 ))

print(error)

#%% Problem 7.1 N=80

L=3;
D=2;
N=80 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+1,retstep = True) # make our grid

y = np.sin(np.pi * x/L)

vy0 = np.zeros(N+2)

y[0] = 0
y[N] = 0

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

tau = 0.35*h**2/D;

y[0] = 0
y[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = y[1:-1] + D*tau/h**2 * (y[2:] -2*y[1:-1] + y[:-2])
    y = np.copy(ynew);
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f} C=0.35 err=8.44e-6'.format(t))
        plt.ylim([0,0.65])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

error = np.sqrt( np.mean( (y - analytic)**2 ))

print(error)

#%% Problem 7.1 D's

L=3;
D=2;
N=20 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+1,retstep = True) # make our grid

y = np.sin(np.pi * x/L)

vy0 = np.zeros(N+2)

y[0] = 0
y[N] = 0

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

tau = 0.35*h**2/D;

y[0] = 0
y[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = y[1:-1] + D*tau/h**2 * (y[2:] -2*y[1:-1] + y[:-2])
    y = np.copy(ynew);
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f} C=0.35 D makes things diffuse more'.format(t))
        plt.ylim([0,0.65])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

error = np.sqrt( np.mean( (y - analytic)**2 ))

print(error)

#%% Problem 7.1 Derivative BC

L=3;
D=2;
N=20 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = np.sin(np.pi * x/L)

y[0] = y[1] 
y[N+1] = y[N]

tau = 0.35*h**2/D;

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = y[1:-1] + D*tau/h**2 * (y[2:] -2*y[1:-1] + y[:-2])
    y = np.copy(ynew);
    
    y[0] = y[1] 
    y[N+1] = y[N]
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 50 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f} C=0.35 D makes things diffuse more'.format(t))
        plt.ylim([0,1])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

error = np.sqrt( np.mean( (y - analytic)**2 ))

print(error)

#%% Problem 7.2 a

tau2 = 0.5
tmax2 = 20
t2 = np.arange(0,tmax2,tau2)
y2 = np.zeros_like(t2)

y2[0] = 1

for i in range(len(t2)-1):
    y2[i+1] = y2[i] - tau2*y2[i]

plt.figure(1)
plt.plot(t2,y2,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eulers Method tau=0.5')

tau2 = 1
tmax2 = 20
t2 = np.arange(0,tmax2,tau2)
y2 = np.zeros_like(t2)

y2[0] = 1

for i in range(len(t2)-1):
    y2[i+1] = y2[i] - tau2*y2[i]

plt.figure(2)
plt.plot(t2,y2,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eulers Method tau=1')

tau2 = 2
tmax2 = 20
t2 = np.arange(0,tmax2,tau2)
y2 = np.zeros_like(t2)

y2[0] = 1

for i in range(len(t2)-1):
    y2[i+1] = y2[i] - tau2*y2[i]

plt.figure(3)
plt.plot(t2,y2,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eulers Method tau=2')

#%% Problem 7.2 b

tau2 = 4.5
tmax2 = 20
t2 = np.arange(0,tmax2,tau2)
y2 = np.zeros_like(t2)

y2[0] = 1

for i in range(len(t2)-1):
    #y2[i+1] = (y2[i]/(2*tau2) - y2[i]/2)/(1/tau2 - 1/(2*tau2) + 1/2)
    y2[i+1] = (1-tau2)*y2[i]/(1+tau2)

plt.figure(1)
plt.plot(t2,y2,'b')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Eulers Method tau=4')

#%% Problem 7.2 c

tau2 = 4.5
tmax2 = 20
t2 = np.arange(0,tmax2,tau2)
y2 = np.zeros_like(t2)

y2[0] = 1

for i in range(len(t2)-1):
    y2[i+1] = y2[i]/(1+tau2)

plt.figure(1)
plt.plot(t2,y2,'b')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Eulers Method tau=4')

#%% Problem 7.3 

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=3;
D=2;
N=100 # the number of grid points
a=0
b=L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = np.sin(np.pi * x/L)

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = 0
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1)) # load b with zeros

k=1
j=1

for i in b[1:N,0:N]: # build our matrix
    b[j,k]= 0
    b[j,j-1]=1
    b[j,j] =  2*h**2 / (tau*D) - 2
    b[j,j+1] = 1
    j=j+1
    k=k+1
   
p=0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    p=p+1
    t = t + tau
    # matrix multiply to get the right-hand side
    r = b@T
    # set r as appropriate for the boundary conditions
    r[0] = 0
    r[-1] = 0
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    T = la.solve(A,r)
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if p % 2 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('7.3 time={:1.3f} tau=2'.format(t))
        plt.ylim([0,1])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw


#%% Problem 7.3 c

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=3;
D=2;
N=100 # the number of grid points
a=0
b=L
tau=0.1

x,h = np.linspace(a,b,N+2,retstep = True) # make the grid

T = np.sin(np.pi * x/L)

T[0]=-T[1]
T[N+1] = -T[N]

A = np.zeros((N+2,N+2)) # load a matrix A with zeros

for i in range(1,N+1): # build our matrix
    A[i,i] = 0
    A[i,i-1] = -1
    A[i,i] = 2*h**2 / (tau*D) + 2
    A[i,i+1] = -1

A[0,0] = 0.5
A[0,1] = -0.5
A[-1,-1] = -0.5
A[-1,-2] = 0.5

b = np.zeros((N+2,N+2)) # load b with zeros

for i in range(1,N+1): # build our matrix
    b[i,i]= 0
    b[i,i-1]=1
    b[i,i] =  2*h**2 / (tau*D) - 2
    b[i,i+1] = 1
   
p=0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    p=p+1
    t = t + tau
    # matrix multiply to get the right-hand side
    r = b@T
    # set r as appropriate for the boundary conditions
    r[0] = 0
    r[-1] = 0
    # Solve AT = r. The T we get is for the next time step.
    # We don't need to keep track of previous T values, so just
    # load the new T directly into T itself
    
    T[0]=-T[1]
    T[N+1] = -T[N]
    
    T = la.solve(A,r)
    
    
    
    a = np.pi / L;
    gamma = a**2 * D;
    analytic = np.exp(-gamma*t)*np.sin(a*x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if p % 2 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,analytic,'or')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('7.3 time={:1.3f} tau=2'.format(t))
        plt.ylim([0,1])
        plt.xlim([0,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw