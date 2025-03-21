# Carter Colton

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=200 # the number of grid points
a=-L
b=L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

norm=np.trapz(np.conjugate(T)*T,x)

print(norm)

T[-L] = 0;
T[L]=0;

V = np.zeros(N+1) # load b with zeros
        
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 5
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 2 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.2a time={:1.3f},norm={:1.3f} tau=0.1'.format(t,norm))
        plt.ylim([-1,1])
        plt.xlim([-L,L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%%

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
h=1;
m=1;
sig=0.5;
p=2*np.pi;
L=10;
D=h*j/(2*m);
N=200 # the number of grid points
a=-L
b=L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V = np.zeros(N+1) # load b with zeros
        
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
norm3=np.zeros((N+1),dtype=np.complex_)
q=0
t = 0
tmax = 20 
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
    
    norm3[q]=np.trapz(np.conjugate(T)*T*x,x)
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps

t,h3 = np.linspace(0,50,N+1,retstep = True) # make the grid
plt.figure(2) # clear the figure window
plt.plot(t,norm3,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('8.2c')
plt.ylim([-10,10])
plt.xlim([0,50])
plt.draw() # Draw the plot
plt.pause(0.1) # Give the computer time to draw

#%% Problem 8.3 Height

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=2;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.98*L and x[n]<=L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.3 time={:1.3f},norm={:1.3f},V0={:1.1f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

#%% Continued Height

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=0.1;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.98*L and x[n]<=L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.3 Height time={:1.3f},norm={:1.3f},V0={:1.1f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Continued Height

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=0.5;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.98*L and x[n]<=L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.2 Height time={:1.3f},norm={:1.3f},V0={:1.1f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
      
#%% Continued Height

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=10;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.98*L and x[n]<=L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.3 Height time={:1.3f},norm={:1.3f},V0={:1.1f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% Width

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=0.5;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.5*L and x[n]<=1.5*L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.3 Width time={:1.3f},norm={:1.3f},V0={:1.1f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
       
#%% Height and Width

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

j=complex(0,1);
hbar=1;
m=1;
sig=2;
p=2*np.pi;
L=10;
D=hbar*j/(2*m);
N=300 # the number of grid points
a=-2*L
b=3*L
tau=0.1

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

T = 1/(np.sqrt(sig*np.sqrt(np.pi))) *np.exp(j*p*x)*np.exp(-x**2/(2*sig**2))

T[-L] = 0;
T[L]=0;

V0=0.01;
V = np.zeros(N+1) # load b with zeros
n=1;
for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.5*L and x[n]<=1.5*L:
        V[n]=V0
    else:
        f=0;

overlay = np.zeros(N+1)
l=1;
for l in range(1,N):
    l=l+1
    overlay[l]=V[l]/V0;
    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros
c=1;
d=0;

for i in A[1:N,0:N]: # build our matrix
    A[c,c-1] = -1
    A[c,c] = 2*h**2 / (tau*D) + 2 +V[c]/(2*h*j)
    A[c,c+1] = -1
    c = c + 1
    d = d + 1

A[0,0] = 1
A[0,1] = 1
A[-1,-1] = 1
A[-1,-2] = 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

k=1
j1=1

for i in b[1:N,0:N]: # build our matrix
    b[j1,j1-1]=1
    b[j1,j1] =  2*h**2 / (tau*D) - 2 - V[k]/(2*h*j)
    b[j1,j1+1] = 1
    j1=j1+1
    k=k+1
   
q=0
t = 0
tmax = 20
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    q=q+1
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
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if q % 5 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,T,'b-',x,np.conjugate(T)*T,'r',x,overlay,'m')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('8.3 H&W time={:1.3f},norm={:1.3f},V0={:1.2f} tau=0.1'.format(t,norm,V0))
        plt.ylim([-0.5,0.5])
        plt.xlim([-2*L,3*L])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

