# Carter Colton - Lab 5

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = 0.01 * np.exp(-(x-L/2)**2 / 0.02)

vy0 = np.zeros(N+2)

y[0] = -y[1]
y[N+1] = -y[N]

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = 0.2*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

yold = np.zeros(N+2)

yold[1:-1] = y[1:-1] - vy0[1:-1]*tau + (c**2 * tau**2)/(2*h**2) *(y[2:] -2*y[1:-1] + y[:-2]);

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
    
    ynew[1:-1] = 2*y[1:-1] - yold[1:-1] + (c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2])
    yold = np.copy(y);
    y = np.copy(ynew);
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
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% 5.3 d

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = 0.01 * np.exp(-(x-L/2)**2 / 0.02)

vy0 = np.zeros(N+2)

y[0] = -y[1]
y[N+1] = -y[N]

plt.figure(2)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

yold = np.zeros(N+2)

yold[1:-1] = y[1:-1] - vy0[1:-1]*tau + (c**2 * tau**2)/(2*h**2) *(y[2:] -2*y[1:-1] + y[:-2]);

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
    
    ynew[1:-1] = 2*y[1:-1] - yold[1:-1] + (c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2])
    yold = np.copy(y);
    y = np.copy(ynew);
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
        plt.title('time={:1.3f} tau=1'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% 5.3 d continued

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = 0.01 * np.exp(-(x-L/2)**2 / 0.02)

vy0 = np.zeros(N+2)

y[0] = -y[1]
y[N+1] = -y[N]

plt.figure(3)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = 1.1*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

yold = np.zeros(N+2)

yold[1:-1] = y[1:-1] - vy0[1:-1]*tau + (c**2 * tau**2)/(2*h**2) *(y[2:] -2*y[1:-1] + y[:-2]);

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
    
    ynew[1:-1] = 2*y[1:-1] - yold[1:-1] + (c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2])
    yold = np.copy(y);
    y = np.copy(ynew);
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
        plt.title('time={:1.3f} tau > h/c'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% 5.3 e

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = 0.01 * np.exp(-(x-L/2)**2 / 0.02)

vy0 = np.zeros(N+2)

plt.figure(4)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = 0.2*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

yold = np.zeros(N+2)

yold[1:-1] = y[1:-1] - vy0[1:-1]*tau + (c**2 * tau**2)/(2*h**2) *(y[2:] -2*y[1:-1] + y[:-2]);

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure() # Open the figure window
# the loop that steps the solution along
while t < tmax:
    
    j = j+1
    t = t + tau
    
    ynew[1:-1] = 2*y[1:-1] - yold[1:-1] + (c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2])
    ynew[0] = ynew[1]
    ynew[N+1] = ynew[N]
    yold = np.copy(y);
    y = np.copy(ynew);
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
        plt.title('Derivative BCs time={:1.3f} tau > 0.2'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

#%% 5.3 f

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = np.zeros(N+2)

vy0 = 0.1 * np.exp(-(x-L/2)**2 / 0.02)

plt.figure(5)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = 0.2*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

yold = np.zeros(N+2)

yold[1:-1] = y[1:-1] - vy0[1:-1]*tau + (c**2 * tau**2)/(2*h**2) *(y[2:] -2*y[1:-1] + y[:-2]);

vy0[0] = 0
vy0[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 2
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = 2*y[1:-1] - yold[1:-1] + (c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2])
    yold = np.copy(y);
    y = np.copy(ynew);
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
        plt.title('Initial Velocity time={:1.3f} tau = 0.2'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw
        
#%% 5.4 c

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1;
N=200 # the number of grid points
a=0
b=L
x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = 0.01 * np.exp(-(x-L/2)**2 / 0.02)

vy0 = np.zeros(N+2)

y[0] = -y[1]
y[N+1] = -y[N]

plt.figure(6)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE')

c = 2;
tau = 0.2*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

gamma = 0.2;
yold = np.zeros(N+2)

f=vy0[1:-1]

yold[1:-1] = (-2*vy0[1:-1]*tau + ((1/(2 + gamma*tau))*((4*y[1:-1])+((2*c**2 * tau**2)/(h**2))\
                *(y[2:] -2*y[1:-1] + y[:-2]))))/(1+(2/(2+gamma*tau))-((gamma*tau)/(2+gamma*tau)));

y[0] = 0
y[L] = 0

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 25
ymax = np.array([])
plt.figure(1) # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    ynew[1:-1] = (1/(2+gamma*tau))*(4*y[1:-1] - 2*yold[1:-1] + gamma*tau*yold[1:-1] + (2*c**2 * tau**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2]))
    yold = np.copy(y);
    y = np.copy(ynew);
    
    ymax = np.append(ymax,max(y))
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 1000 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-0.03,0.03])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(0.1) # Give the computer time to draw

N1 = np.size(ymax)
t,h1 = np.linspace(0,25,N1,retstep = True)
actual = ymax[0]*np.exp(-gamma * t/2)
plt.figure(7)
plt.plot(t,ymax,'-b',t,actual,'-or') # plot x and y
plt.legend(['Numerical', 'Exact'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving PDE - Damping Term')

#%% 5.5

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
T=127;
N=200 # the number of grid points
a=0
b=L
w=400
mu=0.003

x,h = np.linspace(a,b,N+2,retstep = True) # make our grid

y = np.zeros(N+2)

#y[0] = -y[1]
#y[N+1] = -y[N]

c = np.sqrt(T/mu);
tau = 0.2*h/c;

z=y[:-2]
b=y[1:-1]
l=y[2:]

gamma = 2;
yold = np.zeros(N+2)

ynew = np.zeros_like(y)
j = 0
t = 0
tmax = 25

n = 1
fx=np.zeros(N+2);

for n in range(1,N): # right hand side
    n = n + 1
    if x[n]>=0.8 and x[n]<=1:
        fx[n]=0.73
        
plt.figure() # Open the figure window
# the loop that steps the solution along
while t < tmax:
    j = j+1
    t = t + tau
    
    #ynew[1:-1] = (2*y[1:-1]/tau**2 + yold[1:-1]*gamma*(2*tau) - yold[1:-1]/tau**2 + (fx[1:-1]/mu)*np.cos(w*t) + (c**2 / h**2) * (y[2:] -2*y[1:-1] + y[:-2]))/(1/(tau**2 + gamma))
    ynew[1:-1] = (2*y[1:-1]-yold[1:-1] + ((c**2 * tau**2)/h**2) *((y[2:] -2*y[1:-1] + y[:-2])) + yold[1:-1]*tau/2 + (fx[1:-1]/mu)*tau**2 * np.cos(w*t))/(1+gamma*tau/2)
    yold = np.copy(y);
    y = np.copy(ynew);
    
   # Use leapfrog and the boundary conditions to load
    # ynew with y at the next time step using y and yold
    # update yold and y for next timestep
    # remember to use np.copy
    # make plots every 50 time steps
    if j % 200 == 0:
        plt.clf() # clear the figure window
        plt.plot(x,y,'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t))
        plt.ylim([-0.0003,0.0003])
        plt.xlim([0,1])
        plt.draw() # Draw the plot
        plt.pause(.1) # Give the computer time to draw



