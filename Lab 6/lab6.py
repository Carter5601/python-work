# Carter Colton

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

# Make 1D x and y arrays
Nx=30
a=0
b=2
x,hx = np.linspace(a,b,Nx,retstep = True)
Ny=50
c=-1
d=3
y,hy = np.linspace(c,d,Ny,retstep = True)

# Make the 2D grid and evaluate a function
X, Y = np.meshgrid(x,y,indexing='ij') # 'xy' flips it so that it reads Z(y,x)
Z = X**2 + Y**2

# Plot the function as a surface
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
fig.colorbar(surf)
plt.title('First 2D Plot')

# Problem 1.b


# Define our function
f = np.exp(-(X**2 + Y**2)) * np.cos(5*np.sqrt(X**2 + Y**2))

# Plot the function as a surface
fig2 = plt.figure(2)
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(X, Y, f, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
fig2.colorbar(surf2)
plt.title('Evaluating a 2D Function')

#%% Problem 2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

# Make 1D x and y arrays
Nx=50
a=-5
b=5
x,hx = np.linspace(a,b,Nx+1,retstep = True)
Ny=50
c=-5
d=5
y,hy = np.linspace(c,d,Ny+1,retstep = True)

# Make the 2D grid and evaluate a function
X, Y = np.meshgrid(x,y,indexing='ij') # 'xy' flips it so that it reads Z(y,x)
Z = np.exp(-5*(X**2 + Y**2))

sig = 2;
mu = 0.3;
tau = 0.4;

zold = np.zeros((Nx+1,Ny+1))

vz0 = np.zeros((Nx+1,Ny+1))

for j in range(1,Nx):
    for k in range(1,Ny):
        zold[j,k] = (-2*tau*vz0 + 2*Z[j,k] + (sig/mu)*tau**2 * ((Z[j+1,k]-2*Z[j,k]+Z[j-1,k])/hx**2 + (Z[j,k+1]-2*Z[j,k]+Z[j,k-1])/hy**2))/2

Z[0,0] = 0
Z[Nx,Ny] = 0

znew = np.zeros((Nx+1,Ny+1))      
tfinal=10
t=np.arange(0,tfinal,tau)
skip=10
fig = plt.figure(1)
# here is the loop that steps the solution along
for m in range(len(t)):
    # Your code to step the solution
    # make plots every skip time steps
    for j in range(1,Nx):
        for k in range(1,Ny):
                znew[j,k] = (sig/mu)*tau**2 * ((Z[j+1,k]-2*Z[j,k]+Z[j-1,k])/hx**2 + (Z[j,k+1]-2*Z[j,k]+Z[j,k-1])/hy**2) - zold[j,k] + 2*Z[j,k]
                zold = np.copy(Z);
                Z = np.copy(znew);
    
    if m % skip == 0:
        plt.clf()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z)
        ax.set_zlim(-0.5, 0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.draw()
        plt.pause(0.1)

#%% Problem 6.2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Make 1D x and y arrays
Nx=100
a=-5
b=5
x,hx = np.linspace(a,b,Nx,retstep = True)
Ny=100
c=-5
d=5
y,hy = np.linspace(c,d,Ny,retstep = True)

# Make the 2D grid and evaluate a function
X, Y = np.meshgrid(x,y,indexing='ij') # 'xy' flips it so that it reads Z(y,x)

# For problem 6.2a
#Z = np.exp(-5*(X**2 + Y**2))
#vz0 = np.zeros((Nx,Ny))

# For problem 6.2d
Z = np.zeros((Nx,Ny))
vz0 = np.exp(-5*(X**2 + Y**2))

sig = 2;
mu = 0.3;
s = np.sqrt(sig/mu)
tau = 0.707*hx/s;

zold = np.zeros((Nx,Ny))

zold[1:-1,1:-1] = Z[1:-1,1:-1] + (sig/mu)*tau**2 * \
                   ((Z[2:,1:-1]-2*Z[1:-1,1:-1]+Z[:-2,1:-1])/hx**2 + \
                    (Z[1:-1,2:]-2*Z[1:-1,1:-1]+Z[1:-1,:-2])/hy**2)/2 - 2*tau*vz0[1:-1,1:-1]

znew = np.zeros((Nx,Ny))  

tfinal=10
t=np.arange(0,tfinal,tau)
zcen = np.zeros(len(t))
skip=50
fig = plt.figure()
# here is the loop that steps the solution along
for m in range(len(t)):
    # Your code to step the solution
    # make plots every skip time steps
    znew[1:-1,1:-1] = (sig/mu)*tau**2 * ((Z[2:,1:-1]-2*Z[1:-1,1:-1]+Z[:-2,1:-1])/hx**2 \
                    + (Z[1:-1,2:]-2*Z[1:-1,1:-1]+Z[1:-1,:-2])/hy**2) - zold[1:-1,1:-1] + \
                    2*Z[1:-1,1:-1] #(2-4*(sig/mu)*(tau**2/hx**2))*Z[1:-1,1:-1]
    zold = np.copy(Z);
    Z = np.copy(znew);
    zcen[m] = znew[50,50] 
    
    if m % skip == 0:
        plt.clf()
        ax = plt.axes(projection='3d')
        # ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z, cmap=cm.viridis)
        ax.set_zlim(-0.5, 0.5)
        #plt.contourf(X,Y,Z)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.draw()
        plt.title('Solving 2D Wave Equation - Tau = 0.707')
        #plt.show()
        plt.pause(0.1)

plt.figure(5)
plt.plot(t,zcen,'-b') # plot x and y
plt.xlabel('t')
plt.ylabel('z centered')
plt.title('Z Centered for 2D Wave Equation')