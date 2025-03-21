# Carter Colton

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=50
a=-1
b=1
w=0.1
x,h = np.linspace(a,b,N+1,retstep = True)

for i in range (1,N):
    x[i+1] = x[i]
    x[i+1] = np.exp(-x[i])
    
print('This is the convergence of e^-x')
print(x)

for i in range (1,N):
    x[i+1] = x[i]
    x[i+1] = -np.log(x[i])
    
print('This is the convergence of lnx')
print(x)

#%% Problem 2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = 0
xmax = 2
Nx = 80
x,hx = np.linspace(xmin,xmax,Nx,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ny = 40
y,hy = np.linspace(ymin,ymax,Ny,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = 0.5*np.ones_like(X)

# Enforce boundary conditions
V[:,0] = 0
V[:,-1] = 0
V[0,:] = 1
V[-1,:] = 1

# Allow possibility of charge distribution
rho = np.zeros_like(X)

# Iterate
denom = 2/hx2 + 2/hy2
fig = plt.figure(1)
for n in range(200):
    # make plots every few steps
    if n % 10 == 0:
        plt.clf()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X,Y,V, cmap=cm.viridis)
        ax.set_zlim(-0.1, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.draw()
        plt.pause(0.1)
# Iterate the solution
for j in range(1,Nx-1):
    for k in range(1,Ny-1):
        V[j,k] = ((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom
        
#%% Problem 4

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=50
a=-1
b=1
x,h = np.linspace(a,b,N+1,retstep = True)

def witerator(w):
    for i in range(1,N):
        x[i+1] = x[i]
        x[i+1] = w*np.exp(-x[i]) + (1-w)*x[i]
    return print(x)

#%% Test 1 Problem 4

witerator(0.01)

#%% Test 2 Problem 4

witerator(0.1)

#%% Test 3 Problem 4

witerator(0.25)

#%% Test 4 Problem 4

witerator(0.5)

#%% Test 5 Problem 4

witerator(0.75)

#%% Test 6 Problem 4

witerator(1)

#%% Test 7 Problem 4

witerator(2)

#%% Test 8 Problem 4

witerator(5)

#%% Test 9 Problem 4

witerator(0.6)

#%% Test 10 Problem 4

witerator(0.64)

#%% Problem 5

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = -2
xmax = 2
Lx = 4
Nx = 30
x,hx = np.linspace(xmin,xmax,Nx,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ly = 2
Ny = 30
y,hy = np.linspace(ymin,ymax,Ny,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = 0.5*np.ones_like(X)

# Settle all the w stuff
R = (hy**2 * np.cos(np.pi/Nx) + hx**2 * np.cos(np.pi/Ny))/(hx**2 + hy**2)
w = 2/(1+ np.sqrt(1-R**2))

# Enforce boundary conditions
V[:,0] = 0
V[:,-1] = 0
V[0,:] = 1
V[-1,:] = 1

# Allow possibility of charge distribution
rho = np.zeros_like(X)
e = np.zeros_like(V)
e[4,4] = 1
Vscale = 1

# Iterate
denom = 2/hx2 + 2/hy2
fig = plt.figure(1)
l=0
while np.amax(e) > 1/10000:
    # make plots every few steps
# Iterate the solution
    l=l+1
    for j in range(1,Nx - 1):
        for k in range(1,Ny - 1):
            V[j,k] = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            lhs = V[j,k]
            rhs = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            e[j,k] = abs((lhs - rhs)/Vscale)
            
print(l)
print(1/(12*Nx**2))

plt.clf()
ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Problem 6a

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = -2
xmax = 2
Lx = 4
Nx = 30
x,hx = np.linspace(xmin,xmax,Nx+1,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ly = 2
Ny = 30
y,hy = np.linspace(ymin,ymax,Ny+1,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = 0.5*np.ones_like(X)

# Settle all the w stuff
R = (hy**2 * np.cos(np.pi/Nx) + hx**2 * np.cos(np.pi/Ny))/(hx**2 + hy**2)
w = 2/(1+ np.sqrt(1-R**2))

# Enforce boundary conditions
V[:,0] = 0
V[:,-1] = 0
V[0,:] = -1
V[-1,:] = 1

# Allow possibility of charge distribution
rho = np.zeros_like(X)
e = np.zeros_like(V)
e[4,4] = 1
Vscale = 1

# Iterate
denom = 2/hx2 + 2/hy2
fig = plt.figure(1)
while np.amax(e) > 1/10000:
    # make plots every few steps
# Iterate the solution
    for j in range(1,Nx):
        for k in range(1,Ny):
            V[j,k] = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            lhs = V[j,k]
            rhs = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            e[j,k] = abs((lhs - rhs)/Vscale)


plt.clf()
ax = plt.axes(projection='3d')
#ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Problem 6b

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = -2
xmax = 2
Lx = 4
Nx = 30
x,hx = np.linspace(xmin,xmax,Nx+1,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ly = 2
Ny = 30
y,hy = np.linspace(ymin,ymax,Ny+1,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = 0.5*np.ones_like(X)

# Settle all the w stuff
R = (hy**2 * np.cos(np.pi/Nx) + hx**2 * np.cos(np.pi/Ny))/(hx**2 + hy**2)
w = 2/(1+ np.sqrt(1-R**2))



# Allow possibility of charge distribution
rho = np.zeros_like(X)
e = np.zeros_like(V)
e[4,4] = 1
Vscale = 1

# Iterate
denom = 2/hx2 + 2/hy2
l=0
fig = plt.figure(1)
while np.amax(e) > 1/10000:
    # make plots every few steps
# Iterate the solution
    for j in range(1,Nx):
        for k in range(1,Ny):
            
            #Enforce boundary conditions on x
            V[0,:] = -V[2,:]*(1/3) + V[1,:]*4/3
            V[-1,:] = 1
            # Enforce boundary conditions on y
            V[:,0] = 0
            V[:,-1] =-V[:,-3]*(1/3) + V[:,-2]*4/3
            V[j,k] = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            lhs = V[j,k]
            rhs = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
            e[j,k] = abs((lhs - rhs)/Vscale)
           

            
            
        

plt.clf()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(-Y,X,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('y')
plt.ylabel('x')
plt.draw()
plt.pause(0.1)

#%% Problem 6c

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = -2
xmax = 2
Lx = 4
Nx = 30
x,hx = np.linspace(xmin,xmax,Nx+1,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ly = 2
Ny = 30
y,hy = np.linspace(ymin,ymax,Ny+1,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = 0.5*np.ones_like(X)
mask = np.ones_like(V)

# Settle all the w stuff
R = (hy**2 * np.cos(np.pi/Nx) + hx**2 * np.cos(np.pi/Ny))/(hx**2 + hy**2)
w = 2/(1+ np.sqrt(1-R**2))

# Enforce boundary conditions
V[:,0] = 0
V[:,-1] = 0
V[0,:] = -1
V[-1,:] = 1

# Change some interior points
for i in range(1,6):
    mask[12+i,12] = 0
    mask[12+i,14] = 0
    mask[12+i,16] = 0
    mask[12+i,18] = 0
    
    

# Allow possibility of charge distribution
rho = np.zeros_like(X)
e = np.zeros_like(V)
e[4,4] = 1
Vscale = 1

# Iterate
denom = 2/hx2 + 2/hy2
fig = plt.figure(1)
while np.amax(e) > 1/10000:
    # make plots every few steps
# Iterate the solution
    for j in range(1,Nx):
        for k in range(1,Ny):
            if mask[j,k] == 1:
                V[j,k] = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
                lhs = V[j,k]
                rhs = (w*((V[j+1,k] + V[j-1,k])/hx2 +(V[j,k+1] + V[j,k-1])/hy2 + rho[j,k]) / denom) + (1-w)*V[j,k]
                e[j,k] = abs((lhs - rhs)/Vscale)
            else:
                V[j,k] = 0


plt.clf()
ax = plt.axes(projection='3d')
#ax = fig.gca(projection='3d')
surf = ax.plot_surface(Y,X,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('y')
plt.ylabel('x')
plt.draw()
plt.pause(0.1)
