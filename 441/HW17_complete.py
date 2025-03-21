#%% Part 1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

Vleft=0
Vright=0
Vbottom=0
Vtop=100

# Make the grid
xmin = -1
xmax = 1
Nx = 50

x = np.zeros(Nx)
for p in range(0,Nx):
    x[p] = xmin + p * ((xmax-xmin)/(Nx-1))

ymin = -1
ymax = 1
Ny = 50

y = np.zeros(Ny)
for j in range(0,Ny):
    y[j] = ymin + j * ((ymax - ymin)/(Ny-1))
    
X,Y = np.meshgrid(x,y,indexing='ij')

def defineMatricesA3(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return A

def defineMatricesB2(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return b

eps = np.ones((Nx*Nx,Ny*Ny))
Q = np.zeros(Nx*Ny)

A = defineMatricesA3(eps, Q)
B = defineMatricesB2(eps, Q)

V = np.linalg.solve(A,B)

Vnew=np.reshape(V,(Nx,Ny))

ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Vnew, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Part 2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

Vleft=0
Vright=0
Vbottom=0
Vtop=100

# Make the grid
xmin = -1
xmax = 1
Nx = 50

x = np.zeros(Nx)
for p in range(0,Nx):
    x[p] = xmin + p * ((xmax-xmin)/(Nx-1))

ymin = -1
ymax = 1
Ny = 50

y = np.zeros(Ny)
for j in range(0,Ny):
    y[j] = ymin + j * ((ymax - ymin)/(Ny-1))
    
X,Y = np.meshgrid(x,y,indexing='ij')

def defineMatricesA3(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return A

def defineMatricesB2(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return b

eps = np.ones((Nx*Nx,Ny*Ny))
Q = np.zeros(Nx*Ny)

for i in range(0,Nx):
    for j in range(0,Ny):
        if -0.3 < x[i] < 0.1 and -0.1 < y[j] < 0.3:
            eps[i,j] = 10
            
for i in range(0,Nx):
    for j in range(0,Ny):
        if np.linalg.norm(np.subtract([x[i], y[j]], [0.5,0.5])) <= 0.3:
            eps[i,j] = 4

A = defineMatricesA3(eps, Q)
B = defineMatricesB2(eps, Q)

V = np.linalg.solve(A,B)

Vnew = np.reshape(V,(Nx,Ny))

ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Vnew, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Part 3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

Vleft=0
Vright=0
Vbottom=0
Vtop=100

# Make the grid
xmin = -1
xmax = 1
Nx = 50

x = np.zeros(Nx)
for p in range(0,Nx):
    x[p] = xmin + p * ((xmax-xmin)/(Nx-1))

ymin = -1
ymax = 1
Ny = 50

y = np.zeros(Ny)
for j in range(0,Ny):
    y[j] = ymin + j * ((ymax - ymin)/(Ny-1))
    
X,Y = np.meshgrid(x,y,indexing='ij')

def defineMatricesA3(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return A

def defineMatricesB2(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return b

eps = np.ones((Nx*Nx,Ny*Ny))
Q = np.zeros(Nx*Ny)

for i in range(0,Nx):
    for j in range(0,Ny):
        if -0.3 < x[i] < 0.1 and -0.1 < y[j] < 0.3:
            eps[i,j] = 100
            
for i in range(0,Nx):
    for j in range(0,Ny):
        if np.linalg.norm(np.subtract([x[i], y[j]], [0.5,0.5])) <= 0.3:
            eps[i,j] = 1000

A = defineMatricesA3(eps, Q)
B = defineMatricesB2(eps, Q)

V = np.linalg.solve(A,B)

Vnew = np.reshape(V,(Nx,Ny))

ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Vnew, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Part 4
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
import scipy.special as sps

Vleft=0
Vright=0
Vbottom=0
Vtop=100

# Make the grid
xmin = -1
xmax = 1
Nx = 50

x = np.zeros(Nx)
for p in range(0,Nx):
    x[p] = xmin + p * ((xmax-xmin)/(Nx-1))

ymin = -1
ymax = 1
Ny = 50

y = np.zeros(Ny)
for j in range(0,Ny):
    y[j] = ymin + j * ((ymax - ymin)/(Ny-1))
    
X,Y = np.meshgrid(x,y,indexing='ij')

def defineMatricesA3(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return A

def defineMatricesB2(eps, Q):
    A = np.zeros((Nx*Nx,Ny*Ny))
    b = -Q # array of charge at each point
    
    for j in range(0,Ny):
        for i in range(0,Nx):
            ii = i + (j)*Nx # current point
            ip = (i+1) + (j)*Nx # to the right
            im = (i-1) + (j)*Nx # point to the left
            jp = i + (j+1)*Nx # point above
            jm = i + (j-1)*Nx # point below
            
            # give default values so variables
            # have the correct scope to define a0 below.
            eip = 1.0
            eim = 1.0
            ejp = 1.0
            ejm = 1.0
            if i == 0:
                # left boundary
                eim = eps[i,j]
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                b[ii] -= eim*Vleft
                A[ii,ip] = eip
            elif i == (Nx-1):
                # right boundary
                eip = eps[i,j]
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                b[ii] -= eip*Vright
                A[ii,im] = eim
            else:
                # centered in the x-direction
                eip = 0.5*(eps[i,j] + eps[i+1,j])
                eim = 0.5*(eps[i,j] + eps[i-1,j])
                A[ii,ip] = eip
                A[ii,im] = eim
            if j == 0:
                # bottom boundary
                ejm = eps[i,j]
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                b[ii] -= ejm*Vbottom
                A[ii,jp] = ejp
            elif j == (Ny-1):
                # top boundary
                ejp = eps[i,j]
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                b[ii] -= ejp*Vtop
                A[ii,jm] = ejm
            else:
                # centered in the y-direction
                ejm = 0.5*(eps[i,j] + eps[i,j-1])
                ejp = 0.5*(eps[i,j] + eps[i,j+1])
                A[ii,jm] = ejm
                A[ii,jp] = ejp
            a0 = -(eip + eim + ejp + ejm)
            A[ii,ii] = a0
    return b

eps4 = np.ones((Nx*Nx,Ny*Ny))
Q = np.zeros(Nx*Ny)
Q[1250] = -109

for i in range(0,Nx):
    for j in range(0,Ny):
        if -0.3 < x[i] < 0.1 and -0.1 < y[j] < 0.3:
            eps4[i,j] = 100
            
for i in range(0,Nx):
    for j in range(0,Ny):
        if np.linalg.norm(np.subtract([x[i], y[j]], [0.5,0.5])) <= 0.3:
            eps4[i,j] = 1000

A4 = defineMatricesA3(eps4, Q)
B4 = defineMatricesB2(eps4, Q)

V4 = np.linalg.solve(A4,B4)

Vnew4 = np.reshape(V4,(Nx,Ny))

ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Vnew4, cmap=cm.viridis)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Part 5

err = np.dot(V4,A4) - B4
print(np.sqrt(np.mean(err)))

# There is no residual because I used the matrix solver. The residual will be zero.