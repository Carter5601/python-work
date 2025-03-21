#%% Section 1
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

#%% Section 2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
function [x,y,V,charge,totalCharge]=laplaceOptimPlates

tic;
xgrid=200;
ygrid=200;

#create x,y mesh
x=linspace(-1,1,xgrid);
y=linspace(-1,1,ygrid);
[X,Y]=meshgrid(x,y);

#set boundary conditions
leftBC=0;
rightBC=0;
topBC=0;
bottomBC=0;

V=zeros(xgrid,ygrid); #creates a potential matrix of all zeros
#V=1-sqrt(X.^2+Y.^2);


#set the appropriate boundary conditions
V[1,:]=leftBC;
V[end,:]=rightBC;
V[:,1]=bottomBC;
V[:,end]=topBC;

#define the location of the left plate
xL1=round(xgrid*.35);
xL2=round(xgrid*.65);
yL1=round(ygrid*.45);
yL2=round(ygrid*.451);

#define the location of the right plate
xR1=round(xgrid*.35);
xR2=round(xgrid*.65);
yR1=round(ygrid*.55);
yR2=round(ygrid*.551);

maxIterations=2000; #The maximum number of times the iteration procedure will run

change=0;

#can use waitbar to show progress, but it increases runtime dramatically
#h=waitbar(0, sprintf('%1.0e',change),'Name', 'Change is ');
i=1;
for i in maxIterations:
    Vold=V; #
    
    #establish V matrices shifted in each direction needed for vectorization
    Vdown=circshift(V,[1,0]);
    Vup=circshift(V,[-1,0]);
    Vright=circshift(V,[0,1]);
    Vleft=circshift(V,[0,-1]);
    
    #This averaging step solves Laplace's equation
    V=(Vdown+Vup+Vright+Vleft)/4;

    #re-establish the set potential of the plates
    V[xL1:xL2,yL1:yL2]=1; 
    V[xR1:xR2,yR1:yR2]=-1;
    
    #maintain the boundary conditions
    V[1,:]=leftBC;
    V[end,:]=rightBC;
    V[:,1]=bottomBC;
    V[:,end]=topBC;
    
    #track the change in the matrix at each step to know when to quit
    deltaV=V-Vold;
    change=abs(sum(sum(deltaV))/(xgrid*ygrid));
    if i>1000 && change<1e-11:
        break
    
    #waitbar(i/maxIterations,h,sprintf('%1.0e',change));
    
    #conditional to quit the method when changes are within tolerance
    end
end

disp('Number of iterations')
disp(i);

disp('change parameter at final step')
disp(change);


%close(h)
figure
meshc(X,Y,V);




plt.clf()
ax = plt.axes(projection='3d')
#ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)

#%% Initial Potential

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# Make the grid
xmin = -2
xmax = 2
Lx = 4
Nx = 100
x,hx = np.linspace(xmin,xmax,Nx+1,retstep = True)
hx2 = hx**2

ymin = 0
ymax = 2
Ly = 2
Ny = 100
y,hy = np.linspace(ymin,ymax,Ny+1,retstep = True)
hy2 = hy**2
X,Y = np.meshgrid(x,y,indexing='ij')

# Initialize potential
V = np.zeros_like(X)

# Enforce boundary conditions
V[:,0] = 0
V[:,-1] = 0
V[0,:] = 0
V[-1,:] = 0
V[40,30:70] = 100
V[60,30:70] = -100

plt.clf()
ax = plt.axes(projection='3d')
#ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,V, cmap=cm.viridis)
ax.set_zlim(-0.1, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.pause(0.1)