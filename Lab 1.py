# Carter Colton

# Problem 1.2

import matplotlib.pyplot as plt
import numpy as np

# a) 

N=100 # the number of grid points
a=0
b=np.pi
x,h = np.linspace(a,b,N,retstep = True) # Set up our grid 

y = np.sin(x)*np.sinh(x) # Define our function

plt.figure(1)
plt.plot(x,y) # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin and Sinh Curve')

# b) Create a new grid with 100 cells over 0 to 2

# There is always one more grid point than there is cell number.
# This is because it takes two grid points to make one cell

h = 0.02 # Cell width
x2 = np.arange(0,2,h) # Create the grid
y2 = np.cos(x2)

plt.figure(2)
plt.plot(x2,y2) # Plot it
plt.xlabel('x')
plt.ylabel('y')
plt.title('Grid From 0 to 2')

# Compare these two integration results
approx = (np.sum(y2)*h)
exact = (np.sin(2))
print('The approximate value for 2b:')
print(approx)
print('The exact value for 2b:')
print(exact)

# Find how much they are off by dividing by the exact answer
error = approx/exact
print('The percent error for 2b:')
print((error-1)*100) # Gives the percent error

# c) 

wid = np.pi/502
c = np.arange(0+wid,np.pi-wid,wid) # Create the grid
d = np.sin(c)

plt.figure(3)
plt.plot(c,d) # Plot it
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ghost Points')
#%%
# Since the ends of graphs would not be differentiable, ghost 
# Points fix that so that we can use the interior points to make
# Those end points differentiable.


# Problem 1.3

# y-y1=(y2-y1)/(x2-x1) * (x-x1)

# a) Approximate value for y(x) halfway between x1 and x2

# y(x) = y1 + (y2-y1)/(x2-x1) * (0.5*(x2-x1))
# My answer does make sense. If you subtract x1 from x2 you will
# Get the distance between them, then you times that by 1/2 and 
# Add it to x1. That is why there is no minus x1 on the right.
# You can simplify it to
# y(x) = y1 + (y2 - y1)*0.5

# b) Approximate value for y(x) 3/4 of the way from x1 to x2

# y(x) = y1 + (y2-y1)/(x2-x1) * (0.75*(x2-x1))
# You can simplify it to
# y(x) = y1 + (y2 - y1)*0.75
# I do see a pattern. You just do y1 plus y2 - y1 then times by
# The amount that you want to approximate for

# c) Linear Extrapolation

# Proof that y(x2 + h) = 2y2 - y1. Plug x2 + h in for x

# y(x) = y1 + (y2-y1)/(x2-x1) * (x2+h-x1)
# y(x) = y1 + (y2-y1)/(x2-x1) * (2x2-2x1)
# y(x) = y1 + 2(y2-y1)
# y(x) = 2y2 - y1

# Proof that y(x2 + h/2) = 3y2/2 - y1/2. Plug x2 + h/2 in for x

# y(x) = y1 + (y2-y1)/(x2-x1) * (x2 + h/2 - x1)
# y(x) = y1 + (y2-y1)/(x2-x1) * ((3/2)x2 - (3/2)x1)
# y(x) = y1 + (3/2)(y2-y1)
# y(x) = (3/2)y2 - (1/2)y1


# Problem 1.4

import scipy.special as sps

N1=20 # the number of grid points
a1=0
b1=5
x4,h1 = np.linspace(a1,b1,N1,retstep = True)
f = sps.j0(x4) # Define our bessel function

fp = np.zeros_like(f) # Create a zero array 
fp[1:N1-1] = (f[2:N1] - f[0:N1-2])/(2*h1) # Centered Difference First Derivative

fpp = np.zeros_like(f) # Create a zero array
fpp[1:N1-1] = (f[2:N1] - 2*f[1:N1-1] + f[0:N1-2])/h1**2 # Centered Difference Second Derivative

# Extrapolate to get the right start and end points
fp[0]=2*fp[1]-fp[2]
fp[N1-1]=2*fp[N1-2]-fp[N1-3]
fpp[0]=2*fpp[1]-fpp[2]
fpp[N1-1]=2*fpp[N1-2]-fpp[N1-3]

# Plot the Bessel Function and its derivatives
plt.figure(4)
plt.plot(x4,f,'r',x4,fp,'b',x4,fpp,'g')
plt.legend(['Bessel Function','First Derivative','Second Derivative'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fun with Bessel')

# Plot the exact and approximate derivatives
plt.figure(5)
plt.plot(x4,-sps.j1(x4),'r',x4,fp,'b',x4,-sps.j1(x4),'m',x4,fpp,'g')
plt.legend(['First Derivative Exact','First Derivative Approx','Second Derivative Exact', 'Second Derivative Approx'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical vs. Exact')

# Problem 1.5 - 0.1

N5=20 # the number of grid points
a5=-.1
b5=.1
x5,h5 = np.linspace(a5,b5,N5,retstep = True)
g = np.exp(x5)
h5=0.1

gfp = np.zeros_like(g) # Create a zero array
gfp[1:N5-1] = (g[2:N5] - g[1:N5-1])/h5 # Forward Difference First Derivative

gcp = np.zeros_like(g) # Create a zero array 
gcp[1:N5-1] = (g[2:N5] - g[0:N5-2])/(2*h5) # Centered Difference First Derivative

gpp = np.zeros_like(g) # Create a zero array
gpp[1:N5-1] = (g[2:N5] - 2*g[1:N5-1] + g[0:N5-2])/h5**2 # Centered Difference Second Derivative

# Extrapolate to get the right start and end points
gfp[0]=2*gfp[1]-gfp[2]
gfp[N1-1]=2*gfp[N1-2]-gfp[N1-3]
gcp[0]=2*gcp[1]-gcp[2]
gcp[N1-1]=2*gcp[N1-2]-gcp[N1-3]

# Plot the Bessel Function and its derivatives
plt.figure(6)
plt.plot(x5,g,'r',x5,gfp,'b',x5,gcp,'g')
plt.legend(['Exponent','Forward','Centered'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fun with Exponential - 0.1')

# Problem 1.5 - 0.01

N5=20 # the number of grid points
a5=-.1
b5=.1
x5,h5 = np.linspace(a5,b5,N5,retstep = True)
g = np.exp(x5)
h5=0.01

gfp = np.zeros_like(g) # Create a zero array
gfp[1:N5-1] = (g[2:N5] - g[1:N5-1])/h5 # Forward Difference First Derivative

gcp = np.zeros_like(g) # Create a zero array 
gcp[1:N5-1] = (g[2:N5] - g[0:N5-2])/(2*h5) # Centered Difference First Derivative

gpp = np.zeros_like(g) # Create a zero array
gpp[1:N5-1] = (g[2:N5] - 2*g[1:N5-1] + g[0:N5-2])/h5**2 # Centered Difference Second Derivative

# Extrapolate to get the right start and end points
gfp[0]=2*gfp[1]-gfp[2]
gfp[N1-1]=2*gfp[N1-2]-gfp[N1-3]
gcp[0]=2*gcp[1]-gcp[2]
gcp[N1-1]=2*gcp[N1-2]-gcp[N1-3]

# Plot the Bessel Function and its derivatives
plt.figure(7)
plt.plot(x5,g,'r',x5,gfp,'b',x5,gcp,'g')
plt.legend(['Exponent','Forward','Centered'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fun with Exponential - 0.01')

# Problem 1.5 - 0.001

N5=20 # the number of grid points
a5=-.1
b5=.1
x5,h5 = np.linspace(a5,b5,N5,retstep = True)
g = np.exp(x5)
h5=0.001

gfp = np.zeros_like(g) # Create a zero array
gfp[1:N5-1] = (g[2:N5] - g[1:N5-1])/h5 # Forward Difference First Derivative

gcp = np.zeros_like(g) # Create a zero array 
gcp[1:N5-1] = (g[2:N5] - g[0:N5-2])/(2*h5) # Centered Difference First Derivative

gpp = np.zeros_like(g) # Create a zero array
gpp[1:N5-1] = (g[2:N5] - 2*g[1:N5-1] + g[0:N5-2])/h5**2 # Centered Difference Second Derivative

# Extrapolate to get the right start and end points
gfp[0]=2*gfp[1]-gfp[2]
gfp[N1-1]=2*gfp[N1-2]-gfp[N1-3]
gcp[0]=2*gcp[1]-gcp[2]
gcp[N1-1]=2*gcp[N1-2]-gcp[N1-3]

# Plot the Bessel Function and its derivatives
plt.figure(8)
plt.plot(x5,g,'r',x5,gfp,'b',x5,gcp,'g')
plt.legend(['Exponent','Forward','Centered'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fun with Exponential - 0.001')

