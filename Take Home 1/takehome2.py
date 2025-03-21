# Carter Colton

#%% Problem 2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=3.2;
N=30 # the number of grid points
a=0
b1=L

# I set the w to be 400. That is what we have been setting it to in most cases.
w=400;
mu0=0.003;
m=0.003
T=50;
kx=16;

# Build our array for the displacement in x
x,h = np.linspace(a,b1,N+1,retstep = True) # make the grid
# We have a variable mu
mu=mu0*np.exp(x/L);

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)]) # build an array with values of A

c=1; # create iterators 
d=0;

yprime = (3/(2*h)+(2/h)+(1/2*h)); # define y'

# Do a better extrapolation
A[0,0] = 1; # My code only works when this is 1. I don't know why
A[N,N] = 1/h; # Divide the T and m over to the B matrix bc's
A[N,N-1] = -1/h

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1] # be setting d+1 I make it move 1 more column over,
    A[c,d+2] = s[2] # effectively creating a diagonal A matrix
    c = c + 1
    d = d + 1

b = np.zeros((N+1,N+1)) # load b with zeros
b[0,0] = 0;
b[N,N] = -m/2; # The T and w are swallowed up in the lambda. 
b[N,N-1] = -m/2; # We are left with m/2 for the bc.

k=1 #create iterators
j=1
z=0

for i in b[1:N,0:N]: # build our matrix
    b[j,k]= mu[z] # incorporate the variable mu
    j=j+1
    k=k+1
    z=z+1

vals,vecs = la.eig(A,b)

# Compute the eigen-frequencies
w = np.sqrt(-T*np.real(vals))
# Sort the eigenvalues and eigenvectors
ind = np.argsort(w)
w=w[ind]
vecs = vecs[:,ind]

#%% First 3

for i in range(0,3): # plot the function in for loop so we get first 3 eigenfunctions
    g=vecs[:,i]
    plt.figure()
    plt.plot(x,g,'b') # plot x and g
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['g(x)'])
    plt.title('Fun With Eigen n={:1.0f},w={:1.3f}'.format(i,w[i])) # put n and w on graph
