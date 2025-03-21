# Carter Colton

#%% Problem 4.2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=30 # the number of grid points
a=0
L=2
x,h = np.linspace(a,L,N+1,retstep = True) # make our grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([x/(h**2) - 1/(2*h),-(2*x)/(h**2),x/(h**2) + 1/(2*h)])
c=1;
d=0;

j=1

for i in A[1:N,0:N]: # build our matirx
    xnew = x[j]
    A[c,d] = xnew/(h**2) - 1/(2*h)
    A[c,d+1] = -(2*xnew)/(h**2)
    A[c,d+2] = xnew/(h**2) + 1/(2*h)
    c = c + 1
    d = d + 1
    j = j + 1
    
A[0,0] = -1/h; # set our endpoints
A[0,1] = 1/h;
A[N,N-1] = 1/2;
A[N,N] = 1/2;

b = np.zeros((N+1, N+1)) # load b with zeros
n = 1

k=1
o=1

for i in b[1:N,0:N]: # build our matrix
    b[o,k]=1
    o=o+1
    k=k+1

b[0,0] = 1/2;
b[0,1] = 1/2;
b[N,N] = 0;

vals,vecs = la.eig(A,b) # solve for y

for i in range(0,3):
    g=-vecs[:,i]
    plt.figure()
    plt.plot(x,g,'b') # plot x and y
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['g(x)'])
    plt.title('Solving Differential Equation Analytically - Hanging Chain')

y = (sps.j0(2*np.sqrt(x/9.8))) # Define our function
y[2]=0;

wnum = np.zeros(N+1)
w = np.zeros(N+1)

for i in range(0,N):
    wnum[i] = np.sqrt(-9.8*np.real(vals[i]))

w = (sps.jn_zeros(0,N+1)/2) * np.sqrt(9.8/2);

xnew = list(range(1,N+2))

plt.figure(5)
plt.plot(xnew,w,'-b',xnew,wnum,'-r') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Exactly - Bessel')

#%% Problem 4.4

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=100 # the number of grid points
a=-5
b=5
e,h = np.linspace(a,b,N+1,retstep = True) # make our grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([-1/(2*(h**2)),1/(h**2) + (1/2)*e**2,-1/(2*(h**2))])
c=1;
d=0;
j=1

for i in A[1:N,0:N]: # build our matirx
    enew = e[j]
    A[c,d] = -1/(2*(h**2))
    A[c,d+1] = 1/(h**2) + (1/2)*enew**2
    A[c,d+2] = -1/(2*(h**2))
    c = c + 1
    d = d + 1
    j = j + 1
    
A[0,0] = 1/2; # set our endpoints
A[0,1] = 1/2;
A[N,N-1] = 1/2;
A[N,N] = 1/2;

b = np.zeros((N+1, N+1)) # load b with zeros
n = 0

k=1
o=1

for i in b[1:N,0:N]: # build our matrix
    b[o,k] = 1
    o=o+1
    k=k+1
    n=n+1

b[0,0] = 0;
b[0,1] = 0;
b[N,N] = 0;

vals,vecs = la.eig(A,b) # solve for y
rvals = np.real(vals)
ind = np.argsort(rvals)
rvals=rvals[ind]
vecs = vecs[:,ind]

for i in range(0,5):
    g=-vecs[:,i]
    gnorm = np.trapz(g**2,e)
    print(rvals[i])
    plt.figure()
    plt.plot(e,g**2/gnorm,'b') # plot x and y
    plt.xlabel('e')
    plt.ylabel('phi')
    plt.title('Schrodinger - 4.4 i={} $\epsilon_{}={:1.2f}$'.format(i,i,rvals[i]))

#%% Problem 4.5

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=100 # the number of grid points
a=-5
b=5
e,h = np.linspace(a,b,N+1,retstep = True) # make our grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([-1/(2*(h**2)),1/(h**2) + e**4,-1/(2*(h**2))])
c=1;
d=0;
j=1

for i in A[1:N,0:N]: # build our matirx
    enew = e[j]
    A[c,d] = -1/(2*(h**2))
    A[c,d+1] = 1/(h**2) + enew**4
    A[c,d+2] = -1/(2*(h**2))
    c = c + 1
    d = d + 1
    j = j + 1
    
A[0,0] = 1/2; # set our endpoints
A[0,1] = 1/2;
A[N,N-1] = 1/2;
A[N,N] = 1/2;

b = np.zeros((N+1, N+1)) # load b with zeros
n = 0

k=1
o=1

for i in b[1:N,0:N]: # build our matrix
    b[o,k] = 1
    o=o+1
    k=k+1
    n=n+1

b[0,0] = 0;
b[0,1] = 0;
b[N,N] = 0;

vals,vecs = la.eig(A,b) # solve for y
rvals = np.real(vals)
ind = np.argsort(rvals)
rvals=rvals[ind]
vecs = vecs[:,ind]

for i in range(0,5):
    g=-vecs[:,i]
    gnorm = np.trapz(g**2,e)
    print(rvals[i])
    plt.figure()
    plt.plot(e,g**2/gnorm,'b') # plot x and y
    plt.xlabel('e')
    plt.ylabel('phi')
    plt.title('Schrodinger - 4.5 i={} $\epsilon_{}={:1.2f}$'.format(i,i,rvals[i]))
