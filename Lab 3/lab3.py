# Carter Colton 

#%% Problem 3.2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
N=100 # the number of grid points
a=0
b=L

w=400;
mu=0.003;
T=127;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([T/(h**2),-2*T/(h**2) + mu*w**2,T/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Set the endpoints
A[N,N] = 1;

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros(N+1) # load b with zeros
b[0] = 0;
b[N] = 0;
n = 1
f=0;

for n in range(1,N): # right hand side
    b[n] = f
    n = n + 1
    if x[n]>=0.8 and x[n]<=1:
        f=-0.73
    else:
        f=0
print(b)     
yn = la.solve(A,b) # solve for y

plt.figure(1)
plt.plot(x,yn,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['g(x)'])
plt.title('Solving The Wave Equation')

# Problem 3.2b

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps
import Lab3Funcs as l3

N=200
a=400
b=1700
mu=0.003;
T=127;

wplot,dx = np.linspace(a,b,N+1,retstep = True) # make the grid
f=l3.force(x)
gmax=np.zeros(N+1)

plt.figure(2)
for n in range(len(wplot)):
    w = wplot[n]
    g = l3.steadySol(f,h,w,T,mu)
    gmax[n]=np.amax(g)
    plt.clf() # Clear the previous plot
    plt.plot(x,g)
    plt.title('$\omega={:1.2e}$'.format(w))
    plt.xlabel('x')
    plt.ylim([-0.05, 0.05]) # prevent auto-scaling
    plt.draw() # Request to draw the plot now
    plt.pause(0.1) # Give the computer time to draw it
    
#%% Problem 3.2c

plt.figure(3)
plt.plot(wplot,gmax,'b') # plot x and y
plt.xlabel('w')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Maximum Amplitude')

#%% Problem 3.3
    
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
N=30 # the number of grid points
a=0
b=L

w=400;
mu=0.003;
T=127;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Set the endpoints
A[N,N] = 1;

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros((N+1,N+1)) # load b with zeros
b[0,0] = 0;
b[N,N] = 0;
n = 1

for i in b[1:N]:
    b[n] = 0
    n = n + 1

k=1
j=1

for i in b[1:N,0:N]: # build our matrix
    b[j,k]=1
    j=j+1
    k=k+1

vals,vecs = la.eig(A,b)

#%% First 3

for i in range(0,3):
    g=-vecs[:,i]
    plt.figure()
    plt.plot(x,g,'b') # plot x and y
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['g(x)'])
    plt.title('Fun With Eigen')
#%% All 20

for i in range(0,N):
    g=vecs[:,i]
    plt.figure()
    plt.plot(x,g,'b') # plot x and y
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['g(x)'])
    plt.title('Fun With Eigen')

#%% Problem 3.3b
    
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
N=30 # the number of grid points
a=0
b=L

w=400;
mu=0.003;
T=127;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Set the endpoints
A[N,N] = 1;

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros((N+1,N+1)) # load b with zeros
b[0,0] = 0;
b[N,N] = 0;
n = 1

for i in b[1:N]:
    b[n] = 0
    n = n + 1

k=1
j=1

for i in b[1:N,0:N]: # build our matrix
    b[j,k]=1
    j=j+1
    k=k+1

vals,vecs = la.eig(A,b)

for i in range(0,3):
    wcal=-(i+1)*np.pi/L *np.sqrt(T/mu)
    g=vecs[:,i]
    exg=np.sin(wcal*np.sqrt(mu/T)*x)
    plt.figure()
    plt.plot(x,exg,'r',x,g,'b') # plot x and y
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['Exact','Calculated'])
    plt.title('Fun With Eigen')

# Compute the eigen-frequencies
w = np.sqrt(-T*np.real(vals)/mu)
# Sort the eigenvalues and eigenvectors
ind = np.argsort(w)
w=w[ind]
vecs = vecs[:,ind]

for i in range(0,20):  
  g=vecs[:,i]
  wcal=(i+1)*np.pi/L *np.sqrt(T/mu)
  exg=np.sin(wcal*np.sqrt(mu/T)*x)
  print('This is calculated')
  print(wcal)
  print('This is actual')
  print(w[i])
  plt.figure()
  plt.plot(x,exg,'r',x,g,'b') # plot x and y
  plt.xlabel('x')
  plt.ylabel('g')
  plt.legend(['Exact','Calculated'])
  plt.title('Fun With Eigen')


#%% Problem 3.3c

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=100
a=400
b=1700
mu=0.003;
T=127;

# Compute the eigen-frequencies
weig = np.sqrt(-T*np.real(vals)/mu)
# Sort the eigenvalues and eigenvectors
ind = np.argsort(weig)
weig=weig[ind]
vecs = vecs[:,ind]
print(weig[0])
#for n in range(len(wplot)):
 #   w = wplot[n]
  #  g = l3.steadySol(f,h,w,T,mu)
   # gmax[n]=np.amax(g)

    
# Problem 3.3c
plt.figure(3)
plt.plot(wplot,gmax,'b') # plot x and y
plt.xlim(weig[0]-5,weig[0])
plt.xlabel('w')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Maximum Amplitude')

#%% Problem 3.4

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
N=30 # the number of grid points
a=0
b=L

w=400;
mu=0.003;
T=127;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Do a better extrapolation
A[N,N-1] = 3/(2*h);
A[N,N-2] = -2/h;
A[N,N-3] = 1/(2*h)

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros((N+1,N+1)) # load b with zeros
b[0,0] = 0;
b[N,N] = 0;
n = 1

for i in b[1:N]:
    b[n] = 0
    n = n + 1

k=1
j=1

for i in b[1:N,0:N]: # build our matrix
    b[j,k]=1
    j=j+1
    k=k+1

vals,vecs = la.eig(A,b)

# Compute the eigen-frequencies
w = np.sqrt(-T*np.real(vals)/mu)
# Sort the eigenvalues and eigenvectors
ind = np.argsort(w)
w=w[ind]
vecs = vecs[:,ind]

g=-vecs[:,0]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=0')

g=vecs[:,1]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=1')

g=vecs[:,2]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=2')

#%% Problem 3.4b

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=1.2;
N=30 # the number of grid points
a=0
b=L

w=400;
mu=0.003;
T=127;

x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Do a better extrapolation
A[N,N-1] = 3/(2*h) -2;
A[N,N-2] = -2/h;
A[N,N-3] = 1/(2*h)

for i in A[1:N,0:N]: # build our matrix
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros((N+1,N+1)) # load b with zeros
b[0,0] = 0;
b[N,N] = 0;
n = 1

for i in b[1:N]:
    b[n] = 0
    n = n + 1

k=1
j=1

for i in b[1:N,0:N]: # build our matrix
    b[j,k]=1
    j=j+1
    k=k+1

vals,vecs = la.eig(A,b)
# Compute the eigen-frequencies
w = np.sqrt(-T*np.real(vals)/mu)
# Sort the eigenvalues and eigenvectors
ind = np.argsort(w)
w=w[ind]
vecs = vecs[:,ind]

g=vecs[:,0]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=0')

g=vecs[:,1]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=1')

g=-vecs[:,2]
plt.figure()
plt.plot(x,g,'b') # plot x and y
plt.xlabel('x')
plt.ylabel('g')
plt.legend(['g(x)'])
plt.title('Fun With Eigen - n=2')
