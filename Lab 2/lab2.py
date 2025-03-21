
# Carter Colton

#%% Problem 2.2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=30 # the number of grid points
a=0
b=2
x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

y = (16*np.sin(3*x) + np.cos(6 - x) - np.cos(6 + x) + np.cos(2 + 3*x) - 
     np.cos(2 - 3*x))/(16*np.sin(6))  # Define our function

plt.figure(1)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically')

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2) + 9,1/(h**2)])
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
b[N] = 1;
n = 1

for i in b[1:N]: # right hand side
    b[n] = np.sin(x[n])
    n = n + 1

yn = la.solve(A,b) # solve for y

plt.figure(2)
plt.plot(x,y,'-b',x,yn,'or') # plot x and y
plt.legend(['Exact','Numerical'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - sin(x)')
    
#%% Problem 2.3

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=30 # the number of grid points
a=0
b=5
x,h = np.linspace(a,b,N+1,retstep = True) # make our grid

y = -4/(sps.j1(5)) *sps.j1(x) + x # Define our function

plt.figure(3)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically')

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2) + 1/(2*x*h),-2/(h**2) + 1 -1/x**2,1/(h**2) - 1/(2*x*h)])
c=1;
d=0;

A[0,0] = 1; # set our endpoints
A[N,N] = 1;

j=1

for i in A[1:N,0:N]: # build our matirx
    xnew = x[j]
    A[c,d] = 1/(h**2) - 1/(2*xnew*h)
    A[c,d+1] = -2/(h**2) + 1 - 1/xnew**2
    A[c,d+2] = 1/(h**2) + 1/(2*xnew*h)
    c = c + 1
    d = d + 1
    j = j + 1
    
b = np.zeros(N+1) # load b with zeros
b[0] = 0;
b[N] = 1;
n = 1

for i in b[1:N]:
    b[n] = np.copy(x[n])
    n = n + 1

yn = la.solve(A,b) # solve for y

plt.figure(4)
plt.plot(x,y,'-b',x,yn,'or') # plot x and y
plt.legend(['Exact','Numerical'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - Bessel')

# Problem 2.3 b

N=2600 # the number of grid points
a=0
b=5
x,h = np.linspace(a,b,N+1,retstep = True)

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2) + np.sin(x)/(2*h),-2/(h**2) + np.exp(x),1/(h**2) - np.sin(x)/(2*h)])
c=1;
d=0;

A[0,0] = 1; # reset the endpoints
A[N,N] = 1;

j=1

for i in A[1:N,0:N]: # re build the matrix
    xnew = x[j]
    A[c,d] = 1/(h**2) - np.sin(xnew)/(2*h)
    A[c,d+1] = -2/(h**2) + np.exp(xnew)
    A[c,d+2] = 1/(h**2) + np.sin(xnew)/(2*h)
    c = c + 1
    d = d + 1
    j = j + 1
    
b = np.zeros(N+1) # load our b matrix with zeros
b[0] = 0;
b[N] = 3;
n = 1

for i in b[1:N]:
    b[n] = (x[n])**2
    n = n + 1

yn = la.solve(A,b) # solve for y

plt.figure(7)
plt.plot(x,yn,'-r') # plot x and y
plt.legend(['Numerical'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - x^2')

k = int(4.5/h)
print(k)
print(yn[k])

#%% Problem 2.4

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=75 # the number of grid points
a=0
b=2
x,h = np.linspace(a,b,N+1,retstep = True)

y = x/9 - np.sin(3*x)/(27*np.cos(6))  # Define our function

plt.figure(5)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically')

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2) + 9,1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Set the endpoints
A[N,N-1] = 1/h; # Add the additional extrapolation
A[N,N-2] = -1/h;

for i in A[1:N,0:N]: # build A
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros(N+1) # load b with zeros
b[0] = 0;
b[N] = 0;
n = 1

for i in b[1:N]:
    b[n] = np.copy(x[n])
    n = n + 1

#print(A)
#print(b)
yncrude = la.solve(A,b) # solve for y

plt.figure(6)
plt.plot(x,y,'-b',x,yncrude,'or') # plot x and y
plt.legend(['Exact','Numerical'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - 2.4 - x')

# Problem 2.4b

 # the number of grid points
a=0
b=2


y = x/9 - np.sin(3*x)/(27*np.cos(6))  # Define our function

plt.figure(5)
plt.plot(x,y,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically')

A = np.zeros((N+1,N+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2) + 9,1/(h**2)])
c=1;
d=0;

A[0,0] = 1; # Do a better extrapolation
A[N,N-1] = 3/(2*h);
A[N,N-2] = -2/h;
A[N,N-3] = 1/(2*h)

for i in A[1:N,0:N]: # build A
    A[c,d] = s[0]
    A[c,d+1] = s[1]
    A[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros(N+1)
b[0] = 0;
b[N] = 0;
n = 1

for i in b[1:N]:
    b[n] = np.copy(x[n])
    n = n + 1

#print(A)
#print(b)
ynexact = la.solve(A,b) # solve for a better y

plt.figure(8)
plt.plot(x,y,'-b',x,ynexact,'or') # plot x and y
plt.legend(['Exact','Numerical'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solving Differential Equation Analytically - 2.4b - x')

err = np.sqrt(np.mean((yncrude-ynexact)**2)) # do the error

print(err)

#%% Problem 2.5

# You cannot take the y(x) out of the sine to get the matrix A.
# Therefore, it does not work.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N1=30 # the number of grid points
a=0
b1=3
x,h = np.linspace(a,b1,N1+1,retstep = True)  # Define our function

A1 = np.zeros((N1+1,N1+1)) # load a matrix A with zeros
s = np.array([1/(h**2),-2/(h**2),1/(h**2)])
c=1;
d=0;

A1[0,0] = 1; # set the endpoints
A1[N1,N1] = 1;

for i in A1[1:N1,0:N1]: # bild the A matrix
    A1[c,d] = s[0]
    A1[c,d+1] = s[1]
    A1[c,d+2] = s[2]
    c = c + 1
    d = d + 1
    
b = np.zeros(N1+1) # load b with zeros
b[0] = 0;
b[N1] = 0;
n = 1

yn4=np.zeros(N1)
rmserr=1

while rmserr > (1*10**(-5)): # set a while loop to break once we reach the right accuracy
    for n in range(1,N1):
        b[n] = 1 - np.sin(yn4[n])
    yn4 = la.solve(A1,b)
    err = A1@yn4-(1-np.sin(yn4))
    rmserr = np.sqrt(np.mean(err[1:-2]**2))
   

print(yn4)

plt.figure(9)
plt.plot(x,yn4,'-b') # plot x and y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-Linear Iterations')




